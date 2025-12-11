import os
import pickle
import argparse
from absl import app, flags
from ml_collections import config_flags

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import cv2
import pygame 

# Assumes these modules exist in your project structure
from utils_env import create_stateenv, config_state2
from utils.flax_utils import restore_agent
from agents import agents

FLAGS = flags.FLAGS

# ---------- Flags ----------
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('env_name', 'highway-state', 'Environment name.')
config_flags.DEFINE_config_file('agent', 'agents/acfql.py', lock_config=False)

flags.DEFINE_string('ckpt_dir', None, 'Directory containing params_*.pkl')
flags.DEFINE_integer('ckpt_step', None, 'Checkpoint step to load.')

flags.DEFINE_string('classifier_path', 'data/h5_lane_classifier.pkl', 'Path to trained LogisticRegression pickle.')

# Intervention Settings
flags.DEFINE_float('p_val', 0.7, 'Target probability for auto-strength calculation.')
flags.DEFINE_float('fixed_strength', None, 'If set, overrides auto-strength with this constant value.')

flags.DEFINE_integer('eval_episodes', 10, 'Number of evaluation episodes.')
flags.DEFINE_integer('max_steps_per_episode', 2000, 'Safety cap on episode length.')
flags.DEFINE_integer('horizon_length', 9, 'Action chunk length.')
flags.DEFINE_bool('video', False, 'Whether to record video.')
flags.DEFINE_string('video_dir', 'videos/intervention/manual_keyboard', 'Directory for videos.')


# ---------- Wrappers ----------

class TextOverlayWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.current_text = ""
    
    def set_text(self, text):
        self.current_text = text

    def render(self):
        # In human mode, highway-env renders to pygame window.
        frame = self.env.render()
        
        if frame is None or not isinstance(frame, np.ndarray):
            return frame

        frame = frame.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        color = (255, 255, 255)
        thickness = 1
        line_type = cv2.LINE_AA
        
        lines = self.current_text.split('\n')
        x, y = 20, 40
        line_height = 25
        
        for line in lines:
            # Outline
            cv2.putText(frame, line, (x, y), font, font_scale, (0, 0, 0), thickness + 2, line_type)
            # Text
            cv2.putText(frame, line, (x, y), font, font_scale, color, thickness, line_type)
            y += line_height
        
        return frame

class FlattenObsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        orig_shape = env.observation_space.shape
        flat_dim = int(np.prod(orig_shape))
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(flat_dim,), dtype=np.float32,
        )

    def observation(self, observation):
        return np.asarray(observation, dtype=np.float32).reshape(-1)

def make_env_flat_with_overlay(video=False, video_folder=None, video_name_prefix='intervention'):
    config_state2["offscreen_rendering"] = False
    config_state2["policy_frequency"] = 10
    config_state2["simulation_frequency"] = 50
    env = create_stateenv(config_state2)
    
    text_wrapper = TextOverlayWrapper(env)
    env = text_wrapper
    
    if video:
        os.makedirs(video_folder, exist_ok=True)
        env = RecordVideo(
            env,
            video_folder=video_folder,
            name_prefix=video_name_prefix,
            episode_trigger=lambda ep_id: True,
            disable_logger=True,
        )
    env = FlattenObsWrapper(env)
    return env, text_wrapper

# ---------- Intervention Logic ----------

def load_classifier(classifier_path):
    with open(classifier_path, 'rb') as f:
        clf = pickle.load(f)
    return clf

def get_single_direction_vectors(clf, direction):
    """Calculates the Base and Orthogonal vectors for a specific direction."""
    if direction == 'left':
        target_idx = 1 # Assumes class 1 is Left
    elif direction == 'right':
        target_idx = 2 # Assumes class 2 is Right
    else:
        return None, None, None

    # Get indices in the coefficient matrix
    try:
        class_index_in_matrix = np.where(clf.classes_ == target_idx)[0][0]
        straight_index_in_matrix = np.where(clf.classes_ == 0)[0][0] 
    except IndexError:
        print(f"Error: Could not find class index for direction {direction}")
        return None, None, None

    weight_vector = clf.coef_[class_index_in_matrix]
    straight_vector = clf.coef_[straight_index_in_matrix]

    # Gram-Schmidt to make orthogonal to Straight direction
    proj_length = np.dot(weight_vector, straight_vector) / np.dot(straight_vector, straight_vector)
    proj_vector = proj_length * straight_vector
    ortho_weight_vector = weight_vector - proj_vector

    # Normalize
    weight_vector = weight_vector / np.linalg.norm(weight_vector)
    ortho_weight_vector = ortho_weight_vector / np.linalg.norm(ortho_weight_vector)
    
    return jnp.array(weight_vector), jnp.array(ortho_weight_vector), class_index_in_matrix

@jax.jit
def intervened_action_step(params, obs, noise, base_perturbation_vector, ortho_perturbation_vector,
                           clf_w, clf_b, target_class_idx, opp_class_idx, stay_class_idx,
                           apply_intervention, p_val, fixed_strength, use_fixed):
    """
    Standard JAX intervention step. 
    apply_intervention (bool): Whether to add the perturbation.
    """
    x = jnp.concatenate([obs, noise], axis=-1)
    mlp_params = params['modules_actor_onestep_flow']['mlp']
    
    # --- 1. Forward to Feature ---
    for i in range(4): # Dense_0 to Dense_3
        k = mlp_params[f'Dense_{i}']['kernel']
        b = mlp_params[f'Dense_{i}']['bias']
        x = nn.gelu(x @ k + b)
    
    feature = x 
    
    # --- 2. Compute Probs (Softmax) ---
    logits = feature @ clf_w.T + clf_b  
    probs = nn.softmax(logits)          
    
    # --- 3. Compute Strength ---
    def _calc_complex(operands):
        _logits, _base, _ortho, _t_idx, _o_idx, _s_idx, _p = operands
        C = _p / (1.0 - _p)
        L_target = _logits[0, _t_idx]
        L_opp    = _logits[0, _o_idx]
        L_stay   = _logits[0, _s_idx]
        
        term1 = C * jnp.exp(L_opp)
        term2_inside = (C**2) * jnp.exp(2 * L_opp) + 4 * C * jnp.exp(L_target + L_stay)
        term2 = jnp.sqrt(term2_inside)
        
        numerator = term1 + term2
        shift_val = jnp.log(numerator) - jnp.log(2.0) - L_target
        denom = jnp.dot(_base, _ortho)
        return shift_val / (denom + 1e-6)

    def _return_fixed(operands):
        return fixed_strength

    # Logic for Auto: If Straight -> Simple (0.5), Else -> Complex
    # Since we only do L/R here, we mostly rely on complex, but we keep structure
    def _calc_auto(operands):
        return _calc_complex(operands)

    operands = (logits, base_perturbation_vector, ortho_perturbation_vector, 
                target_class_idx, opp_class_idx, stay_class_idx, p_val)

    strength_to_use = jax.lax.cond(
        use_fixed,
        _return_fixed,
        _calc_auto,
        operands
    )
    strength_to_use = jnp.maximum(strength_to_use, 0.0)
    
    # --- 4. Apply Perturbation ---
    mask = apply_intervention.astype(jnp.float32)
    total_perturbation = ortho_perturbation_vector * strength_to_use * mask
    x_intervened = feature + total_perturbation

    # --- 5. Project to Action ---
    k = mlp_params['Dense_4']['kernel']
    b = mlp_params['Dense_4']['bias']
    out = x_intervened @ k + b
    action = jnp.clip(out, -1, 1)
    
    return action, feature, probs, logits, strength_to_use

# ---------- Interactive Loop ----------

def run_interactive_episodes(agent, env, text_wrapper, clf, rng, horizon_length, num_episodes, max_steps, 
                             intervention_data, p_val, fixed_strength_val):
    
    action_dim = env.action_space.shape[-1]
    noise_dim = agent.config['action_dim'] * (agent.config['horizon_length'] if agent.config["action_chunking"] else 1)
    
    clf_w = jnp.array(clf.coef_)       
    clf_b = jnp.array(clf.intercept_)
    idx_stay = int(np.where(clf.classes_ == 0)[0][0])
    idx_left = int(np.where(clf.classes_ == 1)[0][0])
    idx_right = int(np.where(clf.classes_ == 2)[0][0])
        
    use_fixed_strength = (fixed_strength_val is not None)
    f_val = float(fixed_strength_val) if use_fixed_strength else 0.0
    
    pygame.init()
    clock = pygame.time.Clock()
    
    print("\n--- INTERACTIVE MODE START ---")
    print("Controls: Tap LEFT or RIGHT to queue an intervention.")
    print("Behavior: The last key pressed determines the NEXT chunk.")
    
    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        ep_len = 0
        action_queue = []
        chunk_counter = 0
        
        # Latch variable
        pending_intervention = None 
        
        while not done and ep_len < max_steps:
            # --- 1. Event Listener (The Latch) ---
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        pending_intervention = 'left'
                    elif event.key == pygame.K_RIGHT:
                        pending_intervention = 'right'

            # --- 2. Chunk Generation ---
            if len(action_queue) == 0:
                chunk_counter += 1
                rng, key = jax.random.split(rng)
                
                # A. Determine Direction
                target_dir = pending_intervention
                
                if target_dir is None:
                    keys = pygame.key.get_pressed()
                    if keys[pygame.K_LEFT]: target_dir = 'left'
                    elif keys[pygame.K_RIGHT]: target_dir = 'right'

                # B. Prepare Intervention
                if target_dir and target_dir in intervention_data:
                    vec_data = intervention_data[target_dir]
                    base_vec = vec_data['base']
                    ortho_vec = vec_data['ortho']
                    t_idx = vec_data['target_idx']
                    o_idx = idx_right if target_dir == 'left' else idx_left
                    
                    do_intervene = True
                    # Append direction to status text for clarity
                    chunk_type_text = f"INTERVENED ({target_dir.upper()})"
                else:
                    # Default placeholders
                    vec_data = intervention_data['left']
                    base_vec = vec_data['base']
                    ortho_vec = vec_data['ortho']
                    t_idx = vec_data['target_idx']
                    o_idx = idx_right
                    
                    do_intervene = False
                    chunk_type_text = "Standard"

                # C. Consume the Latch
                pending_intervention = None

                # D. JAX Execution
                obs_jax = jnp.array([obs])
                noise = jax.random.normal(key, (1, noise_dim))
                
                # NOTE: Changed unpacking here to capture `logits_jax`
                chunk_raw, _, probs_jax, logits_jax, strength_val = intervened_action_step(
                    agent.network.params, 
                    obs_jax, 
                    noise, 
                    base_vec,
                    ortho_vec,
                    clf_w,
                    clf_b,
                    t_idx,
                    o_idx,
                    idx_stay,
                    do_intervene,
                    p_val,
                    f_val,
                    use_fixed_strength
                )
                
                # E. Update Text (Formatted like video script)
                probs = np.array(probs_jax)[0]
                logits = np.array(logits_jax)[0]
                strength_display = float(strength_val) if do_intervene else 0.0

                p_left = probs[idx_left] 
                p_right = probs[idx_right]
                
                l_stay = logits[idx_stay]
                l_left = logits[idx_left]
                l_right = logits[idx_right]

                # Determine direction text based on probability (not just button press)
                if p_right > p_left:
                    pct = p_right * 100
                    direction_text = "Going Right"
                else:
                    pct = p_left * 100
                    direction_text = "Going Left"

                # Format Info String
                if use_fixed_strength:
                    str_info = f"Fixed: {strength_display:.2f}"
                else:
                    str_info = f"Auto(p={p_val}): {strength_display:.4f}"

                line1 = f"Chunk {chunk_counter} ({chunk_type_text}) | {direction_text}: {pct:.1f}%"
                line2 = f"Logits -> S:{l_stay:.1f} L:{l_left:.1f} R:{l_right:.1f}"
                line3 = f"{str_info}"
                
                text_wrapper.set_text(f"{line1}\n{line2}\n{line3}")
                
                chunk = np.array(chunk_raw).reshape(horizon_length, action_dim)
                action_queue = [a for a in chunk]

            # --- 3. Step Environment ---
            action = action_queue.pop(0)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            env.render()
            
            done = bool(terminated or truncated)
            ep_len += 1
            obs = next_obs
            
            clock.tick(60) 

        print(f"Episode {ep+1} Finished.")
    env.close()

    pygame.quit()

def main(_):
    assert FLAGS.ckpt_dir is not None, "Must provide --ckpt_dir"
    assert FLAGS.ckpt_step is not None, "Must provide --ckpt_step"

    rng = jax.random.PRNGKey(FLAGS.seed)
    
    # 1. Load Agent
    config = FLAGS.agent
    config["horizon_length"] = FLAGS.horizon_length
    
    # Use 'human' render mode for interactive window
    eval_env, _ = make_env_flat_with_overlay(video=False) 
    obs_shape = eval_env.observation_space.shape
    act_shape = eval_env.action_space.shape
    
    example_obs = np.zeros(obs_shape, dtype=np.float32)
    example_action = np.zeros(act_shape, dtype=np.float32)
    
    agent_class = agents[config['agent_name']]
    agent = agent_class.create(FLAGS.seed, example_obs, example_action, config)
    agent = restore_agent(agent, FLAGS.ckpt_dir, FLAGS.ckpt_step)
    eval_env.close()
    
    # 2. Load Classifier
    clf = load_classifier(FLAGS.classifier_path)
    
    # 3. Pre-calculate Vectors for both Left and Right
    print("Pre-calculating intervention vectors...")
    intervention_data = {}
    
    for direction in ['left', 'right']:
        base, ortho, t_idx = get_single_direction_vectors(clf, direction)
        if base is not None:
            intervention_data[direction] = {
                'base': base,
                'ortho': ortho,
                'target_idx': t_idx
            }
        else:
            raise ValueError(f"Failed to generate vectors for {direction}")

    # 4. Run Interactive Eval
    mode_str = f"FIXED {FLAGS.fixed_strength}" if FLAGS.fixed_strength else f"AUTO (p={FLAGS.p_val})"
    print(f"\nStarting Keyboard Intervention | {mode_str}")
    
    # Create env again for the loop
    env, text_wrapper = make_env_flat_with_overlay(
        video=FLAGS.video, 
        video_folder=FLAGS.video_dir, 
        video_name_prefix='manual'
    )
    
    run_interactive_episodes(
        agent=agent,
        env=env,
        text_wrapper=text_wrapper,
        clf=clf,
        rng=rng,
        horizon_length=FLAGS.horizon_length,
        num_episodes=FLAGS.eval_episodes,
        max_steps=FLAGS.max_steps_per_episode,
        intervention_data=intervention_data,
        p_val=FLAGS.p_val,
        fixed_strength_val=FLAGS.fixed_strength
    )
    
    env.close()

if __name__ == "__main__":
    app.run(main)