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
flags.DEFINE_string('intervention_direction', 'right', 'Direction to force: left, right, or straight.')
flags.DEFINE_float('intervention_strength', 1.0, 'Strength of the intervention (alpha).')

flags.DEFINE_integer('eval_episodes', 10, 'Number of evaluation episodes.')
flags.DEFINE_integer('max_steps_per_episode', 2000, 'Safety cap on episode length.')
flags.DEFINE_integer('horizon_length', 7, 'Action chunk length.')
flags.DEFINE_bool('video', False, 'Whether to record video.')
flags.DEFINE_string('video_dir', 'videos/intervention/one-chunk/h9', 'Directory for videos.')


# ---------- Wrappers ----------

class TextOverlayWrapper(gym.Wrapper):
    """
    Intercepts render call and draws multi-line text using OpenCV.
    """
    def __init__(self, env):
        super().__init__(env)
        self.current_text = ""
    
    def set_text(self, text):
        self.current_text = text

    def render(self):
        frame = self.env.render()
        if frame is None or not isinstance(frame, np.ndarray):
            return frame

        frame = frame.copy()

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        color = (255, 255, 255) # White
        thickness = 1
        line_type = cv2.LINE_AA
        
        # Split text into lines
        lines = self.current_text.split('\n')
        
        # Starting position
        x, y = 20, 40
        line_height = 25
        
        for line in lines:
            # Draw Outline (Black)
            cv2.putText(frame, line, (x, y), font, font_scale, (0, 0, 0), thickness + 2, line_type)
            # Draw Text (White)
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

def get_intervention_vector(clf, direction):
    target_idx = -1
    if direction == 'straight' or direction == 'stay':
        target_idx = 0
    elif direction == 'left':
        target_idx = 1
    elif direction == 'right':
        target_idx = 2
    else:
        raise ValueError(f"Unknown direction: {direction}")
    
    class_index = np.where(clf.classes_ == target_idx)[0][0]
    weight_vector = clf.coef_[class_index]
    return jnp.array(weight_vector)

@jax.jit
def intervened_action_step(params, obs, noise, perturbation_vector):
    """
    Manually replicates the actor forward pass.
    """
    x = jnp.concatenate([obs, noise], axis=-1)
    mlp_params = params['modules_actor_onestep_flow']['mlp']
    
    # Forward pass through hidden layers
    for i in range(4): # Dense_0 to Dense_3
        k = mlp_params[f'Dense_{i}']['kernel']
        b = mlp_params[f'Dense_{i}']['bias']
        x = nn.gelu(x @ k + b)
    
    # *** Capture Feature ***
    feature_before = x 
    
    # --- Add Perturbation (Intervention) ---
    x = x + perturbation_vector

    # Projection to action dim (Dense_4)
    k = mlp_params['Dense_4']['kernel']
    b = mlp_params['Dense_4']['bias']
    x = x @ k + b
    x = jnp.clip(x, -1, 1)
    
    return x, feature_before

# ---------- Eval Loop ----------

def run_intervention_episodes(agent, env, text_wrapper, clf, rng, horizon_length, num_episodes, max_steps, perturbation):
    
    action_dim = env.action_space.shape[-1]
    noise_dim = agent.config['action_dim'] * (agent.config['horizon_length'] if agent.config["action_chunking"] else 1)
    
    lane_changes_left = 0
    lane_changes_right = 0
    
    # Pre-allocate a zero perturbation vector for non-intervened steps
    zero_perturbation = jnp.zeros_like(perturbation)

    for ep in range(num_episodes):
        text_wrapper.set_text(f"Ep {ep+1} Start")
        obs, _ = env.reset(seed=int(jax.random.randint(rng, (), 0, 2**31 - 1)))
        
        
        try:
            start_lane = env.unwrapped.vehicle.lane_index[2]
        except:
            start_lane = -1
            
        done = False
        ep_len = 0
        action_queue = []
        First_chunk = True
        chunk_counter = 0
        
        while not done and ep_len < max_steps:
            if len(action_queue) == 0:
                chunk_counter += 1
                rng, key = jax.random.split(rng)
                
                obs_jax = jnp.array([obs])
                noise = jax.random.normal(key, (1, noise_dim))
                
                # Decision: Are we intervening this chunk?
                if First_chunk:
                    current_perturbation = perturbation
                    chunk_type_text = "INTERVENED"
                else:
                    current_perturbation = zero_perturbation
                    chunk_type_text = "Standard"

                # Run Forward Pass (Always get feature)
                chunk_raw, feature_jax = intervened_action_step(
                    agent.network.params, 
                    obs_jax, 
                    noise, 
                    current_perturbation
                )
                
                # --- Classifier / Display Logic ---
                feature_np = np.array(feature_jax)
                
                # 1. Get Probabilities
                probs = clf.predict_proba(feature_np)[0]
                
                # 2. Get Raw Logits (decision_function)
                logits = clf.decision_function(feature_np)[0]
                
                # Map based on classifier classes (0:Stay, 1:Left, 2:Right)
                # Ensure mapping is robust to class order
                idx_stay = np.where(clf.classes_ == 0)[0][0]
                idx_left = np.where(clf.classes_ == 1)[0][0]
                idx_right = np.where(clf.classes_ == 2)[0][0]

                p_left = probs[idx_left]
                p_right = probs[idx_right]
                
                l_stay = logits[idx_stay]
                l_left = logits[idx_left]
                l_right = logits[idx_right]

                # Determine Main Direction String
                if p_right > p_left:
                    pct = p_right * 100
                    direction_text = "Going Right"
                else:
                    pct = p_left * 100
                    direction_text = "Going Left"
                
                # Construct Multi-line Text
                line1 = f"Chunk {chunk_counter} ({chunk_type_text}) | {direction_text}: {pct:.1f}%"
                line2 = f"Logits -> S: {l_stay:.2f}, L: {l_left:.2f}, R: {l_right:.2f}"
                
                text_wrapper.set_text(f"{line1}\n{line2}")
                # ----------------------------------
                
                chunk = np.array(chunk_raw).reshape(horizon_length, action_dim)
                action_queue = [a for a in chunk]
                First_chunk = False

            action = action_queue.pop(0)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)
            ep_len += 1
            obs = next_obs
        
        try:
            end_lane = env.unwrapped.vehicle.lane_index[2]
            if start_lane != -1:
                if end_lane < start_lane:
                    lane_changes_left += 1
                elif end_lane > start_lane:
                    lane_changes_right += 1
        except:
            pass
            
        print(f"Episode {ep+1}: Len {ep_len} | Start {start_lane} -> End {end_lane}")

    print("\n=== Summary ===")
    print(f"Total Episodes: {num_episodes}")
    print(f"Left: {lane_changes_left}, Right: {lane_changes_right}")


def main(_):
    assert FLAGS.ckpt_dir is not None, "Must provide --ckpt_dir"
    assert FLAGS.ckpt_step is not None, "Must provide --ckpt_step"

    rng = jax.random.PRNGKey(FLAGS.seed)
    
    # 1. Load Agent
    config = FLAGS.agent
    config["horizon_length"] = FLAGS.horizon_length
    
    eval_env, _ = make_env_flat_with_overlay(video=False) 
    obs_shape = eval_env.observation_space.shape
    act_shape = eval_env.action_space.shape
    
    example_obs = np.zeros(obs_shape, dtype=np.float32)
    example_action = np.zeros(act_shape, dtype=np.float32)
    
    agent_class = agents[config['agent_name']]
    agent = agent_class.create(FLAGS.seed, example_obs, example_action, config)
    agent = restore_agent(agent, FLAGS.ckpt_dir, FLAGS.ckpt_step)
    eval_env.close()
    
    # 2. Load Classifier & Prepare Vector
    clf = load_classifier(FLAGS.classifier_path)
    base_vector = get_intervention_vector(clf, FLAGS.intervention_direction)
    perturbation = base_vector * FLAGS.intervention_strength
    
    # 3. Run Eval
    print(f"\nRunning Intervention: Force {FLAGS.intervention_direction.upper()} (Strength {FLAGS.intervention_strength})")
    
    env_prefix = f"intervention_{FLAGS.intervention_direction}_str{FLAGS.intervention_strength}"
    
    env, text_wrapper = make_env_flat_with_overlay(
        video=FLAGS.video, 
        video_folder=FLAGS.video_dir, 
        video_name_prefix=env_prefix
    )
    
    run_intervention_episodes(
        agent=agent,
        env=env,
        text_wrapper=text_wrapper,
        clf=clf,
        rng=rng,
        horizon_length=FLAGS.horizon_length,
        num_episodes=FLAGS.eval_episodes,
        max_steps=FLAGS.max_steps_per_episode,
        perturbation=perturbation
    )
    
    env.close()

if __name__ == "__main__":
    app.run(main)