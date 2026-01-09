import os
import pickle
import tqdm
import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import cv2
from absl import app, flags
from ml_collections import config_flags
import ogbench

from utils.flax_utils import restore_agent
from agents import agents

FLAGS = flags.FLAGS

# ---------- General Flags ----------
flags.DEFINE_string('env_name', 'antmaze-large-navigate-singletask-v0', 'Environment name.')
flags.DEFINE_integer('seed', 42, 'Random seed.')

# ---------- Checkpoint ----------
flags.DEFINE_string(
    'ckpt_dir', 
    '/p/yufeng/qc/exp/qc/ogbench_antmaze/antmaze-large-navigate-singletask-v0/sd00020251209_110916', 
    'Directory containing params_*.pkl'
)
flags.DEFINE_integer('ckpt_step', 5000000, 'Checkpoint step.')
config_flags.DEFINE_config_file('agent', 'agents/acfql.py', lock_config=False)

# ---------- Intervention Settings ----------
flags.DEFINE_string('classifier_path', 'data/antmaze_classifiers_nofilter.pkl', 'Path to classifier dictionary.')
flags.DEFINE_integer('horizon_choice', 1, 'Which horizon classifier to use (1-10).')

flags.DEFINE_string('intervention_direction', 'right', 'Target direction: forward, left, right, backward.')
flags.DEFINE_float('intervention_strength', 5.0, 'Strength of the intervention (alpha).')

# ---------- Eval & Video ----------
flags.DEFINE_integer('eval_episodes', 2, 'Number of evaluation episodes.')
flags.DEFINE_integer('max_steps_per_episode', 2000, 'Safety cap.')
flags.DEFINE_bool('video', True, 'Whether to record video.')
flags.DEFINE_string('video_dir', 'eval_results/intervention_fixed', 'Directory for videos.')
flags.DEFINE_integer('horizon_length', 5, 'Horizon length for action chunking.')


# =============================================================================
# 1. Wrappers (Visualization - ADJUSTED FOR SIZE)
# =============================================================================

class TextOverlayWrapper(gym.Wrapper):
    """
    Intercepts render call and draws multi-line text using OpenCV.
    ADJUSTED: Much smaller text and tighter line spacing.
    """
    def __init__(self, env):
        super().__init__(env)
        self.current_text = ""
        if not hasattr(self.env, 'metadata') or self.env.metadata is None:
            self.env.metadata = {}
        if isinstance(self.env.metadata, dict):
            self.env.metadata['render_modes'] = ["rgb_array"]

    def set_text(self, text):
        self.current_text = text

    def render(self):
        frame = self.env.render()
        if frame is None or not isinstance(frame, np.ndarray):
            return frame

        frame = frame.copy()

        # --- UPDATED VISUAL SETTINGS (SMALLER) ---
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.3       # Reduced from 0.4 to 0.3
        color = (255, 255, 0)  # Cyan/Yellow text
        thickness = 1          
        outline_thickness = 2  
        line_type = cv2.LINE_AA
        
        lines = self.current_text.split('\n')
        
        # Tighter top-left placement
        x, y = 4, 10           # Started slightly higher
        line_height = 12       # Reduced from 18 to 12 for tighter packing
        # -------------------------------
        
        for line in lines:
            # Black Outline (for contrast)
            cv2.putText(frame, line, (x, y), font, font_scale, (0, 0, 0), outline_thickness, line_type)
            # Text Color
            cv2.putText(frame, line, (x, y), font, font_scale, color, thickness, line_type)
            y += line_height
        
        return frame

def make_env_with_overlay(env_name, video=False, video_folder=None, video_name_prefix='intervention'):
    env, _, _ = ogbench.make_env_and_datasets(env_name, dataset_dir=".ogbench/data")
    env.unwrapped.render_mode = "rgb_array"
    
    text_wrapper = TextOverlayWrapper(env)
    env = text_wrapper
    
    if video:
        os.makedirs(video_folder, exist_ok=True)
        env = RecordVideo(
            env,
            video_folder=video_folder,
            name_prefix=video_name_prefix,
            episode_trigger=lambda x: True,
            disable_logger=False
        )
    return env, text_wrapper

# =============================================================================
# 2. Intervention Logic
# =============================================================================

def load_classifier_data(path, horizon_idx):
    with open(path, 'rb') as f:
        data_dict = pickle.load(f)
    if horizon_idx not in data_dict:
        raise ValueError(f"Horizon {horizon_idx} not found in {path}.")
    print(f"Loaded classifier for Horizon {horizon_idx}. Test Acc: {data_dict[horizon_idx]['test_acc']:.2%}")
    return data_dict[horizon_idx]['model'], data_dict[horizon_idx]['scaler']

def get_intervention_vector(clf, scaler, direction_str):
    mapping = {'forward': 1, 'left': 2, 'right': 3, 'backward': 4}
    target_class_id = mapping[direction_str.lower()]
    try:
        class_idx = np.where(clf.classes_ == target_class_id)[0][0]
    except IndexError:
        raise ValueError(f"Class {target_class_id} ({direction_str}) not found in classifier.")

    w = clf.coef_[class_idx]
    eff_direction = w / scaler.scale_
    norm = np.linalg.norm(eff_direction)
    if norm > 1e-6:
        eff_direction = eff_direction / norm
        
    return jnp.array(eff_direction)

@jax.jit
def intervened_action_step(params, obs, noise, perturbation_vector):
    x = jnp.concatenate([obs, noise], axis=-1)
    mlp_params = params['modules_actor_onestep_flow']['mlp']
    
    # Encoder layers
    for i in range(4):
        layer_name = f'Dense_{i}'
        k = mlp_params[layer_name]['kernel']
        b = mlp_params[layer_name]['bias']
        x = nn.gelu(x @ k + b)
    
    # --- INTERVENE ---
    x_intervened = x + perturbation_vector
    # -----------------

    # Final Projection
    k = mlp_params['Dense_4']['kernel']
    b = mlp_params['Dense_4']['bias']
    action = x_intervened @ k + b
    return action, x 

# =============================================================================
# 3. Evaluation Loop
# =============================================================================

def run_intervention(agent, env, text_wrapper, clf, scaler, rng, config, perturbation):
    
    action_dim = env.action_space.shape[-1]
    h_len = config['horizon_length']
    noise_dim = config['action_dim'] * (h_len if config["action_chunking"] else 1)
    
    # Stats mappings
    label_mapping = {1: "Fwd", 2: "Left", 3: "Right", 4: "Back"}
    short_mapping = {1: "F", 2: "L", 3: "R", 4: "B"}

    for ep in range(FLAGS.eval_episodes):
        seed = int(jax.random.randint(rng, (), 0, 2**31 - 1))
        obs, _ = env.reset(seed=seed)
        
        done = False
        ep_len = 0
        chunk_count = 0
        action_queue = []
        
        while not done and ep_len < FLAGS.max_steps_per_episode:
            
            if len(action_queue) == 0:
                chunk_count += 1
                rng, key, noise_key = jax.random.split(rng, 3)

                # after 500 steps, go left (Example Logic)
                if ep_len > 500:
                    perturbation = get_intervention_vector(clf, scaler, 'left') * FLAGS.intervention_strength
                    status_text = "FORCE LEFT" # Shortened text
                else:
                    status_text = f"FORCE {FLAGS.intervention_direction.upper()}"
                
                if isinstance(obs, dict):
                    obs_arr = obs['state']
                    obs_jax = jnp.array([obs_arr])
                else:
                    obs_jax = jnp.array([obs])

                noise = jax.random.normal(noise_key, (1, noise_dim))
                
                # 1. Decide Intervention
                current_pert = perturbation
                
                # 2. Forward Pass + Intervene
                chunk_raw, feature_jax = intervened_action_step(
                    agent.network.params, obs_jax, noise, current_pert
                )
                
                # 3. Classifier Prediction (on INTERVENED feature)
                feature_np = np.array(feature_jax)
                feature_scaled = scaler.transform(feature_np)
                probs = clf.predict_proba(feature_scaled)[0]
                pred_idx = np.argmax(probs)
                pred_class = clf.classes_[pred_idx]
                
                # --- UPDATED DISPLAY LOGIC (MORE COMPACT) ---
                line1 = f"Ep{ep+1}|Ch{chunk_count}"
                line2 = status_text + f"({FLAGS.intervention_strength})"
                line3 = f"Pred:{label_mapping.get(pred_class, '?')}"

                # Construct 4-category probability string
                prob_strs = []
                for cid in [1, 2, 3, 4]:
                    lbl = short_mapping.get(cid, "?")
                    try:
                        idx = np.where(clf.classes_ == cid)[0][0]
                        p = probs[idx]
                        # Compact string: F:0.99
                        prob_strs.append(f"{lbl}:{p:.2f}")
                    except IndexError:
                        prob_strs.append(f"{lbl}:--")
                
                # Removed spaces around pipe to save horizontal space
                line4 = "|".join(prob_strs) 
                # -----------------------------

                text_wrapper.set_text(f"{line1}\n{line2}\n{line3}\n{line4}")

                # 5. Queue Actions
                chunk = np.array(chunk_raw).reshape(h_len, action_dim)
                action_queue = [a for a in chunk]

            # --- Step ---
            action = action_queue.pop(0)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)
            ep_len += 1
            obs = next_obs
        
        print(f"Episode {ep+1} finished. Length: {ep_len}")

def main(_):
    # 1. Setup Resources
    clf, scaler = load_classifier_data(FLAGS.classifier_path, FLAGS.horizon_choice)
    base_vector = get_intervention_vector(clf, scaler, FLAGS.intervention_direction)
    perturbation = base_vector * FLAGS.intervention_strength
    
    rng = jax.random.PRNGKey(FLAGS.seed)
    
    # Dummy for shape setup
    temp_env, _, _ = ogbench.make_env_and_datasets(FLAGS.env_name)
    obs_sample = temp_env.observation_space.sample()
    action_sample = temp_env.action_space.sample()
    temp_env.close()
    
    config = FLAGS.agent
    config["horizon_length"] = FLAGS.horizon_length
    agent_class = agents[config['agent_name']]
    agent = agent_class.create(FLAGS.seed, obs_sample, action_sample, config)
    agent = restore_agent(agent, FLAGS.ckpt_dir, FLAGS.ckpt_step)

    # 2. Create Env with Video & Overlay
    print(f"Setting up environment (Video={FLAGS.video})...")
    prefix = f"{FLAGS.intervention_direction}_h{FLAGS.horizon_choice}_str{FLAGS.intervention_strength}"
    
    env, text_wrapper = make_env_with_overlay(
        FLAGS.env_name, 
        video=FLAGS.video, 
        video_folder=FLAGS.video_dir,
        video_name_prefix=prefix
    )

    # 3. Run
    run_intervention(agent, env, text_wrapper, clf, scaler, rng, config, perturbation)
    
    env.close()
    if FLAGS.video:
        print(f"Videos saved to {FLAGS.video_dir}")

if __name__ == "__main__":
    app.run(main)