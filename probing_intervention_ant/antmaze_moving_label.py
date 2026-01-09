import os
os.environ['MUJOCO_GL'] = 'egl' 

import json
import ogbench
import tqdm
import jax
import numpy as np
import gymnasium as gym
from absl import app, flags
from ml_collections import config_flags
from gymnasium.wrappers import RecordVideo

from envs.ogbench_utils import make_ogbench_env_and_datasets
from utils.flax_utils import restore_agent
from agents import agents

FLAGS = flags.FLAGS

# ---------- Path & Env Flags (UPDATED DEFAULTS) ----------
flags.DEFINE_string('env_name', 'antmaze-large-navigate-singletask-v0', 'Environment name.')
flags.DEFINE_integer('seed', 42, 'Random seed.')

# Checkpoint details
flags.DEFINE_string(
    'ckpt_dir', 
    '/p/yufeng/qc/exp/qc/ogbench_antmaze/antmaze-large-navigate-singletask-v0/sd00020251209_110916', 
    'Directory containing params_*.pkl'
)
flags.DEFINE_integer('ckpt_step', 5000000, 'Checkpoint step to load.')

# ---------- Config Flags ----------
config_flags.DEFINE_config_file('agent', 'agents/acfql.py', lock_config=False)
flags.DEFINE_integer('horizon_length', 5, 'Action chunk length h. MUST match training config.')

# ---------- Eval Settings ----------
flags.DEFINE_integer('eval_episodes', 10, 'Number of evaluation episodes for stats.')
flags.DEFINE_integer('video_episodes', 0, 'Number of episodes to record video.')
flags.DEFINE_string('save_dir', 'eval_results/', 'Directory to save videos and stats.')


# --- Helper Function for Probing ---
def get_moving_label(obs_t, obs_t_plus_5, stop_threshold=0.5):
    """
    Returns a label for the movement direction over 5 steps.
    0: Stop/Stuck, 1: Forward, 2: Left, 3: Right, 4: Backward
    """
    # 1. Extract Positions (X, Y) - Indices 0, 1
    pos_t = obs_t[:2]
    pos_future = obs_t_plus_5[:2]
    
    # 2. Extract Current Heading (Yaw) from Quaternion - Indices 3,4,5,6
    w, x, y, z = obs_t[3:7]
    current_yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    
    # 3. Calculate Global Displacement Vector
    dx_global = pos_future[0] - pos_t[0]
    dy_global = pos_future[1] - pos_t[1]
    
    # Check for "Stop/Stuck"
    dist = np.sqrt(dx_global**2 + dy_global**2)
    if dist < stop_threshold:
        return 0  # STOP
        
    # 4. Rotate Vector to Egocentric Frame
    x_local = dx_global * np.cos(current_yaw) + dy_global * np.sin(current_yaw)
    y_local = -dx_global * np.sin(current_yaw) + dy_global * np.cos(current_yaw)
    
    # 5. Classify based on Angle
    angle = np.arctan2(y_local, x_local)
    angle_deg = np.degrees(angle)
    
    if -45 <= angle_deg <= 45:
        return 1  # FORWARD
    elif 45 < angle_deg <= 135:
        return 2  # LEFT
    elif -135 <= angle_deg < -45:
        return 3  # RIGHT
    else:
        return 4  # BACKWARD
# -----------------------------------

def run_eval_loop(agent, env, rng, horizon_length, num_episodes, desc="Eval"):
    """
    Main evaluation loop handling action chunking unrolling and probing data collection.
    """
    successes = []
    returns = []
    lengths = []
    
    # Initialize Probing Counters
    label_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    label_names = {0: "Stop/Stuck", 1: "Forward", 2: "Left", 3: "Right", 4: "Backward"}
    
    action_dim = env.action_space.shape[-1]

    for i in tqdm.tqdm(range(num_episodes), desc=desc):
        rng, seed_key = jax.random.split(rng)
        seed = int(jax.random.randint(seed_key, (), 0, 2**31 - 1))
        
        obs, _ = env.reset(seed=seed)
        
        done = False
        ep_return = 0.0
        ep_len = 0
        ep_success = False
        
        action_queue = []
        obs_history = [] 

        while not done:
            # --- FIX: Handle Dict vs Array Observation ---
            if isinstance(obs, dict):
                current_state = obs['state']
            else:
                # If obs is an array, it is typically [State (29) | Goal (29)]
                # We just slice the first 29 to get the state.
                current_state = obs[:29]
            # ---------------------------------------------

            obs_history.append(current_state)

            # --- Probing Logic ---
            if len(obs_history) > horizon_length*6:
                past_state = obs_history[-(horizon_length*6+1)]
                label = get_moving_label(past_state, current_state)
                label_counts[label] += 1
            # ---------------------

            # Check if we need to sample a new chunk
            if len(action_queue) == 0:
                rng, key = jax.random.split(rng)
                chunk_flat = agent.sample_actions(observations=obs, rng=key)
                chunk = np.array(chunk_flat).reshape(horizon_length, action_dim)
                action_queue = [a for a in chunk]

            # Execute next action
            action = action_queue.pop(0)

            next_obs, reward, terminated, truncated, info = env.step(action)
            
            done = terminated or truncated
            ep_return += reward
            ep_len += 1
            obs = next_obs
            
            if 'success' in info and info['success']:
                ep_success = True

        successes.append(ep_success)
        returns.append(ep_return)
        lengths.append(ep_len)

    # Print Probing Statistics
    print("\n" + "="*40)
    print(f"PROBING STATS (Over {num_episodes} Episodes)")
    print("="*40)
    total_samples = sum(label_counts.values())
    if total_samples > 0:
        for cat_id, count in label_counts.items():
            percentage = (count / total_samples) * 100
            print(f"{label_names[cat_id]:<12} : {count:5d} samples ({percentage:.1f}%)")
    else:
        print("No samples collected (Episodes too short?)")
    print("="*40 + "\n")

    return np.array(successes), np.array(returns), np.array(lengths)


def main(_):
    # 1. Setup paths
    exp_name = os.path.basename(FLAGS.ckpt_dir)
    full_save_dir = os.path.join(FLAGS.save_dir, FLAGS.env_name, exp_name, f"step_{FLAGS.ckpt_step}")
    os.makedirs(full_save_dir, exist_ok=True)
    
    # 2. Create Environment
    print(f"Creating environment '{FLAGS.env_name}'...")
    eval_env,_,_ = ogbench.make_env_and_datasets(
            FLAGS.env_name,
            dataset_dir=".ogbench/data"
        )

    # 3. Restore Agent
    print(f"Restoring agent from {FLAGS.ckpt_dir} at step {FLAGS.ckpt_step}...")
    rng = jax.random.PRNGKey(FLAGS.seed)
    
    obs_sample = eval_env.observation_space.sample()
    action_sample = eval_env.action_space.sample()
    
    config = FLAGS.agent
    config["horizon_length"] = FLAGS.horizon_length

    agent_class = agents[config['agent_name']]
    agent = agent_class.create(
        FLAGS.seed,
        obs_sample,
        action_sample,
        config,
    )

    agent = restore_agent(agent, FLAGS.ckpt_dir, FLAGS.ckpt_step)

    # 4. Run Numerical Evaluation + Probing
    if FLAGS.eval_episodes > 0:
        print(f"\n=== Running {FLAGS.eval_episodes} Eval Episodes ===")
        successes, returns, lengths = run_eval_loop(
            agent, eval_env, rng, FLAGS.horizon_length, FLAGS.eval_episodes, desc="Eval & Probe"
        )

        success_rate = np.mean(successes) * 100
        avg_return = np.mean(returns)
        avg_len = np.mean(lengths)

        print("\n" + "-"*30)
        print(f"Success Rate: {success_rate:.2f}%")
        print(f"Avg Return:   {avg_return:.2f}")
        print(f"Avg Length:   {avg_len:.2f}")
        print("-"*30 + "\n")

    # 5. Run Video Evaluation
    if FLAGS.video_episodes > 0:
        print(f"=== Recording {FLAGS.video_episodes} Video Episodes ===")
        eval_env.close()
        
        # Re-create env for video
        env,_,_ = ogbench.make_env_and_datasets(
            FLAGS.env_name,
            dataset_dir=".ogbench/data"
        )
        
        # Metadata patching for video recording
        env.unwrapped.render_mode = "rgb_array"
        if not hasattr(env.unwrapped, 'metadata') or env.unwrapped.metadata is None:
            env.unwrapped.metadata = {}
        env.unwrapped.metadata = dict(env.unwrapped.metadata)
        env.unwrapped.metadata['render_modes'] = ["rgb_array"]

        video_path = os.path.join(full_save_dir, "videos")
        video_env = RecordVideo(
            env, 
            video_folder=video_path,
            episode_trigger=lambda x: True, 
            name_prefix="eval_video"
        )

        _, _, _ = run_eval_loop(
            agent, video_env, rng, FLAGS.horizon_length, FLAGS.video_episodes, desc="Video"
        )
        video_env.close()
        print(f"Videos saved to: {video_path}")

if __name__ == '__main__':
    app.run(main)