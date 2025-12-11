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

# ---------- Path & Env Flags ----------
flags.DEFINE_string('env_name', 'cube-triple-play-singletask-task2-v0', 'Environment name.')
flags.DEFINE_integer('seed', 42, 'Random seed.')

# Checkpoint details
flags.DEFINE_string(
    'ckpt_dir', 
    '/p/yufeng/qc/exp/qc/reproduce/cube-triple-play-singletask-task2-v0/sd00020251124_103958', 
    'Directory containing params_*.pkl'
)
flags.DEFINE_integer('ckpt_step', 2000000, 'Checkpoint step to load.')

# ---------- Config Flags ----------
config_flags.DEFINE_config_file('agent', 'agents/acfql.py', lock_config=False)
flags.DEFINE_integer('horizon_length', 5, 'Action chunk length h. MUST match training config.')

# ---------- Eval Settings ----------
flags.DEFINE_integer('eval_episodes', 0, 'Number of evaluation episodes for stats.')
flags.DEFINE_integer('video_episodes', 5, 'Number of episodes to record video.')
flags.DEFINE_string('save_dir', 'eval_results/', 'Directory to save videos and stats.')


def run_eval_loop(agent, env, rng, horizon_length, num_episodes, desc="Eval"):
    """
    Main evaluation loop handling action chunking unrolling.
    """
    successes = []
    returns = []
    lengths = []
    
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
        save_frame = True
        while not done:
            # Check if we need to sample a new chunk
            if len(action_queue) == 0:
                rng, key = jax.random.split(rng)
                # Agent returns flattened chunk: (horizon * action_dim,)
                chunk_flat = agent.sample_actions(observations=obs, rng=key)
                # Reshape to (horizon, action_dim)
                chunk = np.array(chunk_flat).reshape(horizon_length, action_dim)
                action_queue = [a for a in chunk]

            # Execute next action in the chunk
            action = action_queue.pop(0)
            # print(f"Step {ep_len}: Taking action {action}")

            next_obs, reward, terminated, truncated, info = env.step(action)
            
            done = terminated or truncated
            ep_return += reward
            ep_len += 1
            obs = next_obs

            frame = env.render()
            #save frame as a png 
            if save_frame:
                frame_path = os.path.join(FLAGS.save_dir, f"episode_{i}_step_{ep_len}.png")
                from PIL import Image
                Image.fromarray(frame).save(frame_path)
                save_frame = False  # Save only the first frame
            
            
            # Check success (standard OGBench key)
            if 'success' in info and info['success']:
                ep_success = True

        successes.append(ep_success)
        returns.append(ep_return)
        lengths.append(ep_len)

    return np.array(successes), np.array(returns), np.array(lengths)


def main(_):
    # 1. Setup paths
    exp_name = os.path.basename(FLAGS.ckpt_dir)
    full_save_dir = os.path.join(FLAGS.save_dir, FLAGS.env_name, exp_name, f"step_{FLAGS.ckpt_step}")
    os.makedirs(full_save_dir, exist_ok=True)
    
    # 2. Create Environment (Fast Mode)
    print(f"Creating environment '{FLAGS.env_name}'...")
    
    # Passing env_only=True skips the dataset download/load entirely.
    # eval_env = make_ogbench_env_and_datasets(
    #     FLAGS.env_name,
    #     env_only=True
    # )
    eval_env,_,_ = ogbench.make_env_and_datasets(
            FLAGS.env_name,
            dataset_dir=".ogbench/data"
        )

    # 3. Restore Agent
    print(f"Restoring agent from {FLAGS.ckpt_dir} at step {FLAGS.ckpt_step}...")
    rng = jax.random.PRNGKey(FLAGS.seed)
    
    # Create dummy samples to initialize agent shape
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

    # 4. Run Numerical Evaluation
    if FLAGS.eval_episodes > 0:
        print(f"\n=== Running {FLAGS.eval_episodes} Eval Episodes ===")
        successes, returns, lengths = run_eval_loop(
            agent, eval_env, rng, FLAGS.horizon_length, FLAGS.eval_episodes, desc="Eval"
        )

        success_rate = np.mean(successes) * 100
        avg_return = np.mean(returns)
        avg_len = np.mean(lengths)

        print("\n" + "-"*30)
        print(f"Success Rate: {success_rate:.2f}%")
        print(f"Avg Return:   {avg_return:.2f}")
        print(f"Avg Length:   {avg_len:.2f}")
        print("-"*30 + "\n")

        # Save stats
        stats_path = os.path.join(full_save_dir, "eval_stats.json")
        with open(stats_path, 'w') as f:
            json.dump({
                "success_rate": float(success_rate),
                "avg_return": float(avg_return),
                "avg_length": float(avg_len),
                "episodes": FLAGS.eval_episodes
            }, f, indent=4)

    # 5. Run Video Evaluation
    if FLAGS.video_episodes > 0:
        print(f"=== Recording {FLAGS.video_episodes} Video Episodes ===")
        
        # Close the previous environment to free resources
        eval_env.close()

        print("Creating video environment...")
        # Re-create environment
        env,_,_ = ogbench.make_env_and_datasets(
            FLAGS.env_name,
            dataset_dir=".ogbench/data"
        )


        # 1. Force the render mode
        env.unwrapped.render_mode = "rgb_array"

        # 2. Force the metadata to declare support for rgb_array
        # This satisfies the passive_env_checker
        if not hasattr(env.unwrapped, 'metadata') or env.unwrapped.metadata is None:
            env.unwrapped.metadata = {}
        env.unwrapped.metadata = dict(env.unwrapped.metadata) # Create a copy to be safe
        env.unwrapped.metadata['render_modes'] = ["rgb_array"]

        # --- Wrapping ---
        print(f"Metadata patched: {env.unwrapped.metadata}")


        # Setup Video Wrapper
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