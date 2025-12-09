import os
import pickle
from absl import app, flags
from ml_collections import config_flags
import tqdm

import jax
import jax.numpy as jnp
import numpy as np
import gymnasium as gym

from utils_env import create_stateenv, config_state2
from utils.flax_utils import restore_agent
from agents import agents
from utils.networks import ActorVectorField

FLAGS = flags.FLAGS

flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('env_name', 'highway-state', 'Environment name.')
config_flags.DEFINE_config_file('agent', 'agents/acfql.py', lock_config=False)
flags.DEFINE_string('ckpt_dir', None, 'Directory containing params_*.pkl')
flags.DEFINE_integer('ckpt_step', None, 'Checkpoint step to load.')
flags.DEFINE_integer('horizon_length', 7, 'Action chunk length h.')
flags.DEFINE_integer('num_data', 10000, 'Target number of samples to collect.')
flags.DEFINE_integer('max_steps_per_episode', 2000, 'Max steps.')
flags.DEFINE_string('output_path', 'data/highway_features.pkl', 'Path to save collected data.')


class FlattenObsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        orig_shape = env.observation_space.shape
        flat_dim = int(np.prod(orig_shape))
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(flat_dim,), dtype=np.float32)

    def observation(self, observation):
        return np.asarray(observation, dtype=np.float32).reshape(-1)

def make_env_flat():
    env = create_stateenv(config_state2)
    env = FlattenObsWrapper(env)
    return env

# =============================================================================
# Feature Extraction Logic
# =============================================================================

def create_feature_extractor(config, example_obs, example_actions):
    """
    Recreates the ActorVectorField and returns a JIT-compiled function
    to extract the last layer features.
    """
    full_action_dim = example_actions.shape[-1]
    
    actor_def = ActorVectorField(
        hidden_dims=config['actor_hidden_dims'],
        action_dim=full_action_dim,
        layer_norm=config['actor_layer_norm'],
        encoder=None, 
        use_fourier_features=config["use_fourier_features"],
        fourier_feature_dim=config["fourier_feature_dim"],
    )

    @jax.jit
    def extract_fn(params, observations, noises):
        actor_params = params['modules_actor_onestep_flow']
        _, state = actor_def.apply(
            {'params': actor_params}, 
            observations, 
            noises, 
            capture_intermediates=True
        )
        features = state['intermediates']['mlp']['feature'][0]
        return features

    return extract_fn

# =============================================================================
# Data Collection Loop
# =============================================================================

def run_collection(agent, env, rng, horizon_length, num_data, max_steps, extract_fn):
    
    # Store collected samples as tuples: (feature, lane_before, lane_after)
    collected_samples = [] 
    
    action_dim = env.action_space.shape[-1]
    noise_dim = agent.config['action_dim'] * (agent.config['horizon_length'] if agent.config["action_chunking"] else 1)

    print(f"Starting data collection until {num_data} samples...")
    pbar = tqdm.tqdm(total=num_data, desc="Collecting Samples")

    # Loop indefinitely until we have enough data
    episode_count = 0
    while len(collected_samples) < num_data:
        episode_count += 1
        obs, _ = env.reset(seed=int(jax.random.randint(rng, (), 0, 2**31 - 1)))
        
        done = False
        ep_len = 0
        action_queue = []
        
        # Store state at the START of a chunk execution
        chunk_start_data = None 

        while not done and ep_len < max_steps:
            # Get Lane Current
            try:
                lane_info = env.unwrapped.vehicle.lane_index
                if isinstance(lane_info, tuple):
                    current_lane = int(lane_info[2])
                else:
                    current_lane = int(lane_info)
            except Exception:
                current_lane = -1 

            # Check if action queue is empty (chunk finished or start of episode)
            if len(action_queue) == 0:
                
                # 1. Process the PREVIOUS chunk if it exists
                if chunk_start_data is not None:
                    lane_before = chunk_start_data['lane_before']
                    lane_after = current_lane
                    feature = chunk_start_data['feature']
                    
                    # Determine whether to save based on logic:
                    # If lane changed -> Save
                    # If lane same -> Save with 5% prob
                    # Filter out invalid lanes (-1)
                    if lane_before != -1 and lane_after != -1:
                        should_save = False
                        if lane_before != lane_after:
                            should_save = True
                        else:
                            if np.random.random() < 0.07:
                                should_save = True
                        
                        if should_save:
                            collected_samples.append({
                                'feature': feature,
                                'lane_before': lane_before,
                                'lane_after': lane_after
                            })
                            pbar.update(1)
                            
                            # Break inner loop if target reached
                            if len(collected_samples) >= num_data:
                                break

                # 2. Prepare the NEW chunk
                rng, key, noise_key = jax.random.split(rng, 3)
                obs_jax = jnp.array([obs])
                
                # Generate noise and features
                noises = jax.random.normal(noise_key, (1, noise_dim))
                features = extract_fn(agent.network.params, obs_jax, noises)[0]
                
                chunk = agent.sample_actions(observations=obs_jax, rng=key)[0]
                chunk = np.array(chunk).reshape(horizon_length, action_dim)
                
                action_queue = [a for a in chunk]
                
                # Save start data for comparison when this chunk ends
                chunk_start_data = {
                    'feature': np.array(features),
                    'lane_before': current_lane
                }
            
            # Execute Action
            action = action_queue.pop(0)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)
            ep_len += 1
            obs = next_obs
            
            if len(collected_samples) >= num_data:
                break

    pbar.close()
    return collected_samples

def main(_):
    assert FLAGS.ckpt_dir is not None, "--ckpt_dir must be set"
    assert FLAGS.ckpt_step is not None, "--ckpt_step must be set"

    rng = jax.random.PRNGKey(FLAGS.seed)
    
    config = FLAGS.agent
    config["horizon_length"] = FLAGS.horizon_length
    
    eval_env = make_env_flat()
    obs_shape = eval_env.observation_space.shape
    act_shape = eval_env.action_space.shape
    
    example_obs = np.zeros(obs_shape, dtype=np.float32)
    example_action = np.zeros(act_shape, dtype=np.float32)
    
    agent_class = agents[config['agent_name']]
    agent = agent_class.create(FLAGS.seed, example_obs, example_action, config)
    
    print(f"Restoring agent from {FLAGS.ckpt_dir}, step {FLAGS.ckpt_step}")
    agent = restore_agent(agent, FLAGS.ckpt_dir, FLAGS.ckpt_step)
    
    if config["action_chunking"]:
        ex_actions_chunk = jnp.zeros((1, config['action_dim'] * config['horizon_length']))
    else:
        ex_actions_chunk = jnp.zeros((1, config['action_dim']))
        
    extract_fn = create_feature_extractor(config, example_obs[None, ...], ex_actions_chunk)

    dataset_list = run_collection(
        agent=agent,
        env=eval_env,
        rng=rng,
        horizon_length=FLAGS.horizon_length,
        num_data=FLAGS.num_data,
        max_steps=FLAGS.max_steps_per_episode,
        extract_fn=extract_fn
    )
    
    if len(dataset_list) == 0:
        print("No valid data collected.")
        return

    # Convert list of dicts to dictionary of arrays for saving
    all_features = np.array([d['feature'] for d in dataset_list])
    all_lane_before = np.array([d['lane_before'] for d in dataset_list])
    all_lane_after = np.array([d['lane_after'] for d in dataset_list])
    
    # Calculate simple stats
    total = len(all_lane_before)
    changed = np.sum(all_lane_before != all_lane_after)
    stayed = total - changed

    print("\n=== Data Collection Summary ===")
    print(f"Total samples collected: {total}")
    print(f"Feature shape: {all_features.shape}")
    print(f"Lane Changes Captured: {changed}")
    print(f"No Change Captured (Subsampled): {stayed}")
    
    os.makedirs(os.path.dirname(FLAGS.output_path), exist_ok=True)
    with open(FLAGS.output_path, 'wb') as f:
        pickle.dump({
            'features': all_features,
            'lane_before': all_lane_before,
            'lane_after': all_lane_after
        }, f)
        
    print(f"Saved to {FLAGS.output_path}")
    eval_env.close()

if __name__ == "__main__":
    app.run(main)