import os
import pickle
import tqdm
import jax
import jax.numpy as jnp
import numpy as np
import gymnasium as gym
from absl import app, flags
from ml_collections import config_flags
import ogbench

from utils.flax_utils import restore_agent
from agents import agents
from utils.networks import ActorVectorField 

FLAGS = flags.FLAGS

# ---------- Flags ----------
flags.DEFINE_string('env_name', 'antmaze-large-navigate-singletask-v0', 'Environment name.')
flags.DEFINE_integer('seed', 42, 'Random seed.')

# Checkpoint details
flags.DEFINE_string(
    'ckpt_dir', 
    '/p/yufeng/qc/exp/qc/ogbench_antmaze/antmaze-large-navigate-singletask-v0/sd00020251209_110916', 
    'Directory containing params_*.pkl'
)
flags.DEFINE_integer('ckpt_step', 5000000, 'Checkpoint step to load.')

# Config
config_flags.DEFINE_config_file('agent', 'agents/acfql.py', lock_config=False)
flags.DEFINE_integer('horizon_length', 5, 'Action chunk length h.')

# Data Collection Settings
flags.DEFINE_integer('num_data', 10000, 'Target number of chunk samples to collect.')
flags.DEFINE_integer('max_steps_per_episode', 10000, 'Max steps per episode.')
flags.DEFINE_string('output_path', 'data/antmaze_raw_sequences.pkl', 'Path to save collected data.')


def get_state_from_obs(obs):
    """Robustly extracts the flat state array from dict or array obs."""
    if isinstance(obs, dict):
        return obs['state']
    # If obs is an array, slice the first 29 (State)
    return obs[:29]

# =============================================================================
# Feature Extraction Logic
# =============================================================================

def create_feature_extractor(config, example_obs, example_actions):
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
# Data Collection Loop (Sequence Preserving)
# =============================================================================

def run_collection(agent, env, rng, horizon_length, num_data, max_steps, extract_fn):
    
    collected_samples = [] 
    
    action_dim = env.action_space.shape[-1]
    noise_dim = agent.config['action_dim'] * (agent.config['horizon_length'] if agent.config["action_chunking"] else 1)
    
    print(f"Starting data collection until {num_data} chunks...")
    pbar = tqdm.tqdm(total=num_data, desc="Collecting Chunks")

    episode_id = 0

    while len(collected_samples) < num_data:
        # Start new episode
        seed = int(jax.random.randint(rng, (), 0, 2**31 - 1))
        obs, _ = env.reset(seed=seed)
        
        done = False
        ep_len = 0
        chunk_index = 0
        action_queue = []
        
        chunk_start_info = None 

        while not done and ep_len < max_steps:
            current_state_flat = get_state_from_obs(obs)

            # --- 1. Chunk Logic ---
            if len(action_queue) == 0:
                
                # A. Save PREVIOUS chunk result
                if chunk_start_info is not None:
                    # We save the transition: State_Start -> State_End
                    collected_samples.append({
                        'episode_id': episode_id,
                        'chunk_index': chunk_index - 1, # The index of the chunk that just finished
                        'feature': chunk_start_info['feature'],
                        'state': chunk_start_info['state'], # State at start of chunk
                        'next_state': current_state_flat,   # State at end of chunk
                    })
                    pbar.update(1)
                    if len(collected_samples) >= num_data:
                        break

                # B. Start NEW chunk
                rng, key, noise_key = jax.random.split(rng, 3)
                
                # Prepare batching for JAX
                if isinstance(obs, dict):
                    obs_batched = jax.tree_map(lambda x: x[None, ...], obs)
                else:
                    obs_batched = obs[None, ...]
                
                noises = jax.random.normal(noise_key, (1, noise_dim))
                
                # Extract Feature
                features = extract_fn(agent.network.params, obs_batched, noises)[0]
                
                # Sample Actions
                chunk_flat = agent.sample_actions(observations=obs, rng=key)
                chunk = np.array(chunk_flat).reshape(horizon_length, action_dim)
                action_queue = [a for a in chunk]
                
                # Store Start Info
                chunk_start_info = {
                    'feature': np.array(features),
                    'state': current_state_flat
                }
                chunk_index += 1

            # --- 2. Execute Action ---
            action = action_queue.pop(0)
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            done = terminated or truncated
            ep_len += 1
            obs = next_obs
            
            # If episode ends mid-chunk, we discard the incomplete chunk
            if done:
                chunk_start_info = None
        
        episode_id += 1

    pbar.close()
    return collected_samples

def main(_):
    # 1. Setup Env
    print(f"Creating environment '{FLAGS.env_name}'...")
    eval_env, _, _ = ogbench.make_env_and_datasets(FLAGS.env_name, dataset_dir=".ogbench/data")

    # 2. Restore Agent
    print(f"Restoring agent from {FLAGS.ckpt_dir}...")
    rng = jax.random.PRNGKey(FLAGS.seed)
    obs_sample = eval_env.observation_space.sample()
    action_sample = eval_env.action_space.sample()
    
    config = FLAGS.agent
    config["horizon_length"] = FLAGS.horizon_length
    agent_class = agents[config['agent_name']]
    agent = agent_class.create(FLAGS.seed, obs_sample, action_sample, config)
    agent = restore_agent(agent, FLAGS.ckpt_dir, FLAGS.ckpt_step)

    # 3. Compile Extractor
    if isinstance(obs_sample, dict):
        example_obs_batched = jax.tree_map(lambda x: x[None, ...], obs_sample)
    else:
        example_obs_batched = obs_sample[None, ...]
    
    if config["action_chunking"]:
        ex_actions_chunk = jnp.zeros((1, config['action_dim'] * config['horizon_length']))
    else:
        ex_actions_chunk = jnp.zeros((1, config['action_dim']))

    extract_fn = create_feature_extractor(config, example_obs_batched, ex_actions_chunk)

    # 4. Run Collection
    dataset_list = run_collection(
        agent=agent,
        env=eval_env,
        rng=rng,
        horizon_length=FLAGS.horizon_length,
        num_data=FLAGS.num_data,
        max_steps=FLAGS.max_steps_per_episode,
        extract_fn=extract_fn
    )

    # 5. Save Structured Data
    if len(dataset_list) > 0:
        # Convert to dictionary of arrays
        data_dict = {
            'features': np.array([d['feature'] for d in dataset_list]),
            'states': np.array([d['state'] for d in dataset_list]),
            'next_states': np.array([d['next_state'] for d in dataset_list]),
            'episode_ids': np.array([d['episode_id'] for d in dataset_list]),
            'chunk_indices': np.array([d['chunk_index'] for d in dataset_list])
        }
        
        print("\n=== Data Collection Summary ===")
        print(f"Total Chunks: {len(data_dict['features'])}")
        print(f"Unique Episodes: {len(np.unique(data_dict['episode_ids']))}")
        
        os.makedirs(os.path.dirname(FLAGS.output_path), exist_ok=True)
        with open(FLAGS.output_path, 'wb') as f:
            pickle.dump(data_dict, f)
        print(f"Saved to {FLAGS.output_path}")

if __name__ == '__main__':
    app.run(main)