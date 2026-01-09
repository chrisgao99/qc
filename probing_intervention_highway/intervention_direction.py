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


# ---------- Wrappers (Same as eval) ----------
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

def make_env_flat(video=False, video_folder=None, video_name_prefix='intervention'):
    env = create_stateenv(config_state2)
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
    return env

# ---------- Intervention Logic ----------

def get_intervention_vector(classifier_path, direction):
    """Loads the classifier and returns the weight vector for the target class."""
    with open(classifier_path, 'rb') as f:
        clf = pickle.load(f)
    
    # Classes are mapped as: 0=Stay, 1=Left, 2=Right
    # clf.coef_ shape is (3, 512) usually for multi_class='multinomial'
    
    print(f"Loaded classifier. Classes: {clf.classes_}")
    
    target_idx = -1
    if direction == 'straight' or direction == 'stay':
        target_idx = 0
    elif direction == 'left':
        target_idx = 1
    elif direction == 'right':
        target_idx = 2
    else:
        raise ValueError(f"Unknown direction: {direction}")
    
    # Ensure we map correctly based on clf.classes_
    # clf.classes_ should be [0, 1, 2]
    # We find the index in coef_ that corresponds to our target class label
    class_index = np.where(clf.classes_ == target_idx)[0][0]
    
    weight_vector = clf.coef_[class_index]
    print(f"Target Direction: {direction} (Class {target_idx})")
    print(f"Weight Vector Norm: {np.linalg.norm(weight_vector):.4f}")
    
    return jnp.array(weight_vector)

@jax.jit
def intervened_action_step(params, obs, noise, perturbation_vector):
    """
    Manually forward passes the actor MLP, injecting the perturbation 
    at the last hidden layer.
    """
    # 1. Prepare Inputs
    # Concatenate [obs, noise] just like ActorVectorField
    # shape: (1, obs_dim + noise_dim)
    x = jnp.concatenate([obs, noise], axis=-1)
    
    # 2. Extract MLP params
    # Path: modules_actor_onestep_flow -> mlp
    # Note: Flax parameters are stored in a nested dict structure.
    # We rely on the standard naming 'Dense_0', 'Dense_1', etc. 
    mlp_params = params['modules_actor_onestep_flow']['mlp']
    
    # 3. Manual Forward Pass (Hardcoded for 4 hidden layers of 512 + 1 projection)
    # The config was (512, 512, 512, 512). 
    # This results in layers: Dense_0, Dense_1, Dense_2, Dense_3, Dense_4.
    # The 'feature' is extracted after Dense_3 (the 4th layer).
    
    # Layer 0
    k, b = mlp_params['Dense_0']['kernel'], mlp_params['Dense_0']['bias']
    x = nn.gelu(x @ k + b)
    
    # Layer 1
    k, b = mlp_params['Dense_1']['kernel'], mlp_params['Dense_1']['bias']
    x = nn.gelu(x @ k + b)
    
    # Layer 2
    k, b = mlp_params['Dense_2']['kernel'], mlp_params['Dense_2']['bias']
    x = nn.gelu(x @ k + b)
    
    # Layer 3 (Last Hidden Layer -> Feature)
    k, b = mlp_params['Dense_3']['kernel'], mlp_params['Dense_3']['bias']
    x = nn.gelu(x @ k + b)
    
    # --- INTERVENTION ---
    # Add the perturbation vector
    x = x + perturbation_vector
    # --------------------

    # Layer 4 (Projection to Action Dim)
    k, b = mlp_params['Dense_4']['kernel'], mlp_params['Dense_4']['bias']
    x = x @ k + b
    
    # Clip actions
    x = jnp.clip(x, -1, 1)
    
    return x

# ---------- Eval Loop ----------

def run_intervention_episodes(agent, env, rng, horizon_length, num_episodes, max_steps, perturbation):
    
    action_dim = env.action_space.shape[-1]
    noise_dim = agent.config['action_dim'] * (agent.config['horizon_length'] if agent.config["action_chunking"] else 1)
    
    lane_changes_left = 0
    lane_changes_right = 0
    
    for ep in range(num_episodes):
        obs, _ = env.reset(seed=int(jax.random.randint(rng, (), 0, 2**31 - 1)))
        
        # Track lane for stats
        try:
            start_lane = env.unwrapped.vehicle.lane_index[2]
        except:
            start_lane = -1
            
        done = False
        ep_len = 0
        action_queue = []
        First_chunk = True
        while not done and ep_len < max_steps:
            if len(action_queue) == 0:
                rng, key = jax.random.split(rng)
                
                # --- Generate Chunk with Intervention ---
                # 1. Generate Noise (Same process as agent.sample_actions would)
                # We need shape (1, noise_dim)
                obs_jax = jnp.array([obs])
                noise = jax.random.normal(key, (1, noise_dim))
                
                # 2. Run Intervened Forward Pass
                if First_chunk:
                    # intervene only on the first chunk
                    chunk = intervened_action_step(
                        agent.network.params, 
                        obs_jax, 
                        noise, 
                        perturbation
                    )
                    First_chunk = False
                else:
                    chunk = agent.sample_actions(obs_jax, key)
                
                # 3. Reshape
                chunk = np.array(chunk[0]).reshape(horizon_length, action_dim)
                # print(chunk)
                action_queue = [a for a in chunk]

            action = action_queue.pop(0)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)
            ep_len += 1
            obs = next_obs
        
        # End of Episode Stats
        try:
            end_lane = env.unwrapped.vehicle.lane_index[2]
            if start_lane != -1:
                if end_lane < start_lane:
                    lane_changes_left += 1
                elif end_lane > start_lane:
                    lane_changes_right += 1
        except:
            pass
            
        print(f"Episode {ep+1}: Len {ep_len} | Start Lane {start_lane} -> End Lane {end_lane}")

    print("\n=== Intervention Summary ===")
    print(f"Total Episodes: {num_episodes}")
    print(f"Lane Changes Left: {lane_changes_left}")
    print(f"Lane Changes Right: {lane_changes_right}")
    print(f"Stayed/Unknown: {num_episodes - lane_changes_left - lane_changes_right}")


def main(_):
    assert FLAGS.ckpt_dir is not None, "Must provide --ckpt_dir"
    assert FLAGS.ckpt_step is not None, "Must provide --ckpt_step"

    rng = jax.random.PRNGKey(FLAGS.seed)
    
    # 1. Load Agent
    config = FLAGS.agent
    config["horizon_length"] = FLAGS.horizon_length
    
    eval_env = make_env_flat(video=False) # Temp env for initialization
    obs_shape = eval_env.observation_space.shape
    act_shape = eval_env.action_space.shape
    
    example_obs = np.zeros(obs_shape, dtype=np.float32)
    example_action = np.zeros(act_shape, dtype=np.float32)
    
    agent_class = agents[config['agent_name']]
    agent = agent_class.create(FLAGS.seed, example_obs, example_action, config)
    agent = restore_agent(agent, FLAGS.ckpt_dir, FLAGS.ckpt_step)
    eval_env.close()
    
    # 2. Prepare Intervention Vector
    # Vector * Strength
    base_vector = get_intervention_vector(FLAGS.classifier_path, FLAGS.intervention_direction)
    perturbation = base_vector * FLAGS.intervention_strength
    
    # 3. Run Eval
    print(f"\nRunning Intervention: Force {FLAGS.intervention_direction.upper()} (Strength {FLAGS.intervention_strength})")
    
    env_prefix = f"intervention_{FLAGS.intervention_direction}_str{FLAGS.intervention_strength}"
    env = make_env_flat(video=FLAGS.video, video_folder=FLAGS.video_dir, video_name_prefix=env_prefix)
    
    run_intervention_episodes(
        agent=agent,
        env=env,
        rng=rng,
        horizon_length=FLAGS.horizon_length,
        num_episodes=FLAGS.eval_episodes,
        max_steps=FLAGS.max_steps_per_episode,
        perturbation=perturbation
    )
    
    env.close()

if __name__ == "__main__":
    app.run(main)