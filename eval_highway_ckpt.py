# eval_highway_ckpt.py
#
# Evaluate a trained AC-FQL chunk policy on highway-state:
# - Load agent config from agents/acfql.py
# - Restore checkpoint params_{step}.pkl
# - Run evaluation episodes, report stats
# - Optionally record videos with RecordVideo

import os
from absl import app, flags
from ml_collections import config_flags

import jax
import jax.numpy as jnp
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RecordVideo

from utils_env import create_stateenv, config_state2
from utils.flax_utils import restore_agent
from agents import agents


FLAGS = flags.FLAGS

# ---------- basic flags ----------
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('env_name', 'highway-state', 'Environment name (for bookkeeping).')

# agent config file (same style as main2.py)
config_flags.DEFINE_config_file(
    'agent', 'agents/acfql.py', lock_config=False
)

# checkpoint info
flags.DEFINE_string(
    'ckpt_dir',
    None,
    'Directory containing params_*.pkl, e.g. /p/yufeng/qc/exp/.../sd0002...'
)
flags.DEFINE_integer(
    'ckpt_step',
    None,
    'Checkpoint step to load, e.g. 100000 or 200000 (for params_{step}.pkl).'
)

# eval settings
flags.DEFINE_integer('eval_episodes', 50, 'Number of evaluation episodes.')
flags.DEFINE_integer(
    'max_steps_per_episode',
    2000,
    'Safety cap on episode length.'
)
flags.DEFINE_integer(
    'horizon_length',
    7,
    'Action chunk length h; MUST match training config.'
)

# video settings
flags.DEFINE_integer(
    'video_episodes',
    0,
    'How many episodes to record as video (separate from eval_episodes).'
)
flags.DEFINE_string(
    'video_dir',
    'videos/highway_eval',
    'Directory to save videos (if video_episodes > 0).'
)

# stats saving
flags.DEFINE_string(
    'stats_path',
    None,
    'If set, save eval stats npz to this path.'
)


# ---------- wrappers ----------

class FlattenObsWrapper(gym.ObservationWrapper):
    """
    Flattens observations from shape (10, 7) -> (70,) for Highway state env.
    Same as in main2.py, so the agent sees the same obs space as in training.
    """
    def __init__(self, env):
        super().__init__(env)
        orig_shape = env.observation_space.shape
        flat_dim = int(np.prod(orig_shape))
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(flat_dim,),
            dtype=np.float32,
        )

    def observation(self, observation):
        return np.asarray(observation, dtype=np.float32).reshape(-1)


def make_env_flat(video=False, video_folder=None, video_name_prefix='highway'):
    """
    Create highway-state env with optional RecordVideo and flatten obs.
    Wrapper order: raw_env -> RecordVideo (optional) -> FlattenObsWrapper
    """
    env = create_stateenv(config_state2)

    if video:
        os.makedirs(video_folder, exist_ok=True)
        env = RecordVideo(
            env,
            video_folder=video_folder,
            name_prefix=video_name_prefix,
            episode_trigger=lambda ep_id: True,  # record every episode in this env
            disable_logger=True,
        )

    env = FlattenObsWrapper(env)
    return env


# ---------- evaluation logic ----------

def run_eval_episodes(agent, env, rng, horizon_length, num_episodes, max_steps):
    """
    Evaluate the chunk policy:
    - agent.sample_actions returns a flattened chunk of shape (h * action_dim,)
    - We reshape to (h, action_dim) and execute sub-actions sequentially.
    """
    returns = []
    lengths = []
    success_flags = []

    action_dim = env.action_space.shape[-1]

    for ep in range(num_episodes):
        print(f"\n=== Eval Episode {ep + 1} / {num_episodes} ===")
        obs, _ = env.reset(seed=int(jax.random.randint(rng, (), 0, 2**31 - 1)))
        done = False
        ep_return = 0.0
        ep_len = 0
        action_queue = []

        while not done and ep_len < max_steps:
            # sample a new chunk if queue is empty
            if len(action_queue) == 0:
                rng, key = jax.random.split(rng)
                chunk = agent.sample_actions(observations=obs, rng=key)
                # chunk: (h * action_dim,) -> (h, action_dim)
                chunk = np.array(chunk).reshape(horizon_length, action_dim)
                action_queue = [a for a in chunk]
                # print("new action chunk is:", action_queue)

            action = action_queue.pop(0)

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)

            ep_return += float(reward)
            ep_len += 1
            obs = next_obs

            if done:
                break

        returns.append(ep_return)
        lengths.append(ep_len)
        # 这里和你之前脚本保持一致：truncated 视为成功
        success_flags.append(bool(truncated))

        print(
            f"[Eval] Episode {ep + 1}/{num_episodes} - "
            f"Return: {ep_return:.2f}, Len: {ep_len}, "
            f"terminated={terminated}, truncated={truncated}"
        )

    return np.array(returns), np.array(lengths), np.array(success_flags)


def main(_):
    assert FLAGS.ckpt_dir is not None, "--ckpt_dir must be set"
    assert FLAGS.ckpt_step is not None, "--ckpt_step must be set"

    # basic seeding
    rng = jax.random.PRNGKey(FLAGS.seed)

    # load agent config
    config = FLAGS.agent
    config["horizon_length"] = FLAGS.horizon_length

    # create eval env (no video) and get shapes
    eval_env = make_env_flat(video=False)
    obs_shape = eval_env.observation_space.shape
    act_shape = eval_env.action_space.shape

    print("Obs shape:", obs_shape)
    print("Act shape:", act_shape)

    # dummy example obs/actions to init agent
    example_obs = np.zeros(obs_shape, dtype=np.float32)
    example_action = np.zeros(act_shape, dtype=np.float32)

    # create and restore agent
    agent_class = agents[config['agent_name']]
    agent = agent_class.create(
        FLAGS.seed,
        example_obs,
        example_action,
        config,
    )

    print(f"Restoring agent from {FLAGS.ckpt_dir}, step {FLAGS.ckpt_step}")
    agent = restore_agent(agent, FLAGS.ckpt_dir, FLAGS.ckpt_step)

    # ---------- evaluation without video ----------
    print("=== Running evaluation episodes (no video) ===")
    returns, lengths, success_flags = run_eval_episodes(
        agent=agent,
        env=eval_env,
        rng=rng,
        horizon_length=FLAGS.horizon_length,
        num_episodes=FLAGS.eval_episodes,
        max_steps=FLAGS.max_steps_per_episode,
    )

    print("\n=== Eval Summary ===")
    print("Episodes:", len(returns))
    print("Average return:", float(returns.mean()))
    print("Std return:", float(returns.std()))
    print("Average length:", float(lengths.mean()))
    print("Std length:", float(lengths.std()))
    print("Success count (truncated=True):", int(success_flags.sum()))

    # optionally save stats
    if FLAGS.stats_path is not None:
        os.makedirs(os.path.dirname(FLAGS.stats_path), exist_ok=True)
        np.savez(
            FLAGS.stats_path,
            returns=returns,
            lengths=lengths,
            success=success_flags,
        )
        print("Saved stats to:", FLAGS.stats_path)

    # ---------- separate video episodes ----------
    if FLAGS.video_episodes > 0:
        print("\n=== Recording video episodes ===")
        video_env = make_env_flat(
            video=True,
            video_folder=FLAGS.video_dir,
            video_name_prefix=f"highway_step{FLAGS.ckpt_step}",
        )

        _ = run_eval_episodes(
            agent=agent,
            env=video_env,
            rng=rng,
            horizon_length=FLAGS.horizon_length,
            num_episodes=FLAGS.video_episodes,
            max_steps=FLAGS.max_steps_per_episode,
        )

        video_env.close()
        print("Videos saved to:", FLAGS.video_dir)

    eval_env.close()


if __name__ == "__main__":
    app.run(main)
