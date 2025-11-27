# eval_ogbench_ckpt.py
#
# Evaluate a trained AC-FQL agent on OGBench envs (e.g., cube-triple-play-singletask-task2-v0):
# - Load agent config from agents/acfql.py
# - Restore checkpoint params_{step}.pkl
# - Use evaluation.evaluate to run episodes and (optionally) record videos

import os
import glob
from absl import app, flags
from ml_collections import config_flags

import jax
import numpy as np
import imageio.v2 as imageio
from envs.ogbench_utils import make_ogbench_env_and_datasets
from utils.flax_utils import restore_agent
from utils.datasets import Dataset
from evaluation import evaluate
from agents import agents

FLAGS = flags.FLAGS

# ---------- basic flags ----------

flags.DEFINE_integer('seed', 0, 'Random seed.')

flags.DEFINE_string(
    'env_name',
    'cube-triple-play-singletask-task2-v0',
    'OGBench environment name.'
)
flags.DEFINE_integer(
    'horizon_length',
    5,
    'Action chunk length h (should match training).'
)
# agent config file (same style as main2.py)
config_flags.DEFINE_config_file(
    'agent', 'agents/acfql.py', lock_config=False
)

# checkpoint info
flags.DEFINE_string(
    'ckpt_dir',
    None,
    'Directory containing params_*.pkl, e.g. /p/yufeng/qc/exp/...'
)
flags.DEFINE_integer(
    'ckpt_step',
    None,
    'Checkpoint step to load, e.g. 100000 or 200000 (for params_{step}.pkl).'
)

# OGBench dataset dir (same as训练时)
flags.DEFINE_string(
    'ogbench_dataset_dir',
    None,
    'Directory with cube-triple-play .npz files (same as used for training).'
)

# eval settings
flags.DEFINE_integer('eval_episodes', 50, 'Number of evaluation episodes.')
flags.DEFINE_integer('video_episodes', 0, 'Number of video episodes.')
flags.DEFINE_integer('video_frame_skip', 3, 'Frame skip for videos.')

# stats saving
flags.DEFINE_string(
    'stats_path',
    None,
    'If set, save eval stats (eval_info dict) as npz to this path.'
)


def main(_):
    assert FLAGS.ckpt_dir is not None, "--ckpt_dir must be set"
    assert FLAGS.ckpt_step is not None, "--ckpt_step must be set"
    assert FLAGS.ogbench_dataset_dir is not None, "--ogbench_dataset_dir must be set"

    _ = jax.random.PRNGKey(FLAGS.seed)

    # ---------- pick one dataset file just to construct env ----------
    dataset_paths = [
        f for f in sorted(glob.glob(f"{FLAGS.ogbench_dataset_dir}/*.npz"))
        if "-val.npz" not in f
    ]
    assert len(dataset_paths) > 0, f"No train datasets found in {FLAGS.ogbench_dataset_dir}"
    dataset_path = dataset_paths[0]
    print("Using dataset for env creation:", dataset_path)

    # make env and datasets
    env, eval_env, train_dataset, val_dataset = make_ogbench_env_and_datasets(
        FLAGS.env_name,
        dataset_path=dataset_path,
        compact_dataset=False,
    )

    # ---------- load agent config & build agent ----------
    config = FLAGS.agent
    # 确保和训练时的 horizon_length 一致
    config["horizon_length"] = FLAGS.horizon_length

    # 这里 train_dataset 可能是 dict，需要先包成 Dataset
    if isinstance(train_dataset, dict):
        train_dataset = Dataset.create(**train_dataset)

    # 和 main2.py 一样，用 sample(()) 拿一个 example_batch
    example_batch = train_dataset.sample(())
    obs_example = example_batch["observations"]
    act_example = example_batch["actions"]
    action_dim = act_example.shape[-1]

    print("Obs example shape:", obs_example.shape)
    print("Act example shape:", act_example.shape)
    print("Action dim:", action_dim)

    agent_class = agents[config["agent_name"]]
    agent = agent_class.create(
        FLAGS.seed,
        obs_example,
        act_example,
        config,
    )

    print(f"Restoring agent from {FLAGS.ckpt_dir}, step {FLAGS.ckpt_step}")
    agent = restore_agent(agent, FLAGS.ckpt_dir, FLAGS.ckpt_step)

    # ---------- evaluation ----------
    print("=== Running evaluate() ===")
    eval_info, videos, video_paths = evaluate(
        agent=agent,
        env=eval_env,
        action_dim=action_dim,
        num_eval_episodes=FLAGS.eval_episodes,
        num_video_episodes=FLAGS.video_episodes,
        video_frame_skip=FLAGS.video_frame_skip,
    )

    print("\n=== Eval Summary (eval_info) ===")
    for k, v in eval_info.items():
        print(f"{k}: {v}")

    if FLAGS.video_episodes > 0 and videos is not None and len(videos) > 0:
        out_dir = os.path.join(
            FLAGS.ckpt_dir,
            f"videos_step{FLAGS.ckpt_step}"
        )
        os.makedirs(out_dir, exist_ok=True)

        print(f"\nSaving {len(videos)} videos to: {out_dir}")
        for i, vid in enumerate(videos):
            # vid: shape (T, H, W, 3), uint8
            out_path = os.path.join(out_dir, f"episode_{i:03d}.mp4")
            # 这里用 imageio 写 mp4，fps 随便取个 30，你可以改
            with imageio.get_writer(out_path, fps=30) as writer:
                for frame in vid:
                    writer.append_data(frame)
            print("  saved:", out_path)
    else:
        print("\nNo videos returned (videos is None or empty).")


if __name__ == "__main__":
    app.run(main)
