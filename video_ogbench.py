import os
import ogbench
import gymnasium as gym
from gymnasium.wrappers import RecordVideo

# 1. Headless setup (Required for your cluster environment)
os.environ['MUJOCO_GL'] = 'egl' 

# dataset_name = 'cube-triple-play-singletask-task2-v0'
dataset_name = 'antmaze-large-navigate-singletask-v0'
dataset_name = 'antmaze-large-navigate-v0'
env, train_dataset, val_dataset = ogbench.make_env_and_datasets(
    dataset_name=dataset_name, 
    dataset_dir=".ogbench/data"
)

# --- THE FIX ---
# Gymnasium checks env.metadata['render_modes'] to validate the mode.
# ogbench doesn't set this, so we must set it manually to avoid the AssertionError.

# 1. Force the render mode
env.unwrapped.render_mode = "rgb_array"

# 2. Force the metadata to declare support for rgb_array
# This satisfies the passive_env_checker
if not hasattr(env.unwrapped, 'metadata') or env.unwrapped.metadata is None:
    env.unwrapped.metadata = {}

# We update the dictionary to include 'rgb_array'
env.unwrapped.metadata = dict(env.unwrapped.metadata) # Create a copy to be safe
env.unwrapped.metadata['render_modes'] = ["rgb_array"]

# --- Wrapping ---
print(f"Metadata patched: {env.unwrapped.metadata}")

env = RecordVideo(
    env, 
    video_folder="videos/ogbench_videos", 
    episode_trigger=lambda x: True
)

# Evaluation Loop
ob, info = env.reset() 
done = False
while not done:
    action = env.action_space.sample()
    ob, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

env.close()
print("Video generation complete.")