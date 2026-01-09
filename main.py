import ogbench
import glob, tqdm, wandb, os, json, random, time, jax
import gym
from absl import app, flags
from ml_collections import config_flags
from log_utils import setup_wandb, get_exp_name, get_flag_dict, CsvLogger

from envs.env_utils import make_env_and_datasets
from envs.ogbench_utils import make_ogbench_env_and_datasets
from envs.robomimic_utils import is_robomimic_env

from utils.flax_utils import save_agent, restore_agent 
from utils.datasets import Dataset, ReplayBuffer

from evaluation import evaluate
from agents import agents
import numpy as np
from utils_ant import get_gc_obs, concat_seq_goal, CustomFixedStateWrapper


if 'CUDA_VISIBLE_DEVICES' in os.environ:
    os.environ['EGL_DEVICE_ID'] = os.environ['CUDA_VISIBLE_DEVICES']
    os.environ['MUJOCO_EGL_DEVICE_ID'] = os.environ['CUDA_VISIBLE_DEVICES']

FLAGS = flags.FLAGS

flags.DEFINE_string('run_group', 'Debug', 'Run group.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('env_name', 'cube-triple-play-singletask-task2-v0', 'Environment (dataset) name.')
flags.DEFINE_string('save_dir', 'exp/', 'Save directory.')

flags.DEFINE_string('restore_path', None, 'Path to restore checkpoint from.')
flags.DEFINE_integer('restore_epoch', None, 'Specific epoch to restore. If None, tries to find the latest.')

flags.DEFINE_list('custom_goal', None, 'Fixed goal coordinates (comma separated, e.g., "20,0").')
flags.DEFINE_list('custom_start', None, 'Fixed start qpos (comma separated, e.g., "0,20,...").')

flags.DEFINE_integer('offline_steps', 1000000, 'Number of online steps.')
flags.DEFINE_integer('online_steps', 1000000, 'Number of online steps.')
flags.DEFINE_integer('buffer_size', 2000000, 'Replay buffer size.')
flags.DEFINE_integer('log_interval', 5000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 100000, 'Evaluation interval.')
flags.DEFINE_integer('save_interval', -1, 'Save interval.')
flags.DEFINE_integer('start_training', 5000, 'when does training start')

flags.DEFINE_integer('utd_ratio', 1, "update to data ratio")

flags.DEFINE_float('discount', 0.99, 'discount factor')

flags.DEFINE_integer('eval_episodes', 50, 'Number of evaluation episodes.')
flags.DEFINE_integer('video_episodes', 0, 'Number of video episodes for each task.')
flags.DEFINE_integer('video_frame_skip', 3, 'Frame skip for videos.')

config_flags.DEFINE_config_file('agent', 'agents/acfql.py', lock_config=False)

flags.DEFINE_float('dataset_proportion', 1.0, "Proportion of the dataset to use")
flags.DEFINE_integer('dataset_replace_interval', 1000, 'Dataset replace interval, used for large datasets because of memory constraints')
flags.DEFINE_string('ogbench_dataset_dir', None, 'OGBench dataset directory')

flags.DEFINE_integer('horizon_length', 5, 'action chunking length.')
flags.DEFINE_bool('sparse', False, "make the task sparse reward")

flags.DEFINE_bool('save_all_online_states', False, "save all trajectories to npy")

class LoggingHelper:
    def __init__(self, csv_loggers, wandb_logger):
        self.csv_loggers = csv_loggers
        self.wandb_logger = wandb_logger
        self.first_time = time.time()
        self.last_time = time.time()

    def log(self, data, prefix, step):
        assert prefix in self.csv_loggers, prefix
        self.csv_loggers[prefix].log(data, step=step)
        self.wandb_logger.log({f'{prefix}/{k}': v for k, v in data.items()}, step=step)

# class CustomFixedStateWrapper(gym.Wrapper):
#     def __init__(self, env, fixed_goal=None, fixed_qpos=None):
#         super().__init__(env)
#         self.fixed_goal = np.array(fixed_goal) if fixed_goal is not None else None
#         self.fixed_qpos = np.array(fixed_qpos) if fixed_qpos is not None else None

#     def reset(self, **kwargs):
#         obs, info = self.env.reset(**kwargs)
        
#         # Force Start State (Qpos)
#         if self.fixed_qpos is not None:
#             if hasattr(self.unwrapped, 'set_state'):
#                 qvel = np.zeros_like(self.unwrapped.data.qvel)
#                 current_qpos = self.unwrapped.data.qpos.copy()
#                 current_qpos[:len(self.fixed_qpos)] = self.fixed_qpos
#                 self.unwrapped.set_state(current_qpos, qvel)
            
#         # Force Goal
#         if self.fixed_goal is not None:
#             if hasattr(self.unwrapped, 'set_goal'):
#                  self.unwrapped.set_goal(self.fixed_goal)
#             elif hasattr(self.unwrapped, 'goal'):
#                  self.unwrapped.goal = self.fixed_goal
            
#             if hasattr(self.unwrapped, 'cur_goal_xy'):
#                  self.unwrapped.cur_goal_xy = self.fixed_goal
#             if hasattr(self.unwrapped, 'target_goal'):
#                  self.unwrapped.target_goal = self.fixed_goal

#         if hasattr(self.unwrapped, '_get_obs'):
#             obs = self.unwrapped._get_obs()
        
#         if 'goal' in info and self.fixed_goal is not None:
#             info['goal'] = self.fixed_goal
#             if isinstance(obs, dict) and 'goal' in obs:
#                 obs['goal'] = self.fixed_goal

#         return obs, info

def main(_):
    exp_name = get_exp_name(FLAGS.seed)
    run = setup_wandb(project='qc', group=FLAGS.run_group, name=exp_name)
    
    FLAGS.save_dir = os.path.join(FLAGS.save_dir, wandb.run.project, FLAGS.run_group, FLAGS.env_name, exp_name)
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    flag_dict = get_flag_dict()

    with open(os.path.join(FLAGS.save_dir, 'flags.json'), 'w') as f:
        json.dump(flag_dict, f)

    config = FLAGS.agent
    
    # data loading
    if FLAGS.ogbench_dataset_dir is not None:
        print("Using custom OGBench dataset directory:", FLAGS.ogbench_dataset_dir)
        # custom ogbench dataset
        assert FLAGS.dataset_replace_interval != 0
        assert FLAGS.dataset_proportion == 1.0
        dataset_idx = 0
        dataset_paths = [
            file for file in sorted(glob.glob(f"{FLAGS.ogbench_dataset_dir}/*.npz")) if '-val.npz' not in file
        ]
        env, eval_env, train_dataset, val_dataset = make_ogbench_env_and_datasets(
            FLAGS.env_name,
            dataset_path=dataset_paths[dataset_idx],
            compact_dataset=False,
        )
    else:
        print("Using default OGBench dataset directory.")
        env, eval_env, train_dataset, val_dataset = make_ogbench_env_and_datasets(FLAGS.env_name)

    # [MODIFIED] Apply CustomFixedStateWrapper to BOTH env and eval_env
    if FLAGS.custom_goal is not None or FLAGS.custom_start is not None:
        print(f"Wrapping both TRAINING env and EVAL env with CustomFixedStateWrapper...")
        c_goal = [float(x) for x in FLAGS.custom_goal] if FLAGS.custom_goal else None
        c_start = [float(x) for x in FLAGS.custom_start] if FLAGS.custom_start else None
        
        if c_goal: print(f"  -> Fixed Goal: {c_goal}")
        if c_start: print(f"  -> Fixed Start: {c_start}")
        
        # Wrap training env
        env = CustomFixedStateWrapper(
            env, 
            fixed_goal=c_goal, 
            fixed_qpos=c_start
        )
        
        # Wrap evaluation env
        eval_env = CustomFixedStateWrapper(
            eval_env, 
            fixed_goal=c_goal, 
            fixed_qpos=c_start
        )

    # --- [START ADDITION] Generate Goals for Offline Dataset ---
    print("Generating goals for dataset (Goal = Final state of trajectory)...")
    
    # 1. Identify where trajectories end
    terminals = train_dataset['terminals'].astype(bool)
    
    # 2. Get the indices of the last steps
    traj_end_idxs = np.nonzero(terminals)[0]
    print(f"Identified {len(traj_end_idxs)} trajectories in the training dataset.")
    
    # 3. Create an array that maps every step 'i' to its trajectory's end index
    step_to_end_idx = np.zeros(len(train_dataset['observations']), dtype=int)
    
    start_idx = 0
    for end_idx in tqdm.tqdm(traj_end_idxs, desc="Relabeling Goals"):
        # Assign the index of the final state to all steps in this trajectory
        step_to_end_idx[start_idx : end_idx + 1] = end_idx
        start_idx = end_idx + 1
        
    # 4. Create the goals array (Same shape as observations)
    train_dataset['goals'] = train_dataset['observations'][step_to_end_idx]
    
    # 5. (Optional) If you have a validation dataset, do the same
    if val_dataset is not None:
        v_terminals = val_dataset['terminals'].astype(bool)
        v_traj_end_idxs = np.nonzero(v_terminals)[0]
        v_step_to_end_idx = np.zeros(len(val_dataset['observations']), dtype=int)
        v_start_idx = 0
        for end_idx in v_traj_end_idxs:
            v_step_to_end_idx[v_start_idx : end_idx + 1] = end_idx
            v_start_idx = end_idx + 1
        val_dataset['goals'] = val_dataset['observations'][v_step_to_end_idx]
    # --- [END ADDITION] ---

    # house keeping
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    obs,info = env.reset()
    
    online_rng, rng = jax.random.split(jax.random.PRNGKey(FLAGS.seed), 2)
    log_step = 0
    
    discount = FLAGS.discount
    config["horizon_length"] = FLAGS.horizon_length

    # handle dataset
    def process_train_dataset(ds):
        ds = Dataset.create(**ds)
        if FLAGS.dataset_proportion < 1.0:
            new_size = int(len(ds['masks']) * FLAGS.dataset_proportion)
            ds = Dataset.create(
                **{k: v[:new_size] for k, v in ds.items()}
            )
        
        if is_robomimic_env(FLAGS.env_name):
            penalty_rewards = ds["rewards"] - 1.0
            ds_dict = {k: v for k, v in ds.items()}
            ds_dict["rewards"] = penalty_rewards
            ds = Dataset.create(**ds_dict)
        
        if FLAGS.sparse:
            sparse_rewards = (ds["rewards"] != 0.0) * -1.0
            ds_dict = {k: v for k, v in ds.items()}
            ds_dict["rewards"] = sparse_rewards
            ds = Dataset.create(**ds_dict)

        return ds
    
    train_dataset = process_train_dataset(train_dataset)
    example_batch = train_dataset.sample(())
    example_obs_gc = get_gc_obs(example_batch['observations'], example_batch['goals'])
    print("example batch goals shape:", example_batch['goals'].shape, example_batch['goals'])
    
    agent_class = agents[config['agent_name']]
    agent = agent_class.create(
        FLAGS.seed,
        example_obs_gc, 
        example_batch['actions'],
        config,
    )

    if FLAGS.restore_path is not None:
        print(f"Restoring agent from {FLAGS.restore_path} (epoch: {FLAGS.restore_epoch})...")
        agent = restore_agent(agent, FLAGS.restore_path, FLAGS.restore_epoch)

    # Setup logging.
    prefixes = ["eval", "env"]
    if FLAGS.offline_steps > 0:
        prefixes.append("offline_agent")
    if FLAGS.online_steps > 0:
        prefixes.append("online_agent")

    logger = LoggingHelper(
        csv_loggers={prefix: CsvLogger(os.path.join(FLAGS.save_dir, f"{prefix}.csv")) 
                    for prefix in prefixes},
        wandb_logger=wandb,
    )

    offline_init_time = time.time()
    # Offline RL
    for i in tqdm.tqdm(range(1, FLAGS.offline_steps + 1)):
        log_step += 1

        if FLAGS.ogbench_dataset_dir is not None and FLAGS.dataset_replace_interval != 0 and i % FLAGS.dataset_replace_interval == 0:
            dataset_idx = (dataset_idx + 1) % len(dataset_paths)
            print(f"Using new dataset: {dataset_paths[dataset_idx]}", flush=True)
            train_dataset, val_dataset = ogbench.make_env_and_datasets(
                FLAGS.env_name,
                dataset_path=dataset_paths[dataset_idx],
                compact_dataset=False,
                dataset_only=True,
                cur_env=env,
            )

            train_dataset = process_train_dataset(train_dataset)
        
        batch = train_dataset.sample_sequence(config['batch_size'], sequence_length=FLAGS.horizon_length, discount=discount)
        batch['observations'] = get_gc_obs(batch['observations'], batch['goals'])
        if 'next_observations' in batch:
            batch['next_observations'] = concat_seq_goal(batch['next_observations'], batch['goals'])
        if 'full_observations' in batch:
             batch['full_observations'] = concat_seq_goal(batch['full_observations'], batch['goals'])

        agent, offline_info = agent.update(batch)

        if i % FLAGS.log_interval == 0:
            logger.log(offline_info, "offline_agent", step=log_step)
        
        # saving
        if FLAGS.save_interval > 0 and i % FLAGS.save_interval == 0:
            save_agent(agent, FLAGS.save_dir, log_step)

        # eval
        if i == FLAGS.offline_steps - 1 or \
            (FLAGS.eval_interval != 0 and i % FLAGS.eval_interval == 0):
            eval_info, _, _ = evaluate(
                agent=agent,
                env=eval_env,
                action_dim=example_batch["actions"].shape[-1],
                num_eval_episodes=FLAGS.eval_episodes,
                num_video_episodes=FLAGS.video_episodes,
                video_frame_skip=FLAGS.video_frame_skip,
            )
            logger.log(eval_info, "eval", step=log_step)

    # transition from offline to online
    print("Converting Replay Buffer to Goal-Conditioned (obs + goal)...")
    
    # --- [START FIX] ---
    dataset_dict = dict(train_dataset)
    
    dataset_dict['observations'] = get_gc_obs(
        dataset_dict['observations'], 
        dataset_dict['goals']
    )
    
    if 'next_observations' in dataset_dict:
        dataset_dict['next_observations'] = get_gc_obs(
            dataset_dict['next_observations'], 
            dataset_dict['goals']
        )

    replay_buffer = ReplayBuffer.create_from_initial_dataset(
        dataset_dict, 
        size=max(FLAGS.buffer_size, train_dataset.size + 1)
    )
        
    ob, info = env.reset()
    current_goal = info.get('goal', None)
    
    action_queue = []
    action_dim = example_batch["actions"].shape[-1]

    # Online RL
    update_info = {}

    from collections import defaultdict
    data = defaultdict(list)
    online_init_time = time.time()
    for i in tqdm.tqdm(range(1, FLAGS.online_steps + 1)):
        log_step += 1
        online_rng, key = jax.random.split(online_rng)

        obs_gc = get_gc_obs(ob, current_goal)
        
        if len(action_queue) == 0:
            action = agent.sample_actions(observations=obs_gc, rng=key)
            action_chunk = np.array(action).reshape(-1, action_dim)
            for action in action_chunk:
                action_queue.append(action)
        action = action_queue.pop(0)
        
        next_ob, int_reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        next_obs_gc = get_gc_obs(next_ob, current_goal)

        if FLAGS.save_all_online_states:
            state = env.get_state()
            data["steps"].append(i)
            data["obs"].append(np.copy(next_ob))
            data["qpos"].append(np.copy(state["qpos"]))
            data["qvel"].append(np.copy(state["qvel"]))
            if "button_states" in state:
                data["button_states"].append(np.copy(state["button_states"]))
        
        env_info = {}
        for key, value in info.items():
            if key.startswith("distance"):
                env_info[key] = value
        logger.log(env_info, "env", step=log_step)

        if 'antmaze' in FLAGS.env_name and (
            'diverse' in FLAGS.env_name or 'play' in FLAGS.env_name or 'umaze' in FLAGS.env_name
        ):
            int_reward = int_reward - 1.0
        elif is_robomimic_env(FLAGS.env_name):
            int_reward = int_reward - 1.0

        if FLAGS.sparse:
            assert int_reward <= 0.0
            int_reward = (int_reward != 0.0) * -1.0

        transition = dict(
            observations=obs_gc,
            actions=action,
            rewards=int_reward,
            terminals=float(done),
            masks=1.0 - terminated,
            next_observations=next_obs_gc,
            goals=current_goal,
        )
        replay_buffer.add_transition(transition)
        
        if done:
            ob, info = env.reset()
            current_goal = info.get('goal', None)
            action_queue = [] 
        else:
            ob = next_ob
            obs_gc = next_obs_gc

        if i >= FLAGS.start_training:
            batch = replay_buffer.sample_sequence(config['batch_size'] * FLAGS.utd_ratio, 
                        sequence_length=FLAGS.horizon_length, discount=discount)
            batch = jax.tree.map(lambda x: x.reshape((
                FLAGS.utd_ratio, config["batch_size"]) + x.shape[1:]), batch)

            agent, update_info["online_agent"] = agent.batch_update(batch)
            
        if i % FLAGS.log_interval == 0:
            for key, info in update_info.items():
                logger.log(info, key, step=log_step)
            update_info = {}

        if i == FLAGS.online_steps - 1 or \
            (FLAGS.eval_interval != 0 and i % FLAGS.eval_interval == 0):
            eval_info, _, _ = evaluate(
                agent=agent,
                env=eval_env,
                action_dim=action_dim,
                num_eval_episodes=FLAGS.eval_episodes,
                num_video_episodes=FLAGS.video_episodes,
                video_frame_skip=FLAGS.video_frame_skip,
            )
            logger.log(eval_info, "eval", step=log_step)

        # saving
        if FLAGS.save_interval > 0 and i % FLAGS.save_interval == 0:
            save_agent(agent, FLAGS.save_dir, log_step)

    end_time = time.time()

    for key, csv_logger in logger.csv_loggers.items():
        csv_logger.close()

    if FLAGS.save_all_online_states:
        c_data = {"steps": np.array(data["steps"]),
                 "qpos": np.stack(data["qpos"], axis=0), 
                 "qvel": np.stack(data["qvel"], axis=0), 
                 "obs": np.stack(data["obs"], axis=0), 
                 "offline_time": online_init_time - offline_init_time,
                 "online_time": end_time - online_init_time,
        }
        if len(data["button_states"]) != 0:
            c_data["button_states"] = np.stack(data["button_states"], axis=0)
        np.savez(os.path.join(FLAGS.save_dir, "data.npz"), **c_data)

    with open(os.path.join(FLAGS.save_dir, 'token.tk'), 'w') as f:
        f.write(run.url)

if __name__ == '__main__':
    app.run(main)