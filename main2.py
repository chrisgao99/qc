import glob, tqdm, wandb, os, json, random, time, jax
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

import gymnasium as gym
import numpy as np  # you already have this, just confirming
from utils_env import create_stateenv, config_state2  # same as in your expert script

if 'CUDA_VISIBLE_DEVICES' in os.environ:
    os.environ['EGL_DEVICE_ID'] = os.environ['CUDA_VISIBLE_DEVICES']
    os.environ['MUJOCO_EGL_DEVICE_ID'] = os.environ['CUDA_VISIBLE_DEVICES']

FLAGS = flags.FLAGS

flags.DEFINE_string('run_group', 'highway', 'Run group.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('env_name', 'cube-triple-play-singletask-task2-v0', 'Environment (dataset) name.')
flags.DEFINE_string('save_dir', 'exp/', 'Save directory.')

flags.DEFINE_integer('offline_steps', 1000000, 'Number of offline steps.')
flags.DEFINE_integer('online_steps', 1000000, 'Number of online steps.')
flags.DEFINE_integer('buffer_size', 2000000, 'Replay buffer size.')
flags.DEFINE_integer('log_interval', 5000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 100000, 'Evaluation interval.')
flags.DEFINE_integer('save_interval', 100000, 'Save interval.')
flags.DEFINE_integer('start_training', 5000, 'when does training start')

flags.DEFINE_string('restore_path', None, 'Directory that contains params_XXXXXX.pkl.')
flags.DEFINE_integer('restore_epoch', None, 'Epoch (step) of params_XXXXXX.pkl to load.')


flags.DEFINE_integer('utd_ratio', 1, "update to data ratio")

flags.DEFINE_float('discount', 0.99, 'discount factor')

flags.DEFINE_integer('eval_episodes', 50, 'Number of evaluation episodes.')
flags.DEFINE_integer('video_episodes', 0, 'Number of video episodes for each task.')
flags.DEFINE_integer('video_frame_skip', 3, 'Frame skip for videos.')

config_flags.DEFINE_config_file('agent', 'agents/acfql.py', lock_config=False)

flags.DEFINE_float('dataset_proportion', 1.0, "Proportion of the dataset to use")
flags.DEFINE_integer('dataset_replace_interval', 1000, 'Dataset replace interval, used for large datasets because of memory constraints')
flags.DEFINE_string('ogbench_dataset_dir', None, 'OGBench dataset directory')
flags.DEFINE_string(
    'highway_dataset_path',
    None,
    'Path to Highway expert dataset (.npz) with keys [observations, actions, rewards, terminals].'
)

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

class FlattenObsWrapper(gym.ObservationWrapper):
    """
    Flattens observations from shape (10, 7) -> (70,) for Highway state env.
    Makes env obs match the flattened obs used in the offline dataset.
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

    # ----- helper: sample batch with lane-change filtering (for Highway) -----
def sample_lane_filtered_batch(train_dataset, batch_size, sequence_length, discount):
    """
    和 Dataset.sample_sequence 行为尽量一致，只是：
    - 如果有 'lanes'：
        用 lanes 做窗口级的筛选与加权采样（优先包含 lane change 的窗口）
    - 如果没有 'lanes'：
        直接退回原始 train_dataset.sample_sequence
    """

    # ---------- 如果没有 lanes，退回原始逻辑 ----------
    if "lanes" not in train_dataset:
        # 和原来的完全一样
        return train_dataset.sample_sequence(batch_size, sequence_length, discount)

    # ---------- 第一次调用：预计算所有窗口是否包含变道 ----------
    if not hasattr(sample_lane_filtered_batch, "initialized"):
        lanes_all = np.asarray(train_dataset["lanes"])   # (N,)
        N = lanes_all.shape[0]
        h = sequence_length

        if N < h:
            raise ValueError(f"数据长度 N={N} 小于窗口长度 h={h}，无法采样序列。")

        num_windows = N - h + 1
        lane_change_flags = np.zeros(num_windows, dtype=bool)

        for t in range(num_windows):
            w = lanes_all[t:t + h]            # 长度 h
            lane_change_flags[t] = np.any(w[:-1] != w[1:])  # 窗口内任意一步有变化

        idx_change = np.where(lane_change_flags)[0]
        idx_nochange = np.where(~lane_change_flags)[0]

        sample_lane_filtered_batch.lanes_all = lanes_all
        sample_lane_filtered_batch.idx_change = idx_change
        sample_lane_filtered_batch.idx_nochange = idx_nochange
        sample_lane_filtered_batch.num_windows = num_windows
        sample_lane_filtered_batch.h = h
        sample_lane_filtered_batch.initialized = True

        # 打印统计信息
        num_change = len(idx_change)
        num_nochange = len(idx_nochange)
        total = num_change + num_nochange
        print(f"[Lane stats] window length h={h}")
        print(f"  total windows      : {total}")
        print(f"  with lane change   : {num_change} ({num_change / total:.2%})")
        print(f"  without lane change: {num_nochange} ({num_nochange / total:.2%})")

    # ---------- 采样一批起点 idxs（起点在 0..(num_windows-1)） ----------
    idx_change = sample_lane_filtered_batch.idx_change
    idx_nochange = sample_lane_filtered_batch.idx_nochange
    num_change_total = len(idx_change)
    num_nochange_total = len(idx_nochange)
    h = sample_lane_filtered_batch.h

    if num_change_total == 0 or num_nochange_total == 0:
        # 极端情况：全是直行或全是变道 → 回到均匀采样
        max_start = train_dataset.size - sequence_length + 1
        idxs = np.random.randint(max_start, size=batch_size)
    else:
        # 有变道窗口: 权重 1.0
        # 直行窗口  : 权重 0.05
        alpha = 1.0
        beta = 0.05
        total_weight = alpha * num_change_total + beta * num_nochange_total
        p_change = (alpha * num_change_total) / total_weight

        starts = []
        for _ in range(batch_size):
            if np.random.rand() < p_change:
                t = np.random.choice(idx_change)
            else:
                t = np.random.choice(idx_nochange)
            starts.append(t)
        idxs = np.array(starts, dtype=np.int32)  # (batch_size,)

    # ---------- 从这里开始，复制原始 sample_sequence 逻辑，只是用我们自己的 idxs ----------

    # data 里每个 key 取的是“起点对应的单步数据”（例如 observations 是 (B, 70)）
    data = {k: v[idxs] for k, v in train_dataset.items()}

    # 生成所有时间步的全局 index
    all_idxs = idxs[:, None] + np.arange(sequence_length)[None, :]  # (batch_size, sequence_length)
    all_idxs = all_idxs.flatten()                                   # (batch_size * sequence_length,)

    # 批量抓取序列
    obs_all = train_dataset['observations']
    next_obs_all = train_dataset['next_observations']
    act_all = train_dataset['actions']
    rew_all = train_dataset['rewards']
    mask_all = train_dataset['masks']
    term_all = train_dataset['terminals']

    batch_observations = obs_all[all_idxs].reshape(
        batch_size, sequence_length, *obs_all.shape[1:]
    )
    batch_next_observations = next_obs_all[all_idxs].reshape(
        batch_size, sequence_length, *next_obs_all.shape[1:]
    )
    batch_actions = act_all[all_idxs].reshape(
        batch_size, sequence_length, *act_all.shape[1:]
    )
    batch_rewards = rew_all[all_idxs].reshape(
        batch_size, sequence_length, *rew_all.shape[1:]
    )
    batch_masks = mask_all[all_idxs].reshape(
        batch_size, sequence_length, *mask_all.shape[1:]
    )
    batch_terminals = term_all[all_idxs].reshape(
        batch_size, sequence_length, *term_all.shape[1:]
    )

    # 计算 next_actions（简单地往后移一格，并在末尾截断）
    next_action_idxs = np.minimum(all_idxs + 1, train_dataset.size - 1)
    batch_next_actions = act_all[next_action_idxs].reshape(
        batch_size, sequence_length, *act_all.shape[1:]
    )

    # 计算折扣累计 reward / mask / terminal / valid（照原版）
    rewards = np.zeros((batch_size, sequence_length), dtype=float)
    masks = np.ones((batch_size, sequence_length), dtype=float)
    terminals = np.zeros((batch_size, sequence_length), dtype=float)
    valid = np.ones((batch_size, sequence_length), dtype=float)

    rewards[:, 0] = batch_rewards[:, 0].squeeze()
    masks[:, 0] = batch_masks[:, 0].squeeze()
    terminals[:, 0] = batch_terminals[:, 0].squeeze()

    discount_powers = discount ** np.arange(sequence_length)
    for i in range(1, sequence_length):
        rewards[:, i] = rewards[:, i-1] + batch_rewards[:, i].squeeze() * discount_powers[i]
        masks[:, i] = np.minimum(masks[:, i-1], batch_masks[:, i].squeeze())
        terminals[:, i] = np.maximum(terminals[:, i-1], batch_terminals[:, i].squeeze())
        valid[:, i] = 1.0 - terminals[:, i-1]

    # 组织 observations / next_observations 形状（跟原函数保持一致）
    if len(batch_observations.shape) == 5:
        # 图像情况 (batch, seq, h, w, c) → (batch, h, w, seq, c)
        observations = batch_observations.transpose(0, 2, 3, 1, 4)
        next_observations = batch_next_observations.transpose(0, 2, 3, 1, 4)
    else:
        # 状态情况，保持 (batch, seq, state_dim)
        observations = batch_observations
        next_observations = batch_next_observations

    actions = batch_actions            # (batch, seq, act_dim)
    next_actions = batch_next_actions  # (batch, seq, act_dim)

    # 返回格式完全模仿原 sample_sequence
    return dict(
        observations=data['observations'].copy(),  # 起点 obs: (batch, obs_dim) = (B, 70)
        full_observations=observations,           # 全序列 obs: (batch, seq, obs_dim) = (B, h, 70)
        actions=actions,                          # (B, h, act_dim)
        masks=masks,                              # (B, h)
        rewards=rewards,                          # (B, h)
        terminals=terminals,                      # (B, h)
        valid=valid,                              # (B, h)
        next_observations=next_observations,      # (B, h, obs_dim)
        next_actions=next_actions,                # (B, h, act_dim)
    )


def main(_):
    exp_name = get_exp_name(FLAGS.seed)
    run = setup_wandb(project='qc', group=FLAGS.run_group, name=exp_name)
    
    FLAGS.save_dir = os.path.join(FLAGS.save_dir, wandb.run.project, FLAGS.run_group, FLAGS.env_name, exp_name)
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    flag_dict = get_flag_dict()
    print("Flags:", flag_dict )

    with open(os.path.join(FLAGS.save_dir, 'flags.json'), 'w') as f:
        json.dump(flag_dict, f)

    config = FLAGS.agent
    print("Agent config:", config)
    print(FLAGS.highway_dataset_path)
    if FLAGS.highway_dataset_path is None:
        FLAGS.highway_dataset_path = "/p/yufeng/qc/highway_expert_30000.npz"
    
    # data loading
        # ---------------- data loading ----------------
        # Priority: Highway dataset -> OGBench -> default make_env_and_datasets
    if FLAGS.highway_dataset_path is not None:
        print(f"Loading Highway dataset from: {FLAGS.highway_dataset_path}")

        # 1) Create Highway envs with flattened observations
        raw_env = create_stateenv(config_state2)
        raw_eval_env = create_stateenv(config_state2)
        env = FlattenObsWrapper(raw_env)
        eval_env = FlattenObsWrapper(raw_eval_env)

        # 2) Load offline dataset
        data = np.load(FLAGS.highway_dataset_path)
        observations = data["observations"].astype(np.float32)   # (N, 10, 7)
        actions = data["actions"].astype(np.float32)             # (N, 2)
        rewards = data["rewards"].astype(np.float32)             # (N,)
        terminals_bool = data["terminals"].astype(bool)          # (N,)
        lanes = data["lanes"].astype(np.int32)                    # (N,)

        N = observations.shape[0]

        # Flatten obs: (N, 10, 7) -> (N, 70)
        flat_obs = observations.reshape(N, -1)                   # (N, 70)
        print("Highway flat obs shape:", flat_obs.shape)

        # 3) Build next_observations by a simple shift
        #    For t < N-1: next_obs[t] = obs[t+1]
        #    For t == N-1: next_obs[N-1] = obs[N-1] (won't matter because mask will be 0 if terminal)
        next_flat_obs = np.empty_like(flat_obs)
        next_flat_obs[:-1] = flat_obs[1:]
        next_flat_obs[-1] = flat_obs[-1]

        # 4) Masks: 1 - terminals
        terminals = terminals_bool.astype(np.float32)  # (N,)
        masks = 1.0 - terminals                        # (N,)

        # 5) Build raw dict that Dataset.create expects
        train_dataset = dict(
            observations=flat_obs,          # (N, 70)
            next_observations=next_flat_obs,# (N, 70)
            actions=actions,                # (N, 2)
            rewards=rewards,                # (N,)
            terminals=terminals,            # (N,)
            masks=masks,                    # (N,)
            lanes=lanes,                    # (N,)
        )
        # For now, reuse the same data as 'val'
        val_dataset = train_dataset



    elif FLAGS.ogbench_dataset_dir is not None and FLAGS.ogbench_dataset_dir != "":
        # ===== original OGBench path =====
        assert FLAGS.dataset_replace_interval != 0
        assert FLAGS.dataset_proportion == 1.0
        dataset_idx = 0
        dataset_paths = [
            file for file in sorted(glob.glob(f"{FLAGS.ogbench_dataset_dir}/*.npz"))
            if '-val.npz' not in file
        ]
        env, eval_env, train_dataset, val_dataset = make_ogbench_env_and_datasets(
            FLAGS.env_name,
            dataset_path=dataset_paths[dataset_idx],
            compact_dataset=False,
        )

    else:
        # ===== generic env loader =====
        print("Using generic make_env_and_datasets")
        env, eval_env, train_dataset, val_dataset = make_env_and_datasets(FLAGS.env_name)


    # house keeping
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    online_rng, rng = jax.random.split(jax.random.PRNGKey(FLAGS.seed), 2)
    log_step = 0
    
    discount = FLAGS.discount
    config["horizon_length"] = FLAGS.horizon_length

    # handle dataset
    def process_train_dataset(ds):
        """
        Generic processing:
        - Wrap dict into Dataset
        - Optionally sub-sample by dataset_proportion
        - Handle robomimic reward shift
        - Handle sparse reward shaping
        """
        ds = Dataset.create(**ds)

        if FLAGS.dataset_proportion < 1.0:
            new_size = int(len(ds['masks']) * FLAGS.dataset_proportion)
            ds = Dataset.create(
                **{k: v[:new_size] for k, v in ds.items()}
            )

        if is_robomimic_env(FLAGS.env_name):
            print("Adjusting rewards for Robomimic environment.")
            penalty_rewards = ds["rewards"] - 1.0
            ds_dict = {k: v for k, v in ds.items()}
            ds_dict["rewards"] = penalty_rewards
            ds = Dataset.create(**ds_dict)

        if FLAGS.sparse:
            # Create a new dataset with modified rewards instead of trying to modify the frozen one
            sparse_rewards = (ds["rewards"] != 0.0) * -1.0
            ds_dict = {k: v for k, v in ds.items()}
            ds_dict["rewards"] = sparse_rewards
            ds = Dataset.create(**ds_dict)

        return ds


    
    train_dataset = process_train_dataset(train_dataset)
    example_batch = train_dataset.sample(())
    
    agent_class = agents[config['agent_name']]

    agent = agent_class.create(
        FLAGS.seed,
        example_batch['observations'],
        example_batch['actions'],
        config,
    )

    # ---- restore checkpoint（if set）----
    if FLAGS.restore_path is not None:
        print(f"Restoring agent from {FLAGS.restore_path}, epoch {FLAGS.restore_epoch}")
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
            train_dataset, val_dataset = make_ogbench_env_and_datasets(
                FLAGS.env_name,
                dataset_path=dataset_paths[dataset_idx],
                compact_dataset=False,
                dataset_only=True,
                cur_env=env,
            )
            train_dataset = process_train_dataset(train_dataset)

        # batch = train_dataset.sample_sequence(config['batch_size'], sequence_length=FLAGS.horizon_length, discount=discount)
        batch = sample_lane_filtered_batch(
            train_dataset,
            batch_size=config['batch_size'],
            sequence_length=FLAGS.horizon_length,
            discount=discount,
        )

        agent, offline_info = agent.update(batch)

        if i % FLAGS.log_interval == 0:
            logger.log(offline_info, "offline_agent", step=log_step)
        
        # saving
        if FLAGS.save_interval > 0 and i % FLAGS.save_interval == 0:
            save_agent(agent, FLAGS.save_dir, log_step)

        # eval
        if i == FLAGS.offline_steps - 1 or \
            (FLAGS.eval_interval != 0 and i % FLAGS.eval_interval == 0):
            # during eval, the action chunk is executed fully
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
    # 注意：在线 replay buffer 不需要 lanes，把它从初始数据里去掉，避免 key mismatch
    init_dataset = dict(train_dataset)
    if "lanes" in init_dataset:
        init_dataset.pop("lanes")

    replay_buffer = ReplayBuffer.create_from_initial_dataset(
        init_dataset,
        size=max(FLAGS.buffer_size, train_dataset.size + 1),
    )

        
    ob, _ = env.reset()
    
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
        
        # during online rl, the action chunk is executed fully
        if len(action_queue) == 0:
            action = agent.sample_actions(observations=ob, rng=key)

            action_chunk = np.array(action).reshape(-1, action_dim)
            for action in action_chunk:
                action_queue.append(action)
        # print("current action queue has these actions:", action_queue)
        action = action_queue.pop(0)
        
        next_ob, int_reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if FLAGS.save_all_online_states:
            state = env.get_state()
            data["steps"].append(i)
            data["obs"].append(np.copy(next_ob))
            data["qpos"].append(np.copy(state["qpos"]))
            data["qvel"].append(np.copy(state["qvel"]))
            if "button_states" in state:
                data["button_states"].append(np.copy(state["button_states"]))
        
        # logging useful metrics from info dict
        env_info = {}
        for key, value in info.items():
            if key.startswith("distance"):
                env_info[key] = value
        # always log this at every step
        logger.log(env_info, "env", step=log_step)

        if 'antmaze' in FLAGS.env_name and (
            'diverse' in FLAGS.env_name or 'play' in FLAGS.env_name or 'umaze' in FLAGS.env_name
        ):
            # Adjust reward for D4RL antmaze.
            int_reward = int_reward - 1.0
        elif is_robomimic_env(FLAGS.env_name):
            # Adjust online (0, 1) reward for robomimic
            int_reward = int_reward - 1.0

        if FLAGS.sparse:
            assert int_reward <= 0.0
            int_reward = (int_reward != 0.0) * -1.0

        transition = dict(
            observations=ob,
            actions=action,
            rewards=int_reward,
            terminals=float(done),
            masks=1.0 - terminated,
            next_observations=next_ob,
        )
        replay_buffer.add_transition(transition)
        
        # done
        if done:
            ob, _ = env.reset()
            action_queue = []  # reset the action queue
        else:
            ob = next_ob

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
