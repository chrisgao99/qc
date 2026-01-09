import gymnasium as gym
import numpy as np
import mujoco

class CustomFixedStateWrapper(gym.Wrapper):
    def __init__(self, env, fixed_goal=None, fixed_qpos=None):
        super().__init__(env)
        self.fixed_goal = np.array(fixed_goal) if fixed_goal is not None else None
        self.fixed_qpos = np.array(fixed_qpos) if fixed_qpos is not None else None
        
        # Cache the ID of the visual target geom once
        self.target_geom_id = mujoco.mj_name2id(
            self.unwrapped.model, mujoco.mjtObj.mjOBJ_GEOM, 'target'
        )

    def reset(self, **kwargs):
        # 1. Reset normally
        obs, info = self.env.reset(**kwargs)
        
        # 2. Force Logical Goal
        if self.fixed_goal is not None:
            # print("previous goal:", self.unwrapped.cur_goal_xy,info['goal'])
            if hasattr(self.unwrapped, 'goal'):
                self.unwrapped.goal = self.fixed_goal
            if hasattr(self.unwrapped, 'cur_goal_xy'):
                self.unwrapped.cur_goal_xy = self.fixed_goal
            info['goal'][:2] = self.fixed_goal
            # print("new goal:", self.unwrapped.cur_goal_xy)
            # print("new info goal:", info['goal'])

            # 3. Force Visual Goal (Update geom position)
            if self.target_geom_id != -1:
                current_pos = self.unwrapped.model.geom_pos[self.target_geom_id]
                # Update X and Y, preserve Z
                new_pos = current_pos.copy()
                new_pos[:2] = self.fixed_goal[:2]
                self.unwrapped.model.geom_pos[self.target_geom_id] = new_pos

        # 4. Force Start Position
        if self.fixed_qpos is not None:
            current_qpos = self.unwrapped.data.qpos
            if self.fixed_qpos.shape == (2,):
                self.unwrapped.data.qpos[:2] = self.fixed_qpos
            elif self.fixed_qpos.shape == current_qpos.shape:
                self.unwrapped.data.qpos[:] = self.fixed_qpos
            
            self.unwrapped.data.qvel[:] = 0.0 

        # 5. Apply changes to physics engine
        mujoco.mj_forward(self.unwrapped.model, self.unwrapped.data)

        # 6. Re-compute Observation
        if hasattr(self.unwrapped, '_get_obs'):
            obs = self.unwrapped._get_obs()
        else:
            obs, _, _, _, _ = self.env.step(np.zeros(self.action_space.shape))
            
        return obs, info



def get_gc_obs(obs, goal):
    # obs: (..., obs_dim),goal: (..., obs_dim), returns: (..., obs_dim * 2)
    return np.concatenate([obs, goal[..., :2]], axis=-1)

def concat_seq_goal(seq_data, goals):
    # seq_data: (Batch, Horizon, Dim)
    # goals: (Batch, Dim)
    seq_len = seq_data.shape[1]
    goals = goals[..., :2]
    # Expand goals: (B, D) -> (B, 1, D) -> (B, Horizon, D)
    goals_expanded = np.tile(goals[:, None, :], (1, seq_len, 1))
    return np.concatenate([seq_data, goals_expanded], axis=-1)


