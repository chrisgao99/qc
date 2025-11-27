from collections import deque
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from highway_env import utils
from highway_env.envs import HighwayEnv
from highway_env.vehicle.controller import ControlledVehicle
from gymnasium.wrappers import RecordVideo


class CustomHighwayEnv(HighwayEnv):
    """
    A custom highway driving environment with a penalty for frequent steering changes.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.steering_history = None  # Initialize steering history deque

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        # config.update(
        #     {
        #         "steering_change_reward": -0.2,  # Penalty for frequent steering changes
        #         "steering_history_length": 5,  # Number of timesteps to track steering actions
        #         "off_road_reward": -2.0,  # Reward for going off-road
        #         "collision_reward": -2.0,  # Reward for collisions
        #         "right_lane_reward": 0.0,  # Reward for rightmost lane
        #         "normalize_reward": False,  # Disable reward normalization by default
        #         "offroad_terminal": True,  # Terminate episode if off-road
        #     }
        # )
        return config

    def _reset(self) -> None:
        super()._reset()
        # Initialize deque for steering history
        self.steering_history = deque(maxlen=self.config["steering_history_length"])

    def _rewards(self, action) -> dict[str, float]:
        """
        Calculate rewards including a penalty for frequent steering changes.
        :param action: the last action performed (steering angle as action[0])
        :return: dictionary of reward components
        """
        neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
        lane = (
            self.vehicle.target_lane_index[2]
            if isinstance(self.vehicle, ControlledVehicle)
            else self.vehicle.lane_index[2]
        )

        lane_index = self.vehicle.lane_index
        current_lane = self.road.network.get_lane(lane_index)
        long, lateral_offset = current_lane.local_coordinates(self.vehicle.position)
        lateral_factor = np.clip(1 - 2 * abs(lateral_offset) / current_lane.width, 0.0, 1.0)
        # Use forward speed rather than speed
        forward_speed = self.vehicle.speed * np.cos(self.vehicle.heading)
        
        # Check if 'reward_speed_range' is in config to avoid KeyError
        high_speed_reward = 0.0
        if "reward_speed_range" in self.config:
            high_speed_reward = (
                (forward_speed - self.config["reward_speed_range"][0]) * 0.1
                if forward_speed < self.config["reward_speed_range"][1]
                else 0.1 * (self.config["reward_speed_range"][1] - self.config["reward_speed_range"][0])
            ) * lateral_factor

        # Calculate steering change penalty
        steering_penalty = 0.0
        if len(self.steering_history) > 1:
            # Sum of absolute differences between consecutive steering actions
            steering_changes = [
                abs(self.steering_history[i] - self.steering_history[i - 1])
                for i in range(1, len(self.steering_history))
            ]
            steering_penalty = np.mean(steering_changes) if steering_changes else 0.0
            # Normalize penalty to [0, 1] range for consistency
            steering_penalty = np.clip(steering_penalty, 0, 1)

        return {
            "collision_reward": float(self.vehicle.crashed),
            "right_lane_reward": lane / max(len(neighbours) - 1, 1),
            "high_speed_reward": high_speed_reward,
            "off_road_reward": float(not self.vehicle.on_road),
            "steering_change_reward": steering_penalty,
        }

    def _reward(self, action):
        """
        Compute the total reward, including steering penalty.
        :param action: the last action performed
        :return: the total reward
        """

        rewards = self._rewards(action)
        
        # --- START: Added code for detailed reward printing ---
        # print("--- Reward Components ---")
        # total_reward = 0
        # for name, reward_val in rewards.items():
        #     weight = self.config.get(name, 0)
        #     weighted_reward = weight * reward_val
        #     total_reward += weighted_reward
            # print(f"{name:<25}: raw={reward_val:.3f}, weight={weight:.2f}, weighted={weighted_reward:.3f}")
        # print(f"Total Step Reward: {total_reward:.3f}")
        # print("-------------------------")
        # --- END: Added code ---

        reward = sum(
            self.config.get(name, 0) * reward for name, reward in rewards.items()
        )
        
        if self.config["normalize_reward"]:
            reward = utils.lmap(
                reward,
                [
                    sum(
                        self.config.get(name, 0)
                        for name in ["collision_reward", "off_road_reward", "steering_change_reward"]
                    ),
                    self.config["high_speed_reward"] + self.config["right_lane_reward"],
                ],
                [0, 1],
            )
        return reward
    
    def step(self, action):
        """Override step to reset ego vehicle state after a crash."""
        steering_angle = 0 # Default to 0 if not a continuous action with steering
        if isinstance(self.action_space, spaces.Box) and len(action) > 1:
             # Assuming steering is the second element for ContinuousAction
            steering_angle = float(action[1])

        self.steering_history.append(steering_angle)
        obs, reward, done, truncated, info = super().step(action)
        
        if self.vehicle.crashed:
            lane_width = self.config.get("lane_width", 4.0)
            y_position = self.vehicle.position[1]
            lanes_count = self.config["lanes_count"]
            lane_index = int(np.round(y_position / lane_width + lanes_count / 2))
            lane_index = np.clip(lane_index, 0, lanes_count - 1)
            self.vehicle.position = [
                self.vehicle.position[0],
                (lane_index - lanes_count / 2) * lane_width
            ]
            self.vehicle.heading = 0
            self.vehicle.crashed = False
            self.vehicle.speed = 25.0

            for vehicle in self.road.vehicles[:]:
                if vehicle is not self.vehicle and vehicle.crashed:
                    self.road.vehicles.remove(vehicle)
                    break
        
        return obs, reward, done, truncated, info
    
    def _is_terminated(self) -> bool:
        """The episode is over if the ego vehicle crashed."""
        return (
            False
            or self.vehicle.crashed
            or self.config["offroad_terminal"]
            and not self.vehicle.on_road
        )


if __name__ == "__main__":
    gym.register(id="custom-highway-v0", entry_point=__name__ + ":CustomHighwayEnv")

    config = {
        "lanes_count": 4,
        "action": {
            "type": "ContinuousAction",
        },
        "reward_speed_range": [20, 30], # Added for high_speed_reward calculation
        "steering_change_reward": -0.2,
        "steering_history_length": 5,
        "off_road_reward": -2.0,
        "right_lane_reward": 0.1,
        "collision_reward": -2.0,
        "high_speed_reward": 0.4,
        "normalize_reward": False,
        "offroad_terminal": True,
        "policy_frequency": 15,
        "simulation_frequency": 15,
    }
    env = gym.make("custom-highway-v0",render_mode="rgb_array",config=config)
    env = RecordVideo(env, video_folder="videos",)
    env.reset()

    for step in range(50):
        action = [0.2, -0.1] if step % 2 else [0.2, 0.1]
        print(f"\n--- Step {step + 1}, Action: {action} ---")
        observation, reward, done, tm, info = env.step(action)
        
        if done:
            print(f"Finished after {step + 1} timesteps")
            break

    env.close()
