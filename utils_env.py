from PIL import Image
import numpy as np
import gymnasium as gym
import highway_env
from gymnasium.spaces import Box
from gymnasium.wrappers import RecordVideo
from gymnasium import Wrapper
import numpy as np
from custom_reward_env import CustomHighwayEnv

# Define the ImageOnlyWrapper (same as provided)
class ImageOnlyWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        # print(self.env.observation_space)
        self.observation_space = self.env.observation_space[0]
        self.kinematics = None
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.kinematics = obs[1]
        return obs[0], info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.kinematics = obs[1]
        return obs[0], reward, terminated, truncated, info
    
    def get_kinematics(self):
        if self.kinematics is None:
            return None
        for i in range(len(self.kinematics)):
            if self.kinematics[i, 0] == 0:
                return self.kinematics[:i]
        return self.kinematics
    
class RGBObservationWrapper(gym.ObservationWrapper):
    """Wrapper to replace kinematic observations with RGB images from env.render()."""
    
    def __init__(self, env):
        super().__init__(env)
        # Define observation space for RGB images: shape (height, width, channels)
        self.observation_space = Box(
            low=0,
            high=255,
            shape=(180, 600, 3),
            dtype=np.uint8
        )
    
    def observation(self, observation):
        """Return the RGB image from env.render() as the observation."""
        # Render the environment to get the RGB image
        rgb_obs = self.env.render()
        # Ensure the output is a numpy array with shape (180, 600, 3)
        if not isinstance(rgb_obs, np.ndarray):
            rgb_obs = np.array(rgb_obs)
        # Optionally normalize to [0, 1] for CNN if needed (commented out by default)
        # rgb_obs = rgb_obs / 255.0
        return rgb_obs



# Environment configuration (same as provided)
config = {
    "observation": {
        "type": "TupleObservation",
        "observation_configs": [
            {
                "type": "GrayscaleObservation",
                "observation_shape": (224, 64),
                "stack_size": 1,
                "weights": [0.2989, 0.5870, 0.1140],
                "scaling": 2,
            },
            {
                "type": "Kinematics",
                "vehicles_count": 5,
                "features": ["presence", "x", "y", "vx", "vy"],
                "absolute": False,
                "normalize": False,
            }
        ]
    },
    "action": {
        "type": "ContinuousAction",  # Assuming DiscreteAction; change to ContinuousAction if needed
    },
    "policy_frequency": 15,
    "simulation_frequency": 15,
    "steering_change_reward": 0,  # Penalty for frequent steering changes
    "steering_history_length": 5,  # Number of timesteps to track steering actions
    "off_road_reward": -2.5,
    "right_lane_reward": 0,
    "collision_reward":-2.5,
    "high_speed_reward": 0.4,
    "normalize_reward": False,
    "offroad_terminal": True,
    "offscreen_rendering": True,
    "screen_width": 600,
    "screen_height": 180,
}

# config_state = {
#     "observation": {
#         "type": "Kinematics",
#         "vehicles_count": 10,
#         "features": ["presence", "x", "y", "vx", "vy","heading", "lat_off"], #, "lat_off", "ang_off"
#         "absolute": False,
#         "normalize": True,
#         },
#     "action": {
#         "type": "ContinuousAction",  
#         "longitudinal": False,
#         "speed_range": [27,27.1],   #target_speeds is for meta action
#         "lateral": True,
#         "steering_range": [-np.pi/8, np.pi/8],
#     },
#     'vehicles_density': 1,
#     'duration': 60,
#     "policy_frequency": 10,
#     "simulation_frequency": 10,
#     "steering_change_reward": 0,  
#     "steering_history_length": 5,  
#     "off_road_reward": -10,
#     "right_lane_reward": 0,
#     "collision_reward":-10,
#     "high_speed_reward": 0.6,
#     "normalize_reward": False,
#     "offroad_terminal": True,
#     "offscreen_rendering": True,
#     "screen_width": 600,
#     "screen_height": 180,
# }

config_state2 = {
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 10,
        "features": ["presence", "x", "y", "vx", "vy","heading", "lat_off"], #, "lat_off", "ang_off"
        "absolute": False,
        "normalize": True,
        },
    "action": {
        "type": "ContinuousAction",  
        "longitudinal": True,
        "accel_range": [-2, 2],  # Acceleration range for longitudinal control
        "speed_range": [24, 30],  # target_speeds is for meta action
        "lateral": True,
        "steering_range": [-np.pi/8, np.pi/8],
    },
    'vehicles_density': 1,
    'duration': 60,
    "policy_frequency": 10,
    "simulation_frequency": 10,
    "steering_change_reward": 0,  
    "steering_history_length": 5,  
    "off_road_reward": -10,
    "right_lane_reward": 0,
    "collision_reward":-10,
    "high_speed_reward": 0.6,
    "normalize_reward": False,
    "offroad_terminal": True,
    "offscreen_rendering": True,
    "screen_width": 600,
    "screen_height": 180,
}


config_ogrid = {
    "observation": {
        "type": "OccupancyGrid",
        "features": ["presence", "on_road"],
        "grid_size": [[-18, 18], [-18, 18]],
        "grid_step": [3, 3],
        "as_image": False,
        "align_to_vehicle_axes": True,
    },
    "action": {
        "type": "ContinuousAction",  
        "longitudinal": False,
        "lateral": True,
        "steering_range": [-np.pi/8, np.pi/8],
        "speed_range": [27,27.1],  
    },
    "policy_frequency": 15,
    "simulation_frequency": 15,
    "steering_change_reward": 0,  
    "steering_history_length": 5,  
    "off_road_reward": -10,
    "right_lane_reward": 0,
    "collision_reward":-5,
    "high_speed_reward": 0.8,
    "normalize_reward": False,
    "offroad_terminal": True,
    "offscreen_rendering": True,
    "screen_width": 600,
    "screen_height": 180,
}



def create_imageenv(config):
    """Create and wrap the environment."""
    gym.register(id="custom-highway-v0", entry_point=__name__ + ":CustomHighwayEnv")
    env = gym.make("custom-highway-v0",render_mode="rgb_array",config=config)
    env = ImageOnlyWrapper(env)
    return env

def create_stateenv(config):
    """Create and wrap the environment."""
    gym.register(id="custom-highway-v0", entry_point=__name__ + ":CustomHighwayEnv")
    env = gym.make("custom-highway-v0",render_mode="rgb_array",config=config)
    return env

def create_rgbenv(config):
    gym.register(id="custom-highway-v0", entry_point=__name__ + ":CustomHighwayEnv")
    env = gym.make("custom-highway-v0", render_mode="rgb_array", config=config)
    env = RGBObservationWrapper(env)
    return env
if __name__ == "__main__":
    # Initialize environment
    # env = gym.make('highway-v0', config=config, render_mode="rgb_array")
    # env = RecordVideo(env, "videos", episode_trigger=lambda x: True, name_prefix="highway-tuple")

    # env = create_rgbenv(config)  # Create the environment
    env = create_stateenv(config_state2)  # Create the environment
    # env = create_stateenv(config_ogrid)  # Create the environment


    print(env.observation_space)
    print(env.action_space)

    # Example usage
    env = RecordVideo(env, "videos", episode_trigger=lambda x: True, name_prefix="highway-state")
    obs, info = env.reset()
    # print("kinematics:", env.get_kinematics())
    sreen = env.render()
    Image.fromarray(sreen).save("screen.png")   
    print("screen shape: ",sreen.shape,type(sreen)) 
    # breakpoint()
    
    done = False
    step = 0
    while not done:
        # for vehicle_data in obs[1:]:
        #     presence, x, y, vx, vy, _, _ = vehicle_data
        #     if presence:  # Check if the vehicle is present
        #         speed = np.sqrt(vx**2 + vy**2)
        #         print(f"A nearby vehicle's speed is: {speed:.2f} m/s")

        action = env.action_space.sample()  # throttle, steering
        # action[0] = 0
        action[1] = 0 #if step%2 else -1
        # action[0] = 1
        # action[1] = 0
        obs, reward, terminated, truncated, info = env.step(action)
        # print(type(obs), obs.shape)
        # print("current speed: ",env.unwrapped.vehicle.speed)
        print("current action: ", action)
        print("step reward: ", reward)
        if info["crashed"]:
            print(" reward after crash happens: ", reward)
            print("currrent step: ", step)
            breakpoint()    
        # print("kinematics:", env.env.get_kinematics())
        # print("obs: ", obs)
        print("ego car speed: ", env.unwrapped.vehicle.speed)
        # breakpoint()
        done = terminated or truncated
        step += 1
    print("done at step :", step)

    env.close()

