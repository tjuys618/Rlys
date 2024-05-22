from llama_env import llama_env
from jury_rl import jury_rl
from jurygpt import jurygpt
import gymnasium as gym
import copy
import torch
import warnings

# env.close()
env = gym.make("highway-fast-v0", render_mode="rgb_array")
config = {
        "observation": {
            "type": "Kinematics"
        },
        "action": {
            "type": "DiscreteMetaAction",
        },
        "lanes_count": 4,
        "vehicles_count": 50,
        "controlled_vehicles": 1,
        "initial_lane_id": None,
        "duration": 20,  # [s]
        "ego_spacing": 2,
        "vehicles_density": 1,
        "collision_reward": -1,    # The reward received when colliding with a vehicle.
        "right_lane_reward": 0.1,  # The reward received when driving on the right-most lanes, linearly mapped to
                                   # zero for other lanes.
        "high_speed_reward": 0.4,  # The reward received when driving at full speed, linearly mapped to zero for
                                   # lower speeds according to config["reward_speed_range"].
        "lane_change_reward": 0,   # The reward received at each lane change action.
        "reward_speed_range": [20, 30],
        "normalize_reward": True,
        "offroad_terminal": False
}
# env.configure(config)
# obs = env.reset()[0]
torch.set_float32_matmul_precision("high")
warnings.filterwarnings(
    # Triggered internally at ../aten/src/ATen/EmptyTensor.cpp:31
    "ignore", 
    message="ComplexHalf support is experimental and many operators don't support it yet"
)
# 克隆原始环境并将其重置为相同的初始状态
def clone_environment(env, initial_state):
    env_copy = copy.deepcopy(env)  # 深度复制
    # env_copy.reset()  # 重置环境
    # 如果深度复制后的环境没有完全重置，可能需要设定初始状态
    return env_copy



for iter_whole in range(20):
    env.configure(config)
    obs = env.reset()[0]
    cloned_env1 = clone_environment(env, obs)
    cloned_env2 = clone_environment(env, obs)
    cloned_env3 = clone_environment(env, obs)
    jury_rl(cloned_env3, iter_whole, obs)
    llama_env(cloned_env1, iter_whole)
    jurygpt(cloned_env2, iter_whole)
    