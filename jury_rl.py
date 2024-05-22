import gymnasium as gym
import highway_env
from stable_baselines3 import PPO
import numpy as np
import os
# from loguru import logger
from stable_baselines3 import DQN
import itertools
from gymnasium.wrappers import RecordVideo
import os
import base64
from PIL import Image
from io import BytesIO
import dill
from torch.utils.tensorboard import SummaryWriter
from gymnasium.wrappers import RecordVideo
import torch
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

TRAIN = True

def jury_rl(env, iter_whole, obs):
    model_path = "model.zip"
    model = DQN.load(model_path,env,device='cuda')
    # model_device = model.policy.device
    # print("Model is on device:", model_device)
    video_dir = f"videos/{iter_whole}"
    os.makedirs(video_dir, exist_ok=True)
    env = RecordVideo(env, video_folder=video_dir, episode_trigger=lambda e: True, name_prefix="rl")
    total_reward = 0
    count = 0
    done = 0
    writer = SummaryWriter(log_dir="whole")
    with open("rl_output.txt", "a", encoding="utf-8") as f:
        print("\n\n\n", iter_whole, file = f)
    while not done:
        # obs = obs.to("cuda")
        action,_state = model.predict(obs, deterministic=True)
        print("Output device:", action.device if isinstance(action, torch.Tensor) else "CPU")
        # model_device = next(model.parameters()).device
        # print("Model is running on device:", model_device)
        obs, reward, done, truncated, info = env.step(action)
        collision = env.vehicle.crashed
        total_reward += reward
        with open("rl_output.txt", "a", encoding="utf-8") as f:
            print("count；", count, file = f)
            print("reward:", reward, file = f)
        if(collision):
            with open("rl_output.txt", "a", encoding="utf-8") as f:
                print("collision")
            writer.add_scalar('step_count/rl', count+1, iter_whole)
            writer.add_scalar('reward/rl', total_reward, iter_whole)
            env.close()
            return count +1
        elif(done):
            with open("rl_output.txt", "a", encoding="utf-8") as f:
                print("done!")
            writer.add_scalar('step_count/rl', count+1, iter_whole)
            writer.add_scalar('reward/rl', total_reward, iter_whole)
            env.close()
            return count +1
        count += 1
    env.close()
    return count+1

if __name__ == '__main__':
    # Create the environment
    env = gym.make("roundabout", render_mode="rgb_array")
    # env.config["vehicles_count"] = 20
    env.reset()
    obs, info = env.reset()

    # Create the model
    model = DQN('MlpPolicy', env,
                gamma=0.8,
                train_freq=1,
                gradient_steps=1,
                target_update_interval=50,
                device='cuda',
                verbose=1,
                tensorboard_log="highway_dqn/")
    
    # Train the model
    # if TRAIN:
    #     model.learn(total_timesteps=int(2e6))
    #     model.save("highway_dqn/model")
    #     del model

    # Run the trained model and record video
    # model = DQN.load("F:\\bzy\\master\\study\\LLM-reasonable\\highway\\employhighway_dqn_model", env=env)
    env = RecordVideo(env, video_folder="highway_dqn/videos", episode_trigger=lambda e: True)
    env.unwrapped.set_record_video_wrapper(env)
    env.configure({"simulation_frequency": 15})  # Higher FPS for rendering

    for videos in range(10):
        done = truncated = False
        obs, info = env.reset()
        while not (done or truncated):
            # Predict
            action, _states = model.predict(obs, deterministic=True)
            # Get reward
            obs, reward, done, truncated, info = env.step(action)
            image_array = env.render()

            # 将图像数组转换为PIL图像对象
            image = Image.fromarray(image_array)
            # 保存图像到文件（PNG格式）
            image_file_path = "roundabout_env_image.jpg"
            image.save(image_file_path)
            crash = env.get_available_actions()
            print("crash: ", crash)
            vehicle_speed = env.vehicle.speed
            print("vehicle_speed: ", vehicle_speed)
            vehicle_position = env.vehicle.position
            print("vehicle_position: ", vehicle_position)
            vehicel_lane = env.vehicle.lane_index
            print("vehicel_lane: ", vehicel_lane)
            with open('env.pkl', 'wb') as f:
                dill.dump(env, f)
            for i in range(5):
                other_vehicles_speed_x,other_vehicles_speed_y = env.road.vehicles[i].velocity
                print(i,"other_vehicles_speed: ", other_vehicles_speed_x, other_vehicles_speed_y)
                other_vehicles_position = env.road.vehicles[i].position
                print(i,"other_vehicles_position: ", other_vehicles_position)
                lane = env.road.vehicles[i].lane_index
                print(i,"other_vehicles_lane: ", lane)

            # 输出图像文件路径
            print("Image saved as:", image_file_path)

            # image.show()
            # # 将图像转换为base64编码的字符串
            # with BytesIO() as buffer:
            #     image.save(buffer, format="PNG")
            #     image_bytes = buffer.getvalue()
            #     base64_encoded_image = base64.b64encode(image_bytes).decode('utf-8')

            # # 打印base64编码的图像
            # print(base64_encoded_image)
            sorted_obs = sorted(obs[1:], key=lambda x: x[1] ** 2 + x[2] ** 2, reverse=True)
            print("sorted_obs:",sorted_obs)

            lane = env.vehicle.lane_index
            print("lane: ", lane[2])
            other_vehicles = env.road.vehicles[0].lane_index
            print("other vehicles: ", other_vehicles)
            # print("other lane: ", other_lane)
            # Render
            env.render()

    states, actions, rewards = [], [], []
    done = False
    # logger.info("Eval Start")
    step = 0
    obs, info = env.reset()


    while not done:

        action = 0
        next_obs, reward, done, _, info = env.step(action)

        # # 获取对应的连续动作
        # cont_space = env.action_space.space()  # 获取连续动作空间
        # axes = np.linspace(cont_space.low, cont_space.high, env.action_space.actions_per_axis).T
        # all_actions = list(itertools.product(*axes))
        # continuous_action = all_actions[action]

        # logger.info("\nAction {}\n| longitudinal: {:.3f} | lateral: {:.3f} |".format(step, action[0], action[1]))
        # logger.info("Reward: {}".format(reward))

        # 记录状态，动作和奖励
        states.append(obs.tolist())  # 将状态转换为列表
        actions.append(action.tolist())  # 将动作转换为列表
        rewards.append(reward)
        lane = env.vehicle.lane_index
        print("lane: ", lane)
        other_vehicles = env.road.vehicles
        other_lane = env.vehicle.target_lane_index
        print("other lane: ", other_lane)
        step += 1
        obs = next_obs

    np.save("state.npy", states)
    np.save("action.npy", actions)
    np.save("reward.npy", rewards)

    # 关闭环境
    env.close()