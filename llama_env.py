import sys
import time
import warnings
from pathlib import Path
import gymnasium as gym
import highway_env
import itertools
import numpy as np
from typing import Optional
import json
import lightning as L
import torch
import re
import random
import os
from gymnasium.wrappers import RecordVideo


# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from generate import generate
from lit_llama import Tokenizer, LLaMA
from lit_llama.lora import lora
from lit_llama.utils import lazy_load, llama_model_lookup
from scripts.prepare_alpaca import generate_prompt
from torch.utils.tensorboard import SummaryWriter
test_list_seed = [5838, 2421, 7294, 9650, 4176, 6382, 8765, 1348,
                  4213, 2572, 5678, 8587, 512, 7523, 6321, 5214, 31]
lane_name = ["leftmost","left-center","center","right-center","rightmost","far-right"]
# config={
#     "observation": {
#         "type": "Kinematics",
#         "absolute": True
#     },
#     "action": {
#         "type": "DiscreteMetaAction",
#     },
#     "lanes_count": 6,
#     "vehicles_count": 60,
#     "controlled_vehicles": 1,
#     "initial_lane_id": None,
#     "duration": 40,  # [s]
#     "ego_spacing": 2,
#     "vehicles_density": 1,
#     "collision_reward": -1.2,    # The reward received when colliding with a vehicle.
#     "right_lane_reward": 0.1,  # The reward received when driving on the right-most lanes, linearly mapped to
#                                # zero for other lanes.
#     "high_speed_reward": 0.37,  # The reward received when driving at full speed, linearly mapped to zero for
#                                # lower speeds according to config["reward_speed_range"].
#     "lane_change_reward": 0.05,   # The reward received at each lane change action.
#     "reward_speed_range": [20, 30],
#     "normalize_reward": True,
#     "offroad_terminal": False
# }

env = gym.make("highway-fast-v0", render_mode="rgb_array")
# env.configure(config)
seed = random.choice(test_list_seed)
obs, info = env.reset(seed=seed)
done = truancted = False

lora_r = 8
lora_alpha = 16
lora_dropout = 0.05
equal = 0
iter_whole = 0

def llama_env(
    env,
    iter_whole,
    prompt: str = "",
    input: str = "",
    lora_path: Path = Path("out/50000-3double-easy/highway-easy-relative-50000-3double-finetuned.pth"),
    pretrained_path: Path = Path("checkpoints/lit-llama/13B/lit-llama.pth"),
    tokenizer_path: Path = Path("checkpoints/lit-llama/tokenizer.model"),
    quantize: Optional[str] = None,
    max_new_tokens: int = 100,
    top_k: int = 200,
    temperature: float = 0.8,
) -> None:
    # torch.set_float32_matmul_precision("high")
    """Generates a response based on a given instruction and an optional input.
    This script will only work with checkpoints from the instruction-tuned LoRA model.
    See `finetune_lora.py`.

    Args:
        prompt: The prompt/instruction (Alpaca style).
        lora_path: Path to the checkpoint with trained LoRA weights, which are the output of
            `finetune_lora.py`.
        input: Optional input (Alpaca style).
        pretrained_path: The path to the checkpoint with pretrained LLaMA weights.
        tokenizer_path: The tokenizer path to load.
        quantize: Whether to quantize the model and using which method:
            ``"llm.int8"``: LLM.int8() mode,
            ``"gptq.int4"``: GPTQ 4-bit mode.
        max_new_tokens: The number of generation steps to take.
        top_k: The number of top most probable tokens to consider in the sampling process.
        temperature: A value controlling the randomness of the sampling process. Higher values result in more random
            samples.
    """
    # env = gym.make("highway-fast-v0", render_mode="rgb_array")
    # obs, info = env.reset()
    # env.configure({"simulation_frequency": 15})
    writer = SummaryWriter(log_dir="whole")
    video_dir = f"videos/{iter_whole}"
    os.makedirs(video_dir, exist_ok=True)
    env = RecordVideo(env, video_folder=video_dir, episode_trigger=lambda episode_id: True, name_prefix="llama_env")
    assert lora_path.is_file()
    assert pretrained_path.is_file()
    assert tokenizer_path.is_file()
    writer = SummaryWriter(log_dir="whole")
    if quantize is not None:
        raise NotImplementedError("Quantization in LoRA is not supported yet")
    # device_ids = [0, 1]  # 假设你有两张 GPU
    precision = "bf16-true" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "32-true"
    fabric = L.Fabric(accelerator="cuda", devices=1, precision=precision)
    fabric.launch()
    
    tokenizer = Tokenizer(tokenizer_path)
    # action = np.random.randint(0, env.action_space.n)
    # obs, info = env.reset()
    pass_time = 0
    mean_action = 0
    total_reward = 0
    done = 0
    action_time = 0
    is_start = True
    while(not done):
        
        
        # done = collision = False
        print("Loading model ...", file=sys.stderr)
        t0 = time.time()
        with lazy_load(pretrained_path) as pretrained_checkpoint, lazy_load(lora_path) as lora_checkpoint:
            name = llama_model_lookup(pretrained_checkpoint)

            with fabric.init_module(empty_init=True), lora(r=lora_r, alpha=lora_alpha, dropout=lora_dropout, enabled=True):
                model = LLaMA.from_name(name)
                # model = torch.nn.DataParallel(model, device_ids=device_ids)

            # 1. Load the pretrained weights
            model.load_state_dict(pretrained_checkpoint, strict=False)
            # 2. Load the fine-tuned lora weights
            model.load_state_dict(lora_checkpoint, strict=False)

            print(f"Time to load model: {time.time() - t0:.02f} seconds.", file=sys.stderr)

        
        model.eval()
        model = fabric.setup(model)
        model_device = model.device
        print("Model is on device:", model_device)
        prompt = ""
        input = ""
        prompt += "You need to try your best to understand why the data I provide leads to such choices, and you need to get as close as possible to the dataset I provide. The dataset is obtained by the DQN policy. You are the brain of an autonomous driver. Your driving intension is to driving safely and avoid collisions. You are driving on a highway. I'll give you your detail situation. Your available actions are: 0 for LANE_LEFT which means turning to the left lane, 1 for IDLE, 2 for LANE_RIGHT which means turning to the right lane, 3 for FASTER, and 4 for SLOWER. These actions allow adjustments to the ego vehicle's velocity and lane position within the environment. Remember your aim is to go through the highway safely and avoid collisions. Now you need to chose your next step. "
        if(not is_start):
            input += "Your last action is {}.".format(action)
        else:
            is_start= False
        num = -1
        # for j_car in range(5):#第j个车辆
        #     if(obs[j_car][0] != 0):
        #         num += 1
        my_info= [0,0,0,0,0,0]
        near_by = [[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0]]
        for j in range(5):
            vehicle = [0, 0, 0, 0]
            if(j == 0):
                my_info[0] = env.road.vehicles[j].lane_index[2] - 1
                my_info[1] = env.road.vehicles[j].position[0]
                my_info[2] = env.road.vehicles[j].position[1]
                my_info[3] = env.road.vehicles[j].speed
                my_info[4],my_info[5] = env.road.vehicles[j].velocity
                vehicle[0] = 1 if my_info[1] > 0 else -1 if my_info[1] < 0 else vehicle[0]
                vehicle[1] = 1 if my_info[2] > 0 else -1 if my_info[2] < 0 else vehicle[1]
                vehicle[2] = 1 if my_info[4] > 0 else -1 if my_info[4] < 0 else vehicle[2]
                vehicle[3] = 1 if my_info[5] > 0 else -1 if my_info[5] < 0 else vehicle[3]           
            elif(j > 0 and j < 5):
                near_by[j-1][0] = env.road.vehicles[j].lane_index[-1] - 1
                near_by[j-1][1] = ((env.road.vehicles[j].position[0]-env.road.vehicles[0].position[0])**2 + (env.road.vehicles[j].position[1]-env.road.vehicles[0].position[1])**2)**(1/2)
                near_by[j-1][2] = env.road.vehicles[j].position[0]
                near_by[j-1][3] = env.road.vehicles[j].position[1]
                near_by[j-1][4] = env.road.vehicles[j].speed
                near_by[j-1][5], near_by[j-1][6] = env.road.vehicles[j].velocity
                vehicle[0] = 1 if near_by[j-1][2] > my_info[1] else -1 if near_by[j-1][2] < my_info[1] else vehicle[0]
                vehicle[1] = 1 if near_by[j-1][3] > my_info[2] else -1 if near_by[j-1][3] < my_info[2] else vehicle[1]
                vehicle[2] = 1 if near_by[j-1][5] > 0 else -1 if my_info[4] < 0 else vehicle[2]
                vehicle[3] = 1 if near_by[j-1][6] > 0 else -1 if my_info[5] < 0 else vehicle[3]
            input += "{} currently driving with a {} speed of {:.4f} m/s, ".format("You are" if j==0 else "Vehicle "+str(j)+" is", "forward" if vehicle[2]>0 else "backward",my_info[4]*vehicle[2] if j==0 else near_by[j-1][5]*vehicle[2])
            if(vehicle[3]!=0):
                input += "trying to turning {} with a transverse speed of {:.4f} m/s, ".format("right" if vehicle[3]>0 else "left",my_info[5]*vehicle[3] if j==0 else near_by[j-1][6]*vehicle[3])
            input += "{} absolute loacation is ({:.4f},{:.4f}), ".format("your" if j==0 else "vehicle "+str(j)+"'s", my_info[1] if j ==0 else near_by[j-1][2],my_info[2] if j ==0 else near_by[j-1][3])
            if(j == 0):
                input += "and {} in lane {}. ".format("you are", lane_name[int(my_info[0])-1])
                input += "You observe {} other vehicles. ".format(4)
            elif(j < 5):
                # print("near_by:",near_by[i][j-1][0])
                input += "which means {:.4f} on your {} and {:.4f} {} you, ".format((near_by[j-1][3] - my_info[2])*vehicle[1], "right" if vehicle[1] > 0 else "left", (near_by[j-1][2] - my_info[1])*vehicle[0],"in front of" if vehicle[0] > 0 else "behind")
                input += "and {} in lane {}. ".format("it is", lane_name[int(float(near_by[j-1][0]))-1])
        sample = {"instruction": prompt, "input": input}
        prompt = generate_prompt(sample)
        encoded = tokenizer.encode(prompt, bos=True, eos=False, device=model.device)
        t0 = time.perf_counter()
        output = generate(
            model,
            idx=encoded,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            eos_id=tokenizer.eos_id
        )
        t = time.perf_counter() - t0
        output = tokenizer.decode(output)
        output = output.split("### Response:")[1].strip()
        action = re.search(r'\d+', output)
        action = int(action.group())
        if action in env.action_space:
            action_time += 1
                
            print("epoch:", iter_whole,"action_time:",action_time)
            print("action:", action)
            
            if(done):
                if(collision):
                    print("truancted")
                else:
                    print("done")
                    # print("pass_rate:", pass_time/(iter_whole+1))
                
            with open('llama_env.txt', 'a') as f:
                print("\n\n")
                print("epoch:", iter_whole,"action_time:",action_time, file = f)
                print("action:", action, file = f)
                # if(done):
                #     if(collision):
                #         print("truancted", file = f)
                #     else:
                #         pass_time += 1
                #         print("done", file = f)
                #     print("pass_rate:", pass_time/(i + 1), file = f)
        else:
            print("invalid action")
            with open('llama_env.txt', 'a') as f:
                print("invalid action", file = f)
                print("\n\n")
                done = True
        obs, reward, done, info, _ = env.step(action)
        total_reward += reward
        vehicle_states = env.road.vehicles
        collision = env.vehicle.crashed
        if(collision or done):
            with open('llama_env.txt', 'a') as f:
                if(collision):
                    print("collision", file = f)
                elif(done):
                    print("done", file = f)
                writer.add_scalar('step_count/llama_bias', action_time+1, iter_whole)
                writer.add_scalar('reward/llama_bias', total_reward, iter_whole)
            env.close()
            return action_time+1
        print("reward:", reward)
        with open('llama_env.txt', 'a') as f:
            print("llama reward:", reward, file = f)
            print("\n\n")
    env.close()
    return action_time+1
            

    print(f"\n\nTime for inference: {t:.02f} sec total, {max_new_tokens / t:.02f} tokens/sec", file=sys.stderr)
    if fabric.device.type == "cuda":
        print(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB", file=sys.stderr)

        
    



if __name__ == "__main__":
    from jsonargparse import CLI

    torch.set_float32_matmul_precision("high")
    warnings.filterwarnings(
        # Triggered internally at ../aten/src/ATen/EmptyTensor.cpp:31
        "ignore", 
        message="ComplexHalf support is experimental and many operators don't support it yet"
    )
    CLI(llama_env(env, iter_whole))
