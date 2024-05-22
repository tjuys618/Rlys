# import openai
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
import random
import lightning as L
import torch
import re
import subprocess
import os
import time
import importlib
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from generate import generate
from lit_llama import Tokenizer, LLaMA
from lit_llama.lora import lora
from lit_llama.utils import lazy_load, llama_model_lookup, quantization
from scripts.prepare_alpaca import generate_prompt
# from art_gpt import art_gpt
# from math_gpt import math_gpt
# from psychology_gpt import psychology_gpt
# from environmentalist_gpt import environmentalist_gpt
from torch.utils.tensorboard import SummaryWriter
from gymnasium.wrappers import RecordVideo
from PIL import Image

test_list_seed = [5838, 2421, 7294, 9650, 4176, 6382, 8765, 1348,
                  4213, 2572, 5678, 8587, 512, 7523, 6321, 5214, 31]
lane_name = ["leftmost","left-center","center","right-center","rightmost","far-right"]

lora_r = 8
lora_alpha = 16
lora_dropout = 0.05
equal = 0
iter_whole = 0
# openai.api_key = "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"

# config = super().default_config()
config={
    "observation": {
        "type": "Kinematics",
        "absolute": True
    },
    "action": {
        "type": "DiscreteMetaAction",
    },
    "lanes_count": 6,
    "vehicles_count": 60,
    "controlled_vehicles": 1,
    "initial_lane_id": None,
    "duration": 40,  # [s]
    "ego_spacing": 2,
    "vehicles_density": 1,
    "collision_reward": -1.2,    # The reward received when colliding with a vehicle.
    "right_lane_reward": 0.1,  # The reward received when driving on the right-most lanes, linearly mapped to
                               # zero for other lanes.
    "high_speed_reward": 0.37,  # The reward received when driving at full speed, linearly mapped to zero for
                               # lower speeds according to config["reward_speed_range"].
    "lane_change_reward": 0.05,   # The reward received at each lane change action.
    "reward_speed_range": [20, 30],
    "normalize_reward": True,
    "offroad_terminal": False
}

env = gym.make("highway-fast-v0", render_mode="rgb_array")
env.configure(config)
seed = random.choice(test_list_seed)

done = truancted = False

def Steve_with_bias(
    count,
    env,
    fundimental,
    prompt: str = "",
    lora_path: Path = Path("out/50000-3double-easy/highway-easy-relative-50000-3double-finetuned.pth"),
    pretrained_path: Path = Path("checkpoints/lit-llama/13B/lit-llama.pth"),
    tokenizer_path: Path = Path("checkpoints/lit-llama/tokenizer.model"),
    quantize: Optional[str] = None,
    max_new_tokens: int = 100,
    top_k: int = 200,
    temperature: float = 0.8,
) -> None:
    
    # obs, info = env.reset()
    # env.configure({"simulation_frequency": 15})
    assert lora_path.is_file()
    assert pretrained_path.is_file()
    assert tokenizer_path.is_file()

    if quantize is not None:
        raise NotImplementedError("Quantization in LoRA is not supported yet")

    precision = "bf16-true" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "32-true"
    fabric = L.Fabric(accelerator="cuda", devices=1, precision=precision)
    fabric.launch()
    
    tokenizer = Tokenizer(tokenizer_path)
    # action = np.random.randint(0, env.action_space.n)
    
    # obs, info = env.reset()
    # print("i:",i)
    # print("Loading model ...", file=sys.stderr)
    t0 = time.time()

    with lazy_load(pretrained_path) as pretrained_checkpoint, lazy_load(lora_path) as lora_checkpoint:
        name = llama_model_lookup(pretrained_checkpoint)

        with fabric.init_module(empty_init=True), lora(r=lora_r, alpha=lora_alpha, dropout=lora_dropout, enabled=True):
            model = LLaMA.from_name(name)

        # 1. Load the pretrained weights
        model.load_state_dict(pretrained_checkpoint, strict=False)
        # 2. Load the fine-tuned lora weights
        model.load_state_dict(lora_checkpoint, strict=False)

    # print(f"Time to load model: {time.time() - t0:.02f} seconds.", file=sys.stderr)


    model.eval()
    model = fabric.setup(model)
    model_device = model.device
    print("Model is on device:", model_device)
    if(count == 0):
        # obs, reward, done, info, _ = env.reset(seed=seed)
        done = truancted = False
    # else:
    #     obs, reward, done, info, _ = env.step(fundimental["action"])
            
    # num = -1
    # for j in range(4):#第j个车辆
    #     if(near_by[j][0] != 0):
    #         num += 1
        # print(num)
    # for k in range(len(intervals)+1):
    #     if(sorted_q_values[0][1] < intervals[k]):
    #         lane = labels[k]
    #         break
    # if(i != 0):
    #     if(done[i-1]):
    #         prompt += "You have finished the task. Now is a new turn."
    #     if(truancted[i-1]):
    #         prompt += "You have failed the task. Now is a new turn."
    prompt = "You need to try your best to understand why the data I provide leads to such choices, and you need to get as close as possible to the dataset I provide. The dataset is obtained by the DQN policy. You are the brain of an autonomous driver. Your driving intension is to driving safely and avoid collisions. You are going to go through highway. X-axis points forward and Y-axis points to right. I'll give you your detail situation. Your available actions are: 0 for LANE_LEFT which means turning to the left lane, 1 for IDLE, 2 for LANE_RIGHT which means turning to the right lane, 3 for FASTER, and 4 for SLOWER. These actions allow adjustments to the ego vehicle's velocity and lane position within the environment. Remember your aim is to go through the highway safely and avoid collisions. Now you need to chose your next step. "
    if(not fundimental["is_start"]):
        fundimental["input"] += "Your last action is {}.".format(fundimental["action"])
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
        if(j != 0):
            fundimental["input"] += "The vehicle {} is in lane {}, ".format(j, lane_name[int(float(near_by[j-1][0]))-1])
            if((near_by[j-1][3] - my_info[2]) != 0):
                fundimental["input"] += "and {:.4f} on your {}, ".format((near_by[j-1][3] - my_info[2])*vehicle[1], "right" if vehicle[1] > 0 else "left")
            else:
                fundimental["input"] += "which means in the same lane with you, "
            fundimental["input"] += "and {:.4f} {} you, ".format((near_by[j-1][2] - my_info[1])*vehicle[0],"in front of" if vehicle[0] > 0 else "behind")
            fundimental["input"] += "with a {} speed of {:.4f} m/s, ".format("forward" if vehicle[2]>0 else "backward",near_by[j-1][5]*vehicle[2])
            if(vehicle[3]!=0):
                fundimental["input"] += "trying to turning {} with a transverse speed of {:.4f} m/s. ".format("right" if vehicle[3]>0 else "left",near_by[j-1][6]*vehicle[3])
        
        if(j == 0):
            fundimental["input"] += "{} currently driving with a {} speed of {:.4f} m/s, ".format("You are" if j==0 else "The vehicle "+str(j)+" is", "forward" if vehicle[2]>0 else "backward",my_info[4]*vehicle[2] if j==0 else near_by[j-1][5]*vehicle[2])
            fundimental["input"] += "{} absolute loacation is ({:.4f},{:.4f}), ".format("your" if j==0 else "vehicle "+str(j)+"'s", my_info[1] if j ==0 else near_by[j-1][2],my_info[2] if j ==0 else near_by[j-1][3])
            fundimental["input"] += "and {} in lane {}. ".format("you are", lane_name[int(my_info[0])-1])
            fundimental["input"] += "You observe 4 other vehicles. "
        if(vehicle[3]!=0):
            fundimental["input"] += "trying to turning {} with a transverse speed of {:.4f} m/s, ".format("right" if vehicle[3]>0 else "left",my_info[5]*vehicle[3] if j==0 else near_by[j-1][6]*vehicle[3])
    # if(done or truancted):
    # # if(done[i]):
    #     is_start = True
    # else:
    fundimental["is_start"] = False
    sample = {"instruction": prompt, "input": fundimental["input"]}
    print("sample:", sample)
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
    match = re.search(r'\d+', output)
    with open("jury_output.txt", "a", encoding="utf-8") as f:
        print(fundimental["input"], file = f)
        if(match):
            action = int(match.group())
            print("action:",action,file=f)
            fundimental["action"] = action
        else:
            print("valid", file=f)
        # print("output:",output)
        # print("action:",action)
       


def Steve_reason(
    fundimental,
    lora_path: Path = Path("out/50000-3double-easy/highway-easy-relative-50000-3double-finetuned.pth"),
    pretrained_path: Path = Path("checkpoints/lit-llama/13B/lit-llama.pth"),
    tokenizer_path: Path = Path("checkpoints/lit-llama/tokenizer.model"),
    quantize: Optional[str] = None,
    max_new_tokens: int = 100,
    top_k: int = 200,
    temperature: float = 0.8,
) -> None:
    assert lora_path.is_file()
    assert pretrained_path.is_file()
    assert tokenizer_path.is_file()

    if quantize is not None:
        raise NotImplementedError("Quantization in LoRA is not supported yet")

    precision = "bf16-true" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "32-true"
    fabric = L.Fabric(devices=1, precision=precision)

    
    tokenizer = Tokenizer(tokenizer_path)
    # print("Loading model ...", file=sys.stderr)
    t0 = time.time()

    with lazy_load(pretrained_path) as pretrained_checkpoint, lazy_load(lora_path) as lora_checkpoint:
        name = llama_model_lookup(pretrained_checkpoint)

        with fabric.init_module(empty_init=True), lora(r=lora_r, alpha=lora_alpha, dropout=lora_dropout, enabled=True):
            model = LLaMA.from_name(name)

        # 1. Load the pretrained weights
        model.load_state_dict(pretrained_checkpoint, strict=False)
        # 2. Load the fine-tuned lora weights
        model.load_state_dict(lora_checkpoint, strict=False)

    print(f"Time to load model: {time.time() - t0:.02f} seconds.", file=sys.stderr)


    model.eval()
    model = fabric.setup(model)
    action_name = "IDLE"
    input = ""
    match fundimental["action"]:
        case 0:
            action_name = "LANE_LEFT which means turning to the left lane"
        case 2:
            action_name = "LANE_RIGHT which means turning to the right lane"
        case 3:
            action_name = "FASTER"
        case 4:
            action_name = "SLOWER"
    #Amy: You are currently driving with a forwad speed of 0.3125 m/s, your absolute loacation is (0.7545,0.0000),and you are in lane left.You observe 4 other vehicles.The vehicle 1 is in lane right, and 0.4739 meters on your right. and 0.6667 meters in front of you. with a backward speed of 0.0275 m/s.The vehicle 2 is in lane middle, and 0.3553 meters on your right. and 0.3333 meters in front of you. with a backward speed of 0.0145 m/s.The vehicle 3 is in lane middle, and 0.2298 meters on your right. and 0.3333 meters in front of you. with a backward speed of 0.0272 m/s.The vehicle 4 is in lane left, and 0.1090 meters on your right. with a backward speed of 0.0367 m/s. Why you choose action FASTER? \nSteve: I choose action FASTER because if I don’t choose this action, I’ll be stuck behind the slower-moving vehicles, which could lead to unnecessary delays and potentially create a bottleneck in the traffic. By increasing my speed, I can safely overtake them and maintain the flow of traffic. It’s important to note that I will only accelerate if it’s safe to do so, considering the distances between vehicles and the overall traffic conditions. \n Please complete the dialogue for me and REMEMBER ADJUST YOUR RESPONSE BASED ON ACTUAL ACTIONS.Amy: {}Your available actions are: 0 for LANE_LEFT which means turning to the left lane, 1 for IDLE, 2 for LANE_RIGHT which means turning to the right lane, 3 for FASTER, and 4 for SLOWER. These actions allow adjustments to the ego vehicle's velocity and lane position within the environment. Remember your aim is to go through the highway safely and avoid collisions. Now you need to chose your next step.\nSteve:I choose action {}.\n
    if(len(fundimental["A_reason"])==0):
        prompt = "<s>[INST] <<SYS>>You are the brain of an autonomous driver. Your driving intension is to driving safely and avoid collisions. You are driving on a highway. Amy is your student, and you want to teach her to know how to drive. You will answer Amy's questions in detail and rigorously. Answer Amy's questions with MORE THAN 100 WORDS.<</SYS>>"
        input = "Amy:{}\nSteve:I choose action {}. \nAmy:Can you EXPLAIN THE REASON why you choose action {}? Considering the distances between vehicles and the overall traffic conditions.  \nSteve: I think[/INST]".format(fundimental["input"], action_name, action_name)
    else:          
        prompt = "<s>[INST] <<SYS>>You are the brain of an autonomous driver. Your driving intension is to driving safely and avoid collisions. You are driving on a highway. Amy is your student, and you want to teach her to know how to drive. Amy has her own judgment about this situation. You need to think carefully about her judgment and use your own analysis to argue with her.<</SYS>>"
        for A_reason_i in range(len(fundimental["A_reason"])):
            input += "Steve:{}".format(fundimental["reason"][A_reason_i])
            input += "Amy:{}".format(fundimental["A_reason"][A_reason_i])
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
    fundimental["reason"].append(output.split("### Response:")[-1].strip())
    print("len_reason:",len(fundimental["reason"]))
    print("Steve_reason:",output.split("### Response:")[-1].strip())
    with open("jury_output.txt", "a", encoding="utf-8") as f:
        print("Steve_reason:",output.split("### Response:")[-1].strip(),file=f)


def Amy_judge(
    env,
    fundimental,
    max_new_tokens: int = 100,#100
    top_k: int = 200,
    temperature: float = 0.8,
    checkpoint_path: Path = Path("checkpoints/lit-llama/13B/lit-llama.pth"),
    tokenizer_path: Path = Path("checkpoints/lit-llama/tokenizer.model"),
    quantize: Optional[str] = None,
) -> None:

    if(len(fundimental["A_reason"])==0):
        instruction = "instruction: <s>[INST]<<SYS>>You are a student who need to study how to drive in the highway, now Steve is going to teach you, but he may have some errors, you can choose not accept his teaching and feed back with word “wrong” when you think he's wrong.<</SYS>>\n input:{}Steve:{} Take your time and think about it.\n Amy:I think you're".format(fundimental["input"],fundimental["reason"][0])
    else:
        instruction = "instruction: <s>[INST]<<SYS>>You are a student who need to study how to drive in the highway, now Steve is going to teach you, but he may have some errors, you can choose not accept his teaching and feed back with word “wrong” when you think he's wrong.<</SYS>>\n input:{}".format(fundimental["input"])
        for A_i in range(len(fundimental["A_reason"])):
            print("len_s:", len(fundimental["reason"]))
            print("len_A:", len(fundimental["A_reason"]))
            print("i:", A_i)
            instruction += "Steve:{}".format(fundimental["reason"][A_i])
            # instruction += delimiter
            instruction += "Amy:{}".format(fundimental["A_reason"][A_i])
            # instruction += delimiter
        instruction += "Steve:{}".format(fundimental["reason"][len(fundimental["A_reason"])])
        if len(fundimental["A_reason"]) == 1:
            
            instruction += f"""
        Response to user:<output one judgment as a result of you decision, agree with Steve or not, without any action name or explanation. The output decision must be unique and not ambiguous, for example if you decide to agree with Steve, then output "Steve", else output "Amy".>""" 
    # input = "Steve:{}{} Take your time and think about it.\n Amy:I think you're".format(fundimental["input"],fundimental["reason"])
    # sample = {"instruction": instruction, "input": input}

    assert checkpoint_path.is_file(), checkpoint_path
    assert tokenizer_path.is_file(), tokenizer_path

    precision = "bf16-true" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "32-true"
    fabric = L.Fabric(devices=1, precision=precision)

    print("Loading model ...", file=sys.stderr)
    t0 = time.time()
    with lazy_load(checkpoint_path) as checkpoint:
        name = llama_model_lookup(checkpoint)

        with fabric.init_module(empty_init=True), quantization(mode=quantize):
            model = LLaMA.from_name(name)

        model.load_state_dict(checkpoint)
    # print(f"Time to load model: {time.time() - t0:.02f} seconds.", file=sys.stderr)

    model.eval()
    model = fabric.setup(model)

    tokenizer = Tokenizer(tokenizer_path)
    encoded = tokenizer.encode(instruction, bos=True, eos=False, device=fabric.device)
    prompt_length = encoded.size(0)

    L.seed_everything(1234)
    t0 = time.perf_counter()
    y = generate(model, encoded, max_new_tokens, temperature=temperature, top_k=top_k)
    t = time.perf_counter() - t0

    model.reset_cache()
    output = tokenizer.decode(y)
    fundimental["A_reason"].append(output.split("Amy:")[1].strip())
    tokens_generated = y.size(0) - prompt_length
    # print(f"Time for inference {i + 1}: {t:.02f} sec total, {tokens_generated / t:.02f} tokens/sec", file=sys.stderr)
    if fabric.device.type == "cuda":
        pass
        # print(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB", file=sys.stderr)
    print("A_reason:",output.split("Amy:")[-1].strip())
    if re.search(r'\bwrong\b',output.split("Amy:")[-1].strip()):
        fundimental["judge"] = 0
    with open("jury_output.txt", "a", encoding="utf-8") as f:
        print("A_reason:",output.split("Amy:")[-1].strip(),file=f)
    if len(fundimental["A_reason"]) == 2:
        if re.search(r'\bSteve\b', output.split("Amy")[-1].strip()):
            fundimental["jury"] = 1
        else:
            fundimental["jury"] = 0


def Jury_trail(
    env,
    fundimental,
 ) -> None:
    # activate_command = f"pip install -r requirements.txt"
    # subprocess.run(activate_command, shell=True)
    # subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    # subprocess.call([sys.executable] + sys.argv)
    # openai = importlib.import_module("openai")
    # langchain = importlib.import_module("langchain")
    # langsmith = importlib.import_module("langsmith")
    other_llm = ""
    # functions = {
    #     "art_gpt": art_gpt,
    #     "math_gpt": math_gpt,
    #     # "gpt4": gpt4,
    #     "psychology_gpt": psychology_gpt,
    #     "environmentalist_gpt": environmentalist_gpt,
    # }
    input = ""
    for i_reason in range(len(fundimental["reason"])):
        input += "Steve:{}\nAmy:{}".format(fundimental["reason"][i_reason], fundimental["A_reason"][i_reason])
    # ACTIONS_DESCRIPTION = {
    #     0: 'Turn-left - change lane to the left of the current lane',
    #     1: 'IDLE - remain in the current lane with current speed',
    #     2: 'Turn-right - change lane to the right of the current lane',
    #     3: 'Acceleration - accelerate the vehicle',
    #     4: 'Deceleration - decelerate the vehicle'
    # }
    # availableActions = env.get_available_actions()
    # avaliableActionDescription = 'Your available actions are: \n'
    # for action in availableActions:
    #     avaliableActionDescription += ACTIONS_DESCRIPTION[action] + ' Action_id: ' + str(action) + '\n'
    # for function_name,function in functions.items():
    #     # activate_command = f"conda activate gpt4 && pip install -r requirements.txt && python {function_name}.py --prompt={fundimental["input"]+input}"
    #     other_llm += function(avaliableActionDescription,fundimental["input"]+input)
    #     time.sleep(10)
    gpt_input = fundimental["input"]+input
    # gpt_input = "a"
    print("gpt_input:", gpt_input)
    # conda_path = "/workspace/miniconda3"
    activate_command = f"""pip install -r requirements-gpt4.txt && python other_llm.py --input "{gpt_input}" """
    try:
        result = subprocess.run(
            activate_command,
            shell=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
    except subprocess.CalledProcessError as e:
        # 打印详细错误信息
        print("Command failed with exit status:", e.returncode)
        print("Command output:", e.output)
        print("Command error output:", e.stderr.decode("utf-8"))
    # other_llm = result.stdout.decode("utf-8")
    other_llm = str(result).split("other_llm")[2].strip()
    print("other_llm:",other_llm)
    #gpt4v
    # activate_command = f"conda init"
    # subprocess.run(activate_command, shell=True)
    # activate_command = f"conda activate gpt4"
    # subprocess.run(activate_command, shell=True)
    
    from gpt4v import gpt4v
    image_array = env.render()

    # 将图像数组转换为PIL图像对象
    image = Image.fromarray(image_array)
    # 保存图像到文件（PNG格式）
    image_file_path = "highway_env_image.jpg"
    image.save(image_file_path)
    scenario_description = fundimental["input"]
    # scenario_description = "scenario_description"
    # input="input"
    activate_command = f"""pip install -r requirements-gpt4.txt && python gpt4v.py --scenario_description="{scenario_description}" --input="{input}" --other_llm="{other_llm}" """
    try:
        result = subprocess.run(
            activate_command,
            shell=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
    except subprocess.CalledProcessError as e:
        # 打印详细错误信息
        print("Command failed with exit status:", e.returncode)
        print("Command output:", e.output)
        print("Command error output:", e.stderr.decode("utf-8"))
    # other_llm = result.stdout.decode("utf-8")
    dj_llm = str(result).split("gpt4v")[2].strip()
    # dj_llm = result.stdout.decode("utf-8")
    print("dj_llm:",dj_llm)
    # response = gpt4v(fundimental["input"],avaliableActionDescription,input,image_file_path)
    # response = gpt_response["choices"][0]["message"]["content"]
    # if "Amy" in response:
    #     decision_person[1] = 1
    # elif "Steve" in response:
    #     decision_person[1] = 0
    # else:
    #     decision_person[1] = -1
    # print("gpt_first:",response)
    # dj_llm = "gpt4v:"+response+"\n"
    # dj_llm = ""

    #dilu
    # activate_command = f"conda init"
    # subprocess.run(activate_command, shell=True)
    # activate_command = f"conda activate dilu && python dilu_output.py"
    # dilu_action= subprocess.run(activate_command, shell=True, capture_output = True, text = True)
    # activate_command = f"pip install -r DiLu/requirements.txt"
    # subprocess.run(activate_command, shell=True)
    # subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "DiLu/requirements.txt"])
    # subprocess.call([sys.executable] + sys.argv)
    # openai = importlib.import_module("openai")
    # langchain = importlib.import_module("langchain")
    # chromadb = importlib.import_module("chromadb")
    # langsmith = importlib.import_module("langsmith")
    # print(langchain.__version__)
    from DiLu.run_dilu import whole
    os.environ["OPENAI_API_BASE"] = "https://api.pumpkinaigc.online/v1"
    dilu_action, dilu_response = whole(env, other_llm, fundimental["reason"]+fundimental["A_reason"], dj_llm)
    with open("jury_output.txt", "a", encoding="utf-8") as f:
        print("other_llm:",other_llm,"\ndj_llm:",dj_llm,"\ndilu_response:",dilu_response, file = f)
    return int(dilu_action)
    

def jurygpt(env,iter_whole):
    writer = SummaryWriter(log_dir="whole")
    video_dir = f"videos/{iter_whole}"
    os.makedirs(video_dir, exist_ok=True)
    total_reward = 0
    env = RecordVideo(env, video_folder=video_dir, episode_trigger=lambda episode_id: True, name_prefix="jurygpt")
    fundimental = {"input": "", "action": int, "reason": [], "A_reason":[], "judge": int, "jury":int, "is_start":True }
    done = 0
    count = 0
    while(not done):
        with open("jury_output.txt", "a", encoding="utf-8") as f:
            print("\n\n\n\n\n",file=f)
            print("action_time:",count,file=f)
            # print("\n",file=f) 
        Steve_with_bias(count, env, fundimental)
        Steve_reason(fundimental)
        Amy_judge(env, fundimental)
        if(fundimental["judge"] == 0):
            Steve_reason(fundimental)
            Amy_judge(env, fundimental)
            # Steve_reason(fundimental)
            # Amy_judge(env, fundimental)
            if(fundimental["jury"] == 0):
                fundimental["action"] = Jury_trail(env, fundimental)
        fundimental["judge"] = 1
        fundimental["jury"] = 1
        fundimental["reason"] = []
        fundimental["A_reason"] = []
        fundimental["input"] = ""
        obs, reward, done, truncated, info = env.step(fundimental["action"])
        collision = env.vehicle.crashed
        total_reward += reward
        with open("jury_output.txt", "a", encoding="utf-8") as f:
            print("action:", fundimental["action"], file = f)
            if (done):
                print("done", file=f)
            if (env.vehicle.crashed):
                print("truncated", file=f)
        if(collision or done):
            writer.add_scalar('step_count/jurygpt', count+1, iter_whole)
            writer.add_scalar('reward/jurygpt', total_reward, iter_whole)
            env.close()
            return count +1
        count += 1
    env.close()
    return count+1
    
if __name__ == "__main__":
    from jsonargparse import CLI
    torch.set_float32_matmul_precision("high")
    warnings.filterwarnings(
        # Triggered internally at ../aten/src/ATen/EmptyTensor.cpp:31
        "ignore", 
        message="ComplexHalf support is experimental and many operators don't support it yet"
    )
    CLI(jurygpt(env,iter_whole))    