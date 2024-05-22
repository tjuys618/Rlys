import base64
import requests
import textwrap 
from langchain_gpt import gpt4v_langchain
# OpenAI API Key
# api_key = "sk-jpl8fK78V15kb2bvF1B5839755Cb4d139fD246Cf59E4Fa03"

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def action_description():
    ACTIONS_DESCRIPTION = {
        0: 'Turn-left - change lane to the left of the current lane',
        1: 'IDLE - remain in the current lane with current speed',
        2: 'Turn-right - change lane to the right of the current lane',
        3: 'Acceleration - accelerate the vehicle',
        4: 'Deceleration - decelerate the vehicle'
    }
    availableActions = [0,1,2,3,4]
    avaliableActionDescription = 'Your available actions are: \n'
    for action in availableActions:
        avaliableActionDescription += ACTIONS_DESCRIPTION[action] + ' Action_id: ' + str(action) + '\n'
    return avaliableActionDescription

def gpt4v(scenario_description:str = "",input: str="", other_llm: str = "",image_path = "highway_env_image.jpg"):

    avaliableActionDescription = action_description()
    # client = OpenAI(base_url='https://api.pumpkinaigc.online/v1',api_key='sk-jpl8fK78V15kb2bvF1B5839755Cb4d139fD246Cf59E4Fa03')
    delimiter = "####"
    driving_intensions = "Drive safely and avoid collisons"
    print("input:", input)
    print("other_llm:", other_llm)
    # image_path = "highway_env_image.png"
    base64_image = encode_image(image_path)
    prompt = f"""\You are ChatGPT, a large language model trained by OpenAI. Now you act as a mature driving assistant, capable of delivering accurate and correct advice to human drivers in complex urban driving scenarios. Currently, there are two conflicting opinions from Amy and Steve regarding which action should be chosen in the current situation. It's your responsibility to evaluate the scenario and determine the correct course of action. While you will consider the opinions of others, it's important to note that these opinions may contain errors. You will carefully analyze the situation and provide support for the judgment that aligns with the safest and most appropriate action, based on your assessment of the current conditions.
You'll receive a picture of the situition now and thorough description of the current driving scenario, including your past decisions and insights into the intentions of others in this situation. You will also be given the available actions you are allowed to take. All of these elements are delimited by {delimiter}.

Your response should use the following format:
<reasoning>
<reasoning>
<repeat until you have a decision>
Response to user:{delimiter} <output one 'Action_id' as a int number of you decision, without any action name or explanation. The output decision must be unique and not ambiguous, for example if you decide to decelearate, then output "4". > 

Make sure to include {delimiter} to separate every step."""
    prompt += f"""

Here is the current scenario:
{delimiter} Driving scenario description:
{scenario_description}
{delimiter} Driving Intensions:
Your aim is to go through the highway safely and avoid collisions.
{delimiter} Available actions:
{avaliableActionDescription}

Here is the opinions of Steve and Amy:
{delimiter} Steve and Amy's opinions:
{input}
{delimiter} jury's opinions:
{other_llm}
You can stop reasoning once you have a valid action to take and you have sufficient reasons to support one person's judgment over the other. """
    # completion0 = client.chat.completions.create(
    #     model="gpt-4-vision-preview",
    #     messages=[
    #         {
    #             "role": "user",
    #             "content": [
    #                 {
    #                     "type": "text",
    #                     "text": prompt+intput,
    #                 },
    #                 {
    #                     "type": "image_url",
    #                     "image_url": {
    #                         "url": f"data:image/jpeg;base64,{base64_image}"
    #                     },
    #                 },
    #             ],
    #         }
    #     ],
    #     max_tokens=2048,
    # )

    response = gpt4v_langchain(prompt, image_path)
    print("gpt4v:",response)
    return response


if __name__ == "__main__":
    from jsonargparse import CLI
    CLI(gpt4v)