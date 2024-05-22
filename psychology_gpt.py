# import openai
from langchain_gpt import gpt3
# openai.api_key = "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" 
def psychology_gpt(avaliableActionDescription,prompt):
#     completion = openai.ChatCompletion.create(
#       model="gpt-3.5-turbo", 
#       messages=[{"role": "system", "content": """You are a psychologist by profession. Please analyze the current scene based on the common psychological performance of drivers. Please give the psychological derivation process and corresponding answers.And there are two people have different ideas about which action should be choose in this situition, you should consider both their opinions and the current scenario."""+avaliableActionDescription+"""
# few shot:
# In this scenario, I perceive the highway as a canvas where each vehicle contributes to a dynamic composition. To maintain harmony and safety within this artistic space, I must ensure balance and avoid disrupting the flow. Accelerating (action "3") may upset this delicate balance and lead to chaos. Changing lanes could disrupt the composition, potentially causing collisions. Thus, maintaining my current speed and position aligns with the fluidity and aesthetic of the highway "artwork." Hence, the action chosen is "1" .
    
# ---------------------------------------------------------------------------------------
    
# scene：
# """+prompt+""" Now you need to chose your next step, please give me your reason  within 100 words and give me your action only in one word.
# You should output in the following format:
# <reasoning>
# <reasoning>
# <repeat untill you have solid reasons for your specific action choice.>"""}]
#       )
    input = """You are a psychologist by profession. Please analyze the current scene based on the common psychological performance of drivers. Please give the psychological derivation process and corresponding answers.And there are two people have different ideas about which action should be choose in this situition, you should consider both their opinions and the current scenario."""+avaliableActionDescription+"""
few shot:
 git clone -b dev-bian https://github.com/MaomaoSophia/RL-Reasoning
Given the current scenario, the driver's psychological state may involve heightened alertness and a sense of urgency due to the presence of multiple vehicles and the need to avoid collisions. The previous action of accelerating (action 3: FASTER) suggests a proactive response to perceived threats. To maintain a sense of control and alleviate anxiety, the driver may choose to continue accelerating, aiming to create more space between their vehicle and Vehicle 1. This decision reflects a desire to assert control over the situation and minimize potential danger, aligning with the goal of driving safely on the highway. Therefore, the next step is to maintain acceleration (action 3: FASTER).
    
---------------------------------------------------------------------------------------
    
scene：
"""+prompt+""" Now you need to chose your next step, please give me your reason  within 100 words and give me your action only in one word.
You should output in the following format:
<reasoning>
<reasoning>
<repeat untill you have solid reasons for your specific action choice.>"""
    response = gpt3(input)
    print("psychology_gpt:",response)
    return response
    # return completion.choices[0].message['content']

# messages.append({"role":"assisstant","content":gpt_response.choices[0].message})
# print(messages)
# print(completion.choices[0].message['content'])