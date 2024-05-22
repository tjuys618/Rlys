# import openai
from langchain_gpt import gpt3
# openai.api_key = "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" 
def environmentalist_gpt(avaliableActionDescription,prompt):
#     completion = openai.ChatCompletion.create(
#       model="gpt-3.5-turbo", 
#       messages=[{"role": "system", "content": """You are an environmentalist. Please analyze the current scenario and provide the energy loss derivation process and corresponding answers based on the need for environmental protection. And there are two people have different ideas about which action should be choose in this situition, you should consider both their opinions and the current scenario."""+avaliableActionDescription+"""
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
    input = """You are an environmentalist. Please analyze the current scenario and provide the energy loss derivation process and corresponding answers based on the need for environmental protection. And there are two people have different ideas about which action should be choose in this situition, you should consider both their opinions and the current scenario."""+avaliableActionDescription+"""
few shot:
Given the environmental concern, it's crucial to minimize energy consumption and emissions. Accelerating (action "3") would increase fuel consumption unnecessarily, especially when there's no urgent need to do so. Changing lanes might require additional acceleration and deceleration, leading to more energy usage. Hence, the most environmentally friendly option is to maintain the current speed and position. Therefore, the action chosen is "1" (IDLE).
    
---------------------------------------------------------------------------------------
    
scene：
"""+prompt+""" Now you need to chose your next step, please give me your reason  within 100 words and give me your action only in one word.
You should output in the following format:
<reasoning>
<reasoning>
<repeat untill you have solid reasons for your specific action choice.>"""
    response = gpt3(input)
    print("enviconmentalist_gpt:",response)
    return response

# messages.append({"role":"assisstant","content":gpt_response.choices[0].message})
# print(messages)
# print(completion.choices[0].message['content'])