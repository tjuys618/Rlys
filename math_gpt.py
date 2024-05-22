# import openai
from langchain_gpt import gpt3
# openai.api_key = "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" 
def math_gpt(avaliableActionDescription,prompt):
#     completion = openai.ChatCompletion.create(
#       model="gpt-3.5-turbo", 
#       messages=[{"role": "system", "content": """You are a mathematician by profession. Now you are doing a math question related to self-driving vehicles. Please give the mathematical derivation process and the corresponding answer.And there are two people have different ideas about which action should be choose in this situition, you should consider both their opinions and the current scenario."""+avaliableActionDescription+"""
# few shot:
# In this scenario, I perceive the highway as a canvas where each vehicle contributes to a dynamic composition. To maintain harmony and safety within this artistic space, I must ensure balance and avoid disrupting the flow. Accelerating (action "3") may upset this delicate balance and lead to chaos. Changing lanes could disrupt the composition, potentially causing collisions. Thus, maintaining my current speed and position aligns with the fluidity and aesthetic of the highway "artwork." Hence, the action chosen is "1" .
    
# ---------------------------------------------------------------------------------------
    
# scene：
# """+prompt+""" Now you need to chose your next step, please give me your reason  within 100 words and give me your action only in one word.
# You should output in the following format:
# <reasoning>
# <reasoning>
# <repeat untill you have solid reasons for your specific action choice.>"""}]
#      )
    input = """You are a mathematician by profession. Now you are doing a math question related to self-driving vehicles. Please give the mathematical derivation process and the corresponding answer.And there are two people have different ideas about which action should be choose in this situition, you should consider both their opinions and the current scenario."""+avaliableActionDescription+"""
few shot:
To determine the next optimal action, I'll calculate the time to collision (TTC) with each vehicle and assess the potential risks. The TTC is calculated using the formula: TTC = Distance / Relative Speed.

For Vehicle 1:
TTC_1 = 13.4777 / (29.9133 - 22.3079) ≈ 0.65 seconds.

For Vehicle 2:
TTC_2 = 33.7534 / (29.9133 - 17.3323) ≈ 1.15 seconds.

For Vehicle 3:
TTC_3 = 62.0428 / (29.9133 - 21.6010) ≈ 2.03 seconds.

For Vehicle 4:
TTC_4 = 80.7282 / (29.9133 - 20.1546) ≈ 3.44 seconds.

Given the shortest TTC with Vehicle 1, I must take immediate action to avoid a potential collision. Therefore, I'll accelerate (action 3: FASTER) to increase the distance between me and Vehicle 1, aligning with traffic laws to ensure safe driving conditions.
(action 3: FASTER)
---------------------------------------------------------------------------------------
    
scene：
"""+prompt+""" Now you need to chose your next step, please give me your reason  within 100 words and give me your action only in one word.
You should output in the following format:
<reasoning>
<reasoning>
<repeat untill you have solid reasons for your specific action choice.>"""
    response = gpt3(input)
    print("math_gpt:",response)
    return response
# messages.append({"role":"assisstant","content":gpt_response.choices[0].message})
# print(messages)
# print(completion.choices[0].message['content'])