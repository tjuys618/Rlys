# import openai
from langchain_gpt import gpt3

# openai.api_key = "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" 
def art_gpt(avaliableActionDescription,prompt: str = ""):
    input = """"You are an artist. Please analyze the current scene and give the derivation process of artistic behavior and corresponding answers based on your understanding of art knowledge.And there are two people have different ideas about which action should be choose in this situition, you should consider both their opinions and the current scenario."""+avaliableActionDescription+"""
few shot:
In this scenario, I perceive the highway as a canvas where each vehicle contributes to a dynamic composition. To maintain harmony and safety within this artistic space, I must ensure balance and avoid disrupting the flow. Accelerating (action "3") may upset this delicate balance and lead to chaos. Changing lanes could disrupt the composition, potentially causing collisions. Thus, maintaining my current speed and position aligns with the fluidity and aesthetic of the highway "artwork." Hence, the action chosen is "1" (IDLE).
(action:"1",IDLE)
---------------------------------------------------------------------------------------
    
scene："""+prompt+""" Now you need to chose your next step, please give me your reason  within 100 words and give me your action only in one word.
You should output in the following format:
<reasoning>
<reasoning>
<repeat untill you have solid reasons for your specific action choice.>

"""
    
    # completion = openai.ChatCompletion.create(
    #   model="gpt-3.5-turbo", 
    #   messages=[{"role": "system", "content": input}]
    #   )
    response = gpt3(input)
    print("art:", response)
    return response

# messages.append({"role":"assisstant","content":gpt_response.choices[0].message})
# print(messages)
# print(completion.choices[0].message['content'])


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(art_gpt)
    # response = art_gpt()
    # print(response)