from art_gpt import art_gpt
from math_gpt import math_gpt
from psychology_gpt import psychology_gpt
from environmentalist_gpt import environmentalist_gpt
import time
functions = {
    "art_gpt": art_gpt,
    "math_gpt": math_gpt,
    # "gpt4": gpt4,
    "psychology_gpt": psychology_gpt,
    "environmentalist_gpt": environmentalist_gpt,
}


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


def other_llm(input: str=""):
    # other_llm = "aaa"
    # print("input:", input)
    avaliableActionDescription = action_description()
    for function_name,function in functions.items():
        other_llm += function(avaliableActionDescription,input)
        time.sleep(10)
    print("other_llm", other_llm,"other_llm")
    # return other_llm


if __name__ == "__main__":
    from jsonargparse import CLI
    CLI(other_llm)