import os
import sys


from stable_baselines3 import DQN
from gym_Environment import Environment as env


if __name__ == '__main__':
    env = env()
    model = DQN.load("../save/model_DQN")
    observation, info = env.reset()    

    while True:
        action, _states = model.predict(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            print("end")
            env.close()
            break