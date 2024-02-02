import os

from stable_baselines3 import PPO
from gym_Environment import Environment as env

if __name__ == '__main__':
    folder_path = './ml/save'
    os.makedirs(folder_path, exist_ok=True)

    env = env()    
    # model = PPO("MlpPolicy", env, verbose=1)         # 如果obs使用Discrete 用這個
    model = PPO("MultiInputPolicy", env, verbose=1)    # 如果obs使用Dict 用這個
    model.learn(total_timesteps=100_000, log_interval=1)

    model.save("../ml/save/model_PPO")
    print("model save")
    env.close()
    print("PPO end")