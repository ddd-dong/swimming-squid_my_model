import os

from stable_baselines3 import DQN
from gym_Environment import Environment as env

if __name__ == '__main__':
    folder_path = './save'
    os.makedirs(folder_path, exist_ok=True)
    
    env = env()
    # model = DQN("MlpPolicy", env, verbose=1)         # 如果obs使用Discrete 用這個
    model = DQN("MultiInputPolicy", env, verbose=1)    # 如果obs使用Dict 用這個
    model.learn(total_timesteps=100_000, log_interval=1)

    model.save("../ml/save/model_DQN")
    print("model save")    
    env.close()
    print("DQN end")