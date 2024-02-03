import os

from stable_baselines3 import PPO
from gym_Environment import Environment as env
keep_training = True
model_name = "model_PPO"

if __name__ == '__main__':
    folder_path = './ml/save'
    os.makedirs(folder_path, exist_ok=True)

    env = env()    
    if keep_training:
        model = PPO.load(f"../ml/save/{model_name}", env=env)
    else:
        # model = PPO("MlpPolicy", env, verbose=1)         # 如果obs使用Discrete 用這個
        model = PPO("MultiInputPolicy", env, verbose=1)    # 如果obs使用Dict 用這個
    model.learn(total_timesteps=500_00, log_interval=1) #100_000

    model.save("../ml/save/model_PPO")
    print("model save")
    env.close()
    print("PPO end")


            