from stable_baselines3 import PPO
from gym_Environment import Environment as env

if __name__ == '__main__':
    env = env()
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=1, log_interval=1)

    model.save("../save/model_PPO")
    print("model save")
    env.close()