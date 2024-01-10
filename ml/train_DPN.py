from stable_baselines3 import DQN
from gym_Environment import Environment as env

if __name__ == '__main__':
    env = env()
    model = DQN("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=100_000, log_interval=1)

    model.save("../save/model_DQN")
    print("model save")    
    env.close()
    print("DPN end")