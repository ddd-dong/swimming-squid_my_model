from typing import Any
import gymnasium as gym
import numpy as np
from gymnasium import spaces
import math

import socket
import threading
import json


class Environment(gym.Env):    
    def __init__(self) -> None:
        super(Environment, self).__init__()
        # action_space = [["UP"], ["DOWN"], ["LEFT"], ["RIGHT"],["NONE"]]
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Discrete(25)
        
        self.pre_reward = 0
        self.state = 0
        self.scene_info = { "frame": 0, "score": 0, "score_to_pass": 0, "squid_x": 0, "squid_y": 0, "squid_h": 1, "squid_w": 1, "squid_lv": 1, "squid_vel": 1, "status": "GAME_ALIVE", "foods": [ ]}
        self.action = 0
        self.observation = 0
        
        self.client = GameClient()        
        self.client.send_data({"command": 4})
        self.client_thread = threading.Thread(target=self.client.start)
        self.client_thread.start()
        self.running = True
    def step(self, action):              
        reward = 0
        
        self.client.send_data({"command": int(action)})
        
        observation = self.__get_obs()
        
        reward = self.__get_reward(action, observation)
        
        if self.scene_info["status"] != "GAME_ALIVE":
            terminated = 1
            
        else:
            terminated = 0
        truncated = 0
  
        
        
        info = {}
        # print(observation)
        return observation, reward, terminated, truncated, info
    
    def reset(self, seed=None, options=None):            
        self.client.send_data({"command": 4})
        observation = self.__get_obs()
        
        
        info = {}        
        return observation, info
    
    def __get_obs(self):                                             
        while self.running:
            if self.client.data != None:                
                observation = self.__process_scene_data(self.client.data)
                self.client.data = None
                return observation
    
    # 設定Observation
    ### to do
    def __process_scene_data(self, scene_info):
        FOOD_TYPES = ["FOOD_1", "FOOD_2", "FOOD_3"]
        GARBAGE_TYPES = ["GARBAGE_1", "GARBAGE_2", "GARBAGE_3"]

        squid_pos = [scene_info["squid_x"], scene_info["squid_y"]]
        all_food_pos, all_garbage_pos = [], []

        for food in scene_info["foods"]:
            if food["type"] in FOOD_TYPES:
                all_food_pos.append([food["x"], food["y"]])
        
        food_direction = self.__get_direction_to_nearest(squid_pos, all_food_pos) if all_food_pos else 0

        for food in scene_info["foods"]:
            if food["type"] in GARBAGE_TYPES:
                all_garbage_pos.append([food["x"], food["y"]])

        garbage_direction = self.__get_direction_to_nearest(squid_pos, all_garbage_pos) if all_garbage_pos else 0

        return food_direction * 5 + garbage_direction

    # 設定reward
    def __get_reward(self, action, observation):
        reward = self.scene_info["score"] - self.pre_reward
        self.pre_reward = self.scene_info["score"] 
        return reward


    def __calculate_distance(self, point1: list, point2: list)->float:
        """
        :判斷兩點距離
        :type point1 : 點1
        :type point2 : 點2   
        :rtype : 距離             
        """
        return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)
    
    def __find_closest_point(self, points:tuple[list], target_point:list):
        """
        :找距離目標點最近的點
        :type points : 所有點的集合
        :type target_point : 目標點
        :rtype: 距離目標點最近的點
        """
        min_distance = float('inf')
        closest_point = None

        for point in points:
            distance = self.__calculate_distance(point, target_point)
            if distance < min_distance:
                min_distance = distance
                closest_point = point

        return closest_point
    
    def __determine_relative_position(self, reference_point, target_point):
        """
        :判斷目標點在參考點的方位
        :type reference_point : 參考點
        :type target_point : 目標點
        :rtype: 目標點在參考點的方位 
        :0-> 沒有食物或是垃圾 1-> 食物或是垃圾在魷魚右邊 2-> 食物或是垃圾在魷魚是上面 3-> 食物或是垃圾在魷魚左邊 4-> 食物或是垃圾在魷魚下面
        """
        # Calculate relative coordinates
        delta_x = target_point[0] - reference_point[0]
        delta_y = target_point[1] - reference_point[1]

        # Determine the region based on the sum and difference of the relative coordinates
        if delta_x + delta_y > 0:
            if delta_x - delta_y > 0:
                return 1  # Right
            else:
                return 2  # Up
        else:
            if delta_x - delta_y > 0:
                return 3  # Down
            else:
                return 4  # Left
            
    def __get_direction_to_nearest(self, squid_pos, items_pos):
        closest_item_pos = self.__find_closest_point(items_pos, squid_pos)
        return self.__determine_relative_position(squid_pos, closest_item_pos)
    
    def close(self):
        self.client.stop()
        self.running = False
        return super().close()

class GameClient:
    def __init__(self, host='localhost', port=12345):
        self.host = host
        self.port = port
        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client.connect((self.host, self.port))
        self.data = None
        self.running = True

    def send_data(self, data):
        
        json_data = json.dumps(data)
        self.client.send(json_data.encode('utf-8'))

    def receive_data(self):    
        while self.running:
            received = self.client.recv(4096).decode('utf-8')
            if not received:
                break
            # print("get value")
            # print(received)
            self.data = json.loads(received)
            # print(f'Received from server: {self.data}')

    def start(self):
        self.thread = threading.Thread(target=self.receive_data)
        self.thread.start()
        

    def stop(self):
        self.running = False
        self.thread.join()
        self.client.close()


