from typing import Any
import gymnasium as gym
import numpy as np
from gymnasium import spaces
import math
from collections import OrderedDict

import socket
import threading
import json

class Environment(gym.Env):    
    def __init__(self) -> None:
        super(Environment, self).__init__()                   
        
        self.observation_space = spaces.Dict(
            {
                "food_direction" : spaces.Discrete(5),
                "garbage_direction" : spaces.Discrete(5)
            }
        )
        
        self.action_space = spaces.Discrete(5)
        
        self.action_mapping =  [["UP"], ["DOWN"], ["LEFT"], ["RIGHT"],["NONE"]]

        self.pre_reward = 0
        self.state = 0
        self.scene_info = { "frame": 0, "score": 0, "score_to_pass": 0, "squid_x": 0, "squid_y": 0, "squid_h": 1, "squid_w": 1, "squid_lv": 1, "squid_vel": 1, "status": "GAME_ALIVE", "foods": [ ]}
        self.action = 0
        self.observation = 0
        
        self.client = GameClient()                
        self.client.start()
        self.client.send_data({"command": self.action_mapping.index(["NONE"])})
          
    def step(self, action):              
        reward = 0
        
        self.client.send_data({"command": int(action)})
        self.scene_info = self.__get_scene_info()
        observation = self.__get_obs(self.scene_info)
        
        reward = self.__get_reward(action, observation)
        
        if self.scene_info["status"] != "GAME_ALIVE":
            terminated = 1
            
        else:
            terminated = 0
        truncated = 0          
        
        info = {}
        
        return observation, reward, terminated, truncated, info
    
    def reset(self, seed=None, options=None):
        """
        Reset the environment for a new episode.

        This function sends a "NONE" action command to the client, retrieves the scene information,
        and generates an initial observation based on the environment.

        Parameters:
        seed (int or None): A random seed for environment initialization (optional).
        options (dict or None): Additional options for environment initialization (optional).

        Returns:
        OrderedDict: An initial observation containing directions to the nearest food and garbage.
                    Keys: 'food_direction', 'garbage_direction'.
                    Values: The directions to the nearest food and garbage (or 0 if none are present).
        """ 
        self.client.send_data({"command": self.action_mapping.index(["NONE"])})
        self.scene_info = self.__get_scene_info()
        observation = self.__get_obs(self.scene_info)
        
        
        info = {}        
        return observation, info
    
    def __get_scene_info(self):
        """
        Wait for and retrieve scene information from the client.

        This function will wait until `self.client.data` is not None and then
        retrieve the scene information. After retrieval, it sets `self.client.data`
        back to None for future use.

        Returns:
        dict: A dictionary containing information about the scene.
        """
        while self.client.data is None:
            pass
        
        scene_info = self.client.data
        self.client.data = None
        return scene_info

    # 設定Observation
    ### to do            
    def __get_obs(self, scene_info):                                             
        """
        Processes the environmental information to generate an observation.

        Parameters:
        scene_info (dict): A dictionary containing information about the environment.

        Returns:
        OrderedDict: A dictionary with computed observation states based on the environment.
                     Keys: 'food_direction', 'garbage_direction'.
                     Values: The directions to the nearest food and garbage (or 0 if none are present).
        """
        FOOD_TYPES = ["FOOD_1", "FOOD_2", "FOOD_3"]
        GARBAGE_TYPES = ["GARBAGE_1", "GARBAGE_2", "GARBAGE_3"]

        squid_pos = [scene_info["squid_x"], scene_info["squid_y"]]
        all_food_pos = [[food["x"], food["y"]] for food in scene_info["foods"] if food["type"] in FOOD_TYPES]
        all_garbage_pos = [[food["x"], food["y"]] for food in scene_info["foods"] if food["type"] in GARBAGE_TYPES]
        
        # Compute the direction to the nearest food and garbage, or 0 if none are present
        food_direction = self.__get_direction_to_nearest(squid_pos, all_food_pos) if all_food_pos else 0
        garbage_direction = self.__get_direction_to_nearest(squid_pos, all_garbage_pos) if all_garbage_pos else 0

        # Return an ordered dictionary containing the computed directions
        return OrderedDict([('food_direction', food_direction), ('garbage_direction', garbage_direction)])
        

    # 設定reward
    ### to do
    def __get_reward(self, action: int , observation: int):
        """
        Calculates the reward based on the given action and observation.

        Parameters:
        action (int): The selected action.
        observation (int): The current observation state.

        Returns:
        int: The calculated reward based on the action and observation.
        """
        reward = self.scene_info["score"] - self.pre_reward
        self.pre_reward = self.scene_info["score"]        
        return reward

    def __calculate_distance(self, point1: list, point2: list)->float:
        """
        Calculates the Euclidean distance between two points.

        Parameters:
        point1 (list): The coordinates [x, y] of the first point.
        point2 (list): The coordinates [x, y] of the second point.

        Returns:
        float: The Euclidean distance between point1 and point2.
        """
        return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)
    
    def __find_closest_point(self, points:list, target_point:list):
        """
        Finds the point closest to the target point from a given set of points.

        Parameters:
        points (list): A collection of points, each point being a list of coordinates [x, y].
        target_point (list): The target point as a list of coordinates [x, y].

        Returns:
        list: The point from 'points' that is closest to 'target_point'. Returns None if 'points' is empty.
        """
        min_distance = float('inf')
        closest_point = None

        for point in points:
            distance = self.__calculate_distance(point, target_point)
            if distance < min_distance:
                min_distance = distance
                closest_point = point

        return closest_point
    
    def __determine_relative_position(self, reference_point: list, target_point: list) -> int:
        """
        Determines the relative position of the target point in relation to the reference point.

        Parameters:
        reference_point (list): The reference point as a list of coordinates [x, y].
        target_point (list): The target point as a list of coordinates [x, y].

        Returns:
        int: The relative position of the target point with respect to the reference point.
            - 0: No food or garbage.
            - 1: Food or garbage is to the right of the squid.
            - 2: Food or garbage is above the squid.
            - 3: Food or garbage is to the left of the squid.
            - 4: Food or garbage is below the squid.
        """
        # Calculate relative coordinates
        delta_x = target_point[0] - reference_point[0]
        delta_y = target_point[1] - reference_point[1]

        # Determine the region based on the sum and difference of the relative coordinates
        if delta_x + delta_y > 0:
            if delta_x - delta_y > 0:
                return 1  # Right
            else:
                return 4  # Up
        else:
            if delta_x - delta_y > 0:
                return 2  # Down
            else:
                return 3  # Left
            
    def __get_direction_to_nearest(self, squid_pos: list, items_pos: list) -> int:
        """
        Determines the direction of the nearest item from the squid's position.

        Parameters:
        squid_pos (list): The squid's position as a list of coordinates [x, y].
        items_pos (list): A list of positions of items, where each position is a list of coordinates [x, y].

        Returns:
        int: The direction of the closest item relative to the squid's position. The return values are:
            - 0: No item is close or items list is empty.
            - 1: Closest item is to the right of the squid.
            - 2: Closest item is above the squid.
            - 3: Closest item is to the left of the squid.
            - 4: Closest item is below the squid.
        """
        closest_item_pos = self.__find_closest_point(items_pos, squid_pos)
        if closest_item_pos is not None:
            return self.__determine_relative_position(squid_pos, closest_item_pos)
        return 0  # Return 0 if no closest item is found
    
    def close(self):
        self.client.stop()
        self.listening = False
        return super().close()

class GameClient:
    def __init__(self, host='localhost', port=12345):
        """
        Initialize the GameClient.

        Parameters:
        host (str): The hostname or IP address of the server (default is 'localhost').
        port (int): The port number to connect to (default is 12345).
        """
        self.host = host
        self.port = port
        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)        
        self.data = None
        self.running = True
        self.thread = None

    def send_data(self, data):        
        """
        Send data to the server.

        Parameters:
        data (dict): The data to be sent in dictionary format.
        """
        json_data = json.dumps(data)
        self.client.send(json_data.encode('utf-8'))

    def receive_data(self):       
        """
        Receive data from the server and update self.data.
        """ 
        while self.running:            
            received = self.client.recv(4096).decode('utf-8')
            if not received:
                break            
            self.data = json.loads(received)
            

    def start(self):
        """
        Start the client thread to receive data from the server.
        """
        self.running = True
        self.client.connect((self.host, self.port))
        self.thread = threading.Thread(target=self.receive_data)
        self.thread.start()
        

    def stop(self):
        """
        Stop the client and close the connection.
        """
        self.running = False
        if self.thread:            
            self.thread.join()        
        self.client.close()
