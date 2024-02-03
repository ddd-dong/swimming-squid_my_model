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
        
        # old one
        # self.observation_space = spaces.Dict(
        #     {
        #         "food_direction" : spaces.Discrete(5),
        #         "garbage_direction" : spaces.Discrete(5)
        #     }
        # )
        self.observation_space =spaces.Dict(
            {
                # "nearest_garbage3": spaces.Discrete(5),
                "nearest_garbage3_angle": spaces.Discrete(11), #spaces.Box(low=0, high=360, shape=(1,), dtype=np.float16),
                "nearest_garbage3_distance": spaces.Discrete(5),
                "nearest_garbage": spaces.Discrete(5),
                "nearest_garbage_value": spaces.Discrete(4),
                "nearest_garbage_distance": spaces.Discrete(5),
                "second_nearest_garbage": spaces.Discrete(5),
                "second_nearest_garbage_value": spaces.Discrete(4),
                "nearest_food": spaces.Discrete(5),
                "nearest_food_value": spaces.Discrete(4),
                "nearest_food_distance": spaces.Discrete(5),
                "second_nearest_food": spaces.Discrete(5),
                "second_nearest_food_value": spaces.Discrete(4)
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
    def __get_obs_old(self, scene_info):                                             
        """
        Processes the environmental information to generate an observation.

        Parameters:
        scene_info (dict): A dictionary containing information about the environment.

        Returns:
        OrderedDict: A dictionary with computed observation states based on the environment.
                     Keys: 'food_direction', 'garbage_direction'.
                     Values: The directions to the nearest food and garbage (or 0 if none are present).

        example:
            [{'frame': 117, 'squid_x': 310, 'squid_y': 330, 'squid_w': 40, 'squid_h': 60, 'squid_vel': 10, 'squid_lv': 1, 
            'foods': [{'x': 380, 'y': 310, 'w': 30, 'h': 30, 'type': 'FOOD_1', 'score': 1}, 
            {'x': 275, 'y': 384, 'w': 30, 'h': 30, 'type': 'FOOD_1', 'score': 1}, 
            {'x': 449, 'y': 206, 'w': 30, 'h': 30, 'type': 'FOOD_1', 'score': 1}], 
            'score': 4, 'score_to_pass': 10, 'status': 'GAME_ALIVE'}]
        """
        FOOD_TYPES = ["FOOD_1", "FOOD_2", "FOOD_3"]
        GARBAGE_TYPES = ["GARBAGE_1", "GARBAGE_2", "GARBAGE_3"]
        # print([scene_info])

        squid_pos = [scene_info["squid_x"], scene_info["squid_y"]]
        # x y apaoche
        all_food_pos = [[food["x"], food["y"]] for food in scene_info["foods"] if food["type"] in FOOD_TYPES]
        all_garbage_pos = [[food["x"], food["y"]] for food in scene_info["foods"] if food["type"] in GARBAGE_TYPES]
        
        # Compute the direction to the nearest food and garbage, or 0 if none are present
        food_direction = self.__get_direction_to_nearest(squid_pos, all_food_pos) if all_food_pos else 0
        garbage_direction = self.__get_direction_to_nearest(squid_pos, all_garbage_pos) if all_garbage_pos else 0

        # Return an ordered dictionary containing the computed directions
        return OrderedDict([('food_direction', food_direction), ('garbage_direction', garbage_direction)])
        
    def __get_obs(self, scene_info):
        '''
        v2: add garbage3,secound nearest garbage,secound nearest food
        v3: add value
        v4: add angle of garbage3
        v5: add distance
        '''
        FOOD_TYPES = ["FOOD_1", "FOOD_2", "FOOD_3"]
        GARBAGE_TYPES = ["GARBAGE_1", "GARBAGE_2", "GARBAGE_3"]
        FOOD_value_map = {0:0,1:1,2:2,4:3}
        GARBAGE_value_map = {0:0,-1:1,-4:2,-10:3}
        squid_pos = [scene_info["squid_x"], scene_info["squid_y"]]
        
        all_garbage3_pos = []
        all_garbage_pos = []
        all_food_pos = []
        for element in scene_info["foods"]:
            if element["type"] in GARBAGE_TYPES:
                if element["type"] == "GARBAGE_3":
                    all_garbage3_pos.append([element["x"], element["y"]])
                all_garbage_pos.append([element["x"], element["y"],element])
            elif element["type"] in FOOD_TYPES:
                all_food_pos.append([element["x"], element["y"],element])
        # nearest_garbage3 = self.__get_direction_to_nearest(squid_pos, all_garbage3_pos) if all_garbage3_pos else 0
        # print(all_garbage3_pos)
        all_garbage3_pos = sorted(all_garbage3_pos, key=lambda x: self.__calculate_distance(squid_pos, x))
        # print("S:",all_garbage3_pos)
        _g3_a = 0
        if all_garbage3_pos:
            _g3_a = self.__get_the_angle(squid_pos, all_garbage3_pos[0])
            nearest_garbage3_angle = self.__angle_to_direction(_g3_a)    
        else:
            nearest_garbage3_angle = 0
        nearest_garbage3_distance = self.__distance_classification(self.__calculate_distance(squid_pos, all_garbage3_pos[0])) if all_garbage3_pos else 0
        # print("AAA",nearest_garbage3_angle,type(nearest_garbage3_angle),nearest_garbage3_angle.shape,_g3_a)
        
        
        #find two nearest garbage
        if len(all_garbage_pos) >= 2:
            all_garbage_pos = sorted(all_garbage_pos, key=lambda x: self.__calculate_distance(squid_pos, x))
            nearest_garbage = self.__get_direction_to_nearest(squid_pos, all_garbage_pos)
            nearest_garbage_value = GARBAGE_value_map[all_garbage_pos[0][2]["score"]]
            second_nearest_garbage = self.__get_direction_to_nearest(squid_pos, all_garbage_pos[1:])
            second_nearest_garbage_value = GARBAGE_value_map[all_garbage_pos[1][2]["score"]]
        else:
            nearest_garbage = self.__get_direction_to_nearest(squid_pos, all_garbage_pos) if all_garbage_pos else 0
            nearest_garbage_value = GARBAGE_value_map[all_garbage_pos[0][2]["score"]] if all_garbage_pos else 0
            second_nearest_garbage = 0
            second_nearest_garbage_value = 0
        nearest_food_distance = self.__distance_classification(self.__calculate_distance(squid_pos, all_food_pos[0])) if all_food_pos else 0

        #find two nearest food
        if len(all_food_pos) >= 2:
            all_food_pos = sorted(all_food_pos, key=lambda x: self.__calculate_distance(squid_pos, x))
            nearest_food = self.__get_direction_to_nearest(squid_pos, all_food_pos)
            nearest_food_value = FOOD_value_map[all_food_pos[0][2]["score"]]
            second_nearest_food = self.__get_direction_to_nearest(squid_pos, all_food_pos[1:])
            second_nearest_food_value = FOOD_value_map[all_food_pos[1][2]["score"]]
        else:
            nearest_food = self.__get_direction_to_nearest(squid_pos, all_food_pos) if all_food_pos else 0
            nearest_food_value = FOOD_value_map[all_food_pos[0][2]["score"]] if all_food_pos else 0
            second_nearest_food = 0
            second_nearest_food_value = 0
        nearest_garbage_distance = self.__distance_classification(self.__calculate_distance(squid_pos, all_garbage_pos[0])) if all_garbage_pos else 0

        # print("----")
        # print(nearest_garbage3_angle)
        # print("BBB",[('nearest_garbage3', nearest_garbage3),
        #                     ('nearest_garbage', nearest_garbage),('nearest_garbage_value', nearest_garbage_value),
        #                     ('second_nearest_garbage', second_nearest_garbage), ('second_nearest_garbage_value', second_nearest_garbage_value),
        #                     ('nearest_food', nearest_food),( 'nearest_food_value', nearest_food_value),
        #                     ('second_nearest_food', second_nearest_food),( 'second_nearest_food_value', second_nearest_food_value)])
        # print("----")
        return OrderedDict([('nearest_garbage3_angle',nearest_garbage3_angle),("nearest_garbage3_distance",nearest_garbage3_distance),
                            ('nearest_garbage', nearest_garbage), ('nearest_garbage_value', nearest_garbage_value),( 'nearest_garbage_distance', nearest_garbage_distance),
                            ('second_nearest_garbage', second_nearest_garbage),('second_nearest_garbage_value', second_nearest_garbage_value),
                            ('nearest_food', nearest_food),('nearest_food_value', nearest_food_value),( 'nearest_food_distance', nearest_food_distance),
                            ('second_nearest_food', second_nearest_food),('second_nearest_food_value', second_nearest_food_value)
                        ])
    
    
    '''
    上下左右垃圾跟魚的距離
    '''
    # 設定reward
    ### to do
    def __get_reward(self, action: int , observation: int):
        """
        Calculates the reward based on the given action and observation.

        Parameters:
        action (int): The selected action.
            action_mapping =  [["UP"], ["DOWN"], ["LEFT"], ["RIGHT"],["NONE"]]
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
    
    def __get_the_angle(self, point1: list, point2: list) -> float:
        """
        Calculates the angle between two points.

        Parameters:
        point1 (list): The coordinates [x, y] of the first point.
        point2 (list): The coordinates [x, y] of the second point.

        Returns:
        float: The angle between the two points in degrees.
        """
        delta_x = point2[0] - point1[0]
        delta_y = point2[1] - point1[1]
        _angle = math.degrees(math.atan2(delta_y, delta_x))
        return _angle if _angle >= 0 else _angle + 360

    def __distance_classification(self, distance: float) -> int:
        """
        Classifies the distance into one of five categories.

        Parameters:
        distance (float): The distance to be classified.

        Returns:
        int: The category of the distance, which is one of the following:
            - 0: No item is close or items list is empty.
            - 1: Closest item is very close(<100).
            - 2: Closest item is close(<200).
            - 3: Closest item is far(<300).
            - 4: Closest item is very far(>300).
        """
        if distance < 100:
            return 1
        elif distance < 200:
            return 2
        elif distance < 300:
            return 3
        else:
            return 4


    def __angle_to_direction(self, angle: np.float16) -> int:
        '''
        0~30:1
        30~60:2
        60~120:3
        120~150:4
        150~180:5
        180~210:6
        210~240:7
        240~300:8
        300~330:9
        330~360:10
        no angle:0
        '''
        ANGLE_TO_DIR=[30,60,120,150,180,210,240,300,330,360]
        for i in range(len(ANGLE_TO_DIR)):
            if angle < ANGLE_TO_DIR[i]:
                return i+1
        raise ValueError("angle over 360")
    
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
            - 2: Food or garbage is below the squid.
            - 3: Food or garbage is to the left of the squid.
            - 4: Food or garbage is above the squid.
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
