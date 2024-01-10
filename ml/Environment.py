import math
class Environment():
    def __init__(self) -> None:                                                                        
        self.action_space = [["UP"], ["DOWN"], ["LEFT"], ["RIGHT"],["NONE"]]
        self.n_actions = len(self.action_space)
        
        self.action = 0 
        self.observation = 0
        self.pre_reward = 0
    
    def set_scene_info(self, Scene_info: dict):
        """
        Stores the given scene information into the environment.

        Parameters:
        scene_info (dict): A dictionary containing environment information.
        """
        self.scene_info = Scene_info        
    
    def reset(self):
        """
        Resets the environment and returns the initial observation.

        Returns:
        observation: The initial state of the environment after reset.
        """
        observation = self.__get_obs(self.scene_info)

        return observation
    
    def step(self, action: int):   
        """
        Executes a given action in the environment and returns the resulting state.

        Parameters:
        action (int): The action to be performed, representing the squid's movement.

        Returns:
        observation: The current state of the environment after the action.
        reward (int): The reward obtained as a result of performing the action.
        done (bool): Indicates whether the game has ended (True if ended, False otherwise).
        info (dict): Additional information about the current state.
        """
        reward = 0
        observation = self.__get_obs(self.scene_info)                  
        
        reward = self.__get_reward(action, observation)
                
        done = self.scene_info["status"] != "GAME_ALIVE"            

        info = {}

        return observation, reward, done, info
    
    def __get_obs(self, scene_info):
        """
        Processes the environmental information to generate an observation.

        Parameters:
        scene_info (dict): A dictionary containing information about the environment.

        Returns:
        int: The computed observation state based on the environment.
        """
        FOOD_TYPES = ["FOOD_1", "FOOD_2", "FOOD_3"]
        GARBAGE_TYPES = ["GARBAGE_1", "GARBAGE_2", "GARBAGE_3"]

        squid_pos = [scene_info["squid_x"], scene_info["squid_y"]]
        all_food_pos = [[food["x"], food["y"]] for food in scene_info["foods"] if food["type"] in FOOD_TYPES]
        all_garbage_pos = [[food["x"], food["y"]] for food in scene_info["foods"] if food["type"] in GARBAGE_TYPES]
        
        
        food_direction = self.__get_direction_to_nearest(squid_pos, all_food_pos) if all_food_pos else 0
        garbage_direction = self.__get_direction_to_nearest(squid_pos, all_garbage_pos) if all_garbage_pos else 0

        return [food_direction, garbage_direction]
    
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
                return 2  # Up
        else:
            if delta_x - delta_y > 0:
                return 3  # Down
            else:
                return 4  # Left
    
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