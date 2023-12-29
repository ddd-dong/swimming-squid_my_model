import math
class Environment():
    def __init__(self) -> None:                                                                        
        self.action_space = [["UP"], ["DOWN"], ["LEFT"], ["RIGHT"],["NONE"]]
        self.n_actions = len(self.action_space)

           
        
                       
        
        


        
        self.pre_reward = 0
        
        # 遊戲參數
        self.max_episode = 5000   # 執行最大回合數
        self.episode_ctr = 0      # 執行回合數
        self.step_ctr = 0         # 
        # keep_training = True      # 是否接續上次訓練
        
        
        # if not keep_training:
        #     self.QT.q_table.to_pickle('.\\qtable.pickle')
        
    def set_scene_info(self, Scene_info):
        self.scene_info = Scene_info        

    # 設定Observation
    def reset(self):
        squid_pos = [self.scene_info["squid_x"], self.scene_info["squid_y"]] # 魷魚座標
        food_pos = [] 
        
        garbage_pos = []
        food_direction = 0
        garbage_direction = 0
        
        all_food_pos = []    # 所有食物最座標
        all_garbage_pos = [] # 所有垃圾最座標
        
        observation = 0
        # 找所有的食物
        for i in range(len(self.scene_info["foods"])):       
            if self.scene_info["foods"][i]["type"] == "FOOD_1" or self.scene_info["foods"][i]["type"] == "FOOD_2" or self.scene_info["foods"][i]["type"] == "FOOD_3":
                all_food_pos.append([self.scene_info["foods"][i]["x"], self.scene_info["foods"][i]["y"]])

        if len(all_food_pos) > 0:
            food_pos = self.__find_closest_point(all_food_pos, squid_pos) # 找距離魷魚最近的食物
            food_direction = self.__determine_relative_position(squid_pos, food_pos)  # 判斷食物在魷魚的方位         
        else:
            food_direction = 0 # 魷魚身邊沒有食物
        
        # 找所有的垃圾
        for i in range(len(self.scene_info["foods"])):                
            if self.scene_info["foods"][i]["type"] == "GARBAGE_1" or self.scene_info["foods"][i]["type"] == "GARBAGE_2" or self.scene_info["foods"][i]["type"] == "GARBAGE_3":
                all_garbage_pos.append([self.scene_info["foods"][i]["x"], self.scene_info["foods"][i]["y"]])

        if len(all_garbage_pos) > 0:
                garbage_pos = self.__find_closest_point(all_garbage_pos, squid_pos) # 找距離魷魚最近的垃圾
                garbage_direction = self.__determine_relative_position(squid_pos, garbage_pos) # 判斷食物在魷魚的垃圾
        else:
            garbage_direction = 0 # 魷魚身邊沒有垃圾
        
        observation = food_direction * 5 + garbage_direction   # 當前觀察

        return observation
        
    #設reward
    def step(self, action):      
        reward = 0
        observation = self.reset()                    
        ## to do
        reward = self.scene_info["score"] - self.pre_reward

        self.pre_reward = self.scene_info["score"] 
        
        if self.scene_info["status"] != "GAME_ALIVE":
            done = 1
            
        else:
            done = 0

        info = {}

        return observation, reward, done, info
    
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