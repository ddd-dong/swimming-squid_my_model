import socket
import threading
import json

class MLPlay:
    def __init__(self,*args, **kwargs):
        print("Initial ml script")        
                   
        self.scene_info = []
        self.server = GameServer()
        server_thread = threading.Thread(target=self.server.start)
        server_thread.start()
        
    def update(self, scene_info: dict, *args, **kwargs):
        """
        Generate the command according to the received scene information
        """       
        action_space =  [["UP"], ["DOWN"], ["LEFT"], ["RIGHT"],["NONE"]]
         
        if self.server.receive_command != None:
            command = self.server.receive_command["command"]            
            self.server.receive_command = None 
            self.server.send_data(scene_info)
        else:
            command = action_space.index(["NONE"])
            

        
        return action_space[command]

    def reset(self):
        """
        Reset the status
        """
        print("reset ml script")
        
        pass

class GameServer:
    def __init__(self, host='localhost', port=12345):
        self.host = host
        self.port = port
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.bind((self.host, self.port))
        self.client_socket = None
        self.receive_command = None
        self.running = True

    def handle_client(self, client_socket):
        self.client_socket = client_socket
        while self.running:
            received = client_socket.recv(4096).decode('utf-8')
            if not received:
                break
            # print("receive", received)
            self.receive_command = json.loads(received)
            
    def send_data(self, data):
        if self.client_socket:
            json_data = json.dumps(data)
            self.client_socket.send(json_data.encode('utf-8'))

    def start(self):
        self.server.listen(1)
        print(f'Server listening on {self.host}:{self.port}...')

        client, address = self.server.accept()
        print(f'Connected to {address}')
        self.handle_client(client)

    def stop(self):
        self.running = False
        self.server.close()