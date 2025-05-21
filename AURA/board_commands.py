import serial
import time
import threading

# Settin up the connection
ESP_RIGHT = serial.Serial(port='COM5', baudrate=115200, timeout=0.1)
ESP_LEFT = serial.Serial(port='COM6', baudrate=115200, timeout=0.1)


text_to_num = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9"
}
                    
def send_commands(esp, commands):
    temp = " ".join(commands)
    esp.write(bytes(temp,  'utf-8'))
    time.sleep(int(commands[1]))
    
def load_commands(commands):

    if len(commands) == 1:
        # double diag_right
        if commands[0] == "on":
            print("on")
            t1 = threading.Thread(target=send_commands, args=(ESP_RIGHT, ['diag_right', '5']))
            t2 = threading.Thread(target=send_commands, args=(ESP_LEFT, ['diag_right', '5']))  
            t1.start()          
            t2.start()
            t1.join()
            t2.join()
        elif commands[0] == 'go':
            t1 = threading.Thread(target=send_commands, args=(ESP_RIGHT, ['diag_right_up', '3']))
            t2 = threading.Thread(target=send_commands, args=(ESP_LEFT, ['diag_right', '3']))  
            t1.start()          
            t2.start()
            t1.join()
            t2.join()                 
    else:
        # commands should always be divisable by 2 that way the for loop can increment by 2
        for idx in range(0, len(commands), 2):
            command = commands[idx]
            duration = commands[idx+1]
            
            if duration == 'zero':
                if command == "up":
                    # turn drone 90 degrees or arise/descend depending for 1 seconds depending on direction   
                    t1 = threading.Thread(target=send_commands, args=(ESP_LEFT, [command, text_to_num['one']]))
                    t1.start()
                    t1.join()      
                else:              
                    # turn drone 90 degrees or arise/descend depending for 1 seconds depending on direction   
                    t1 = threading.Thread(target=send_commands, args=(ESP_LEFT, [command, text_to_num['two']]))
                    t1.start()
                    t1.join()
            else:
                # moves the drone
                t1 = threading.Thread(target=send_commands, args=(ESP_RIGHT, [command, text_to_num[duration]]))
                t1.start()
                t1.join()               
            