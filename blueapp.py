import RPi.GPIO as GPIO
import time
import os
from bluetooth import BluetoothSocket, RFCOMM
from recog import relay
from recog import buzzer
import socket
import subprocess

recog_process = subprocess.Popen(['python', 'recog.py'])
flaserver_process = subprocess.Popen(['python','flaserver.py'])
visitor_path = 'static/visitors/visitor0.jpg'

while(1):
    try:
        while(1):
            server_socket = BluetoothSocket(RFCOMM)

            port = 22
            server_socket.bind(("", port))
            server_socket.listen(1)

            client_socket, address = server_socket.accept()
            print("Accepted connection from ", address)
            
            if os.path.exists(visitor_path):
                client_socket.send('visitorexist')
                
            while(1):
                data = client_socket.recv(1024)
                if(data.decode("utf-8") == "newuser"):
                    recog_process.terminate()
                    exec(open('newuser.py', 'rt', encoding='UTF8').read())
                    #완료되면 부저 2초간 울림
                    buzzer()
                    recog_process = subprocess.Popen(['python', 'recog.py'])    
                elif(data.decode("utf-8") == "relay"):
                    relay()
                elif(data.decode("utf-8") == "visitor"):  
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.connect(("pwnbit.kr",443))
                    ipsock = sock.getsockname()[0]
                    
                    client_socket.send('http://'+ipsock+':5000/')
                    
                elif(data.decode("utf-8") == "filedelete"):
                    file_delete = 'static/visitors'
                    
                    if os.path.exists(file_delete):
                        for file in os.scandir(file_delete):
                            os.remove(file.path)
                            
                
                elif(data.decode("utf-8") == "q"):
                    break
            client_socket.close()
            server_socket.close()
    except Exception as e:
        print(e)
        pass