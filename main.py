from recommender_system import Pibo_recommender
import argparse
import socket
import threading
import pymysql

def handle_client(client_socket):
    '''
    
    '''
    user = client_socket.recv(65535)
    message = user.decode()

    # message parsing
    message = message.split(' ')
    print("message:", message)
    user_id = int(message[0])
    print("user_id:", user_id)
    command = message[1]
    print("command:", command)

    
    # recommend task to user
    if command == 'recommend':
        recommender = Pibo_recommender.recommend_achievement()
        recommend_task = recommender.recommend(user_id)
        print("recommend_task:", recommend_task)
        client_socket.sendall(recommend_task.encode())
        client_socket.close()

    # update achievement evaluation
    elif command == 'update':
        db_information = pymysql.connect(
            user='capstone2',
            passwd='sirlab2020',
            host='127.0.0.1',
            db='',
            charset='utf8'
        )

    # if command is not recommend or update, close socket
    else:
        client_socket.close()

def accept_func(host, port):
    global server_socket
    #IPv4 protocol, TCP type socket object
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    #to handle "Address already in use" error
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    #bind host and port
    server_socket.bind((host, port))
    #server allows a maximum of 5 queued connections
    server_socket.listen(5)

    while True:
        try:
            # if client is connected, return new socket object and client's address
            client_socket, addr = server_socket.accept()            
        except KeyboardInterrupt:
            print("KeyboardInterrupt")
            break

        # accept input with accept() function
        # and after that, handle client with handle_client function using new thread
        print("Connected by", addr)
        client_handler = threading.Thread(
            target=handle_client,
            args=(client_socket,)
        )
        client_handler.daemon = True
        print("client_handler.daemon:", client_handler.daemon)
        client_handler.start()
    



if __name__ == '__main__':
    argparse = argparse.ArgumentParser(description="\nrecommender system\n-h host\n-p port\n-m message\n")
    argparse.add_argument('-h', help="host")
    argparse.add_argument('-p', help="port")
    argparse.add_argument('-m', help="message")

    args = argparse.parse_args()
    try:
        host = args.h
        port = int(args.p)
        message = args.m
    except:
        pass
    
    accept_func(host, port)


    




'''
## SAVING TRAINED MODEL
from surprise import dump
import os
model_filename = "./model.pickle"
print (">> Starting dump")
# Dump algorithm and reload it.
file_name = os.path.expanduser(model_filename)
dump.dump(file_name, algo=algo)
print (">> Dump done")
print(model_filename)

## LOAD SAVED MODEL
def load_model(model_filename):
    print (">> Loading dump")
    from surprise import dump
    import os
    file_name = os.path.expanduser(model_filename)
    _, loaded_model = dump.load(file_name)
    print (">> Loaded dump")
    return loaded_model
'''