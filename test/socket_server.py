'''
from socket import *

serverSock = socket(AF_INET, SOCK_STREAM)
serverSock.bind(('', 8080))
serverSock.listen(1)

connectionSock, addr = serverSock.accept()

print(str(addr),'에서 접속이 확인되었습니다.')

data = connectionSock.recv(1024)
print('받은 데이터 : ', data.decode('utf-8'))

connectionSock.send('I am a server.'.encode('utf-8'))
print('메시지를 보냈습니다.')
'''
#-*- coding:utf-8 -*-
import socket

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(('ip-172-31-2-206.ap-northeast-2.compute.internal', 8080))

server.listen(0)

client, addr = server.accept()

data = client.recv(65535)

print("receieved data:", data.decode())
