'''
from socket import *

clientSock = socket(AF_INET, SOCK_STREAM)
clientSock.connect(('13.125.51.21', 8080))

print('연결 확인 됐습니다.')
clientSock.send('I am a client'.encode('utf-8'))

print('메시지를 전송했습니다.')

data = clientSock.recv(1024)
print('받은 데이터 : ', data.decode('utf-8'))
'''

import socket

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

sock.connect(('13.125.96.31', 8080))

sock.send("hello!".encode())
