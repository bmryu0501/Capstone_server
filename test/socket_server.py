#-*- coding:utf-8 -*-
import socket

ip_adress = 'ip-172-31-2-206.ap-northeast-2.compute.internal'

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(('ip_adress', 8080))

while True:
	server.listen(0)

	client, addr = server.accept()

	data = client.recv(65535)

	print("receieved data:", data.decode())
