import socket

ip_adress = '13.209.85.23'

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

sock.connect((ip_adress, 8080))

sock.send("byebye!".encode())
