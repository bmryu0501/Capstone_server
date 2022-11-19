import socket

ip_adress = '13.209.85.23'

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

sock.connect((ip_adress, 8080))

sock.send("recommend 12".encode())

# listen to server
data = sock.recv(65535)
print(data.decode())

sock.close()