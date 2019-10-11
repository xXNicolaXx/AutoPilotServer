import socket
import sys
import datetime


HOST = '192.168.0.176'  # this is your localhost
PORT = 8888

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# socket.socket: must use to create a socket.
# socket.AF_INET: Address Format, Internet = IP Addresses.
# socket.SOCK_STREAM: two-way, connection-based byte streams.
print('socket created')

# Bind socket to Host and Port
try:
    s.bind((HOST, PORT))
except socket.error as err:
    print('Bind Failed, Error Code: ' + str(err))
    sys.exit()

print('Socket Bind Success!')
# listen(): This method sets up and start TCP listener.
s.listen(10)
print('Socket is now listening')
conn, addr = s.accept()
print('Connect with ' + addr[0] + ':' + str(addr[1]))
data = b''
while True:
    buf = conn.recv(4096)
    data = data + buf
    if len(buf) < 4096:
        date_string = str(datetime.datetime.now().timestamp()).replace('.', '')
        with open('./images/' + date_string + '.jpg', 'wb') as f:
            f.write(data)
            f.close()
            data = b''
            conn.send(str.encode("OK\n"))