import socket
import sys
import datetime
import time


HOST = '192.168.1.8'  # this is your localhost
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
queueArray = []
dateArray = []
while True:
    buf = conn.recv(8192)

    if buf[-3:] != b'eof':
        data = data + buf
    else:
        data = data + buf[:-3]
        queueArray.append(data)
        date_string = str(datetime.datetime.now().timestamp()).replace('.', '')
        dateArray.append(date_string)
        data = b''
    if len(queueArray) >= 100:
        break
for index, item in enumerate(queueArray):
    with open('./images/' + dateArray[index] + '.jpg', 'wb') as f:
        f.write(item)
print("Finish")



