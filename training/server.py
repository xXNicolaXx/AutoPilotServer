import socket
import sys
import datetime
import csv

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
imageArray = []
dateArray = []
directionArray = []
while True:
    buf = conn.recv(8192)

    if buf[-3:] != b'eof':
        data = data + buf
    else:
        data = data + buf[:-4]
        direction = buf[-4:len(buf) - 3].decode('utf-8')
        directionArray.append(direction)
        imageArray.append(data)
        date_string = str(datetime.datetime.now().timestamp()).replace('.', '')
        dateArray.append(date_string)
        data = b''
    if len(imageArray) >= 10:
        break
print("Saving collected images")
for index, item in enumerate(imageArray):
    with open('./images/' + dateArray[index] + '.jpg', 'wb') as f:
        f.write(item)
print("Saving data to csv")
with open('data.csv', 'w') as writeFile:
    writer = csv.writer(writeFile, delimiter=',', quotechar='|')
    for index, item in enumerate(dateArray):
        writer.writerow(['images/' + item + '.jpg', str(directionArray[index])])

print("Finish")
