from socket import *
from subprocess import PIPE, Popen
import time

def get_cpu_temperature():
    """get cpu temperature using vcgencmd"""
    process = Popen(['vcgencmd', 'measure_temp'], stdout=PIPE)
    output, _error = process.communicate()
    return float(output[output.index('=') + 1:output.rindex("'")])

HOST = '203.253.21.155'
PORT = 5001

s = socket(AF_INET, SOCK_STREAM)
s.connect((HOST, PORT))

while True:
        msg = "'pi1' : "+str(get_cpu_temperature())+', '
        s.send(msg.encode('utf-8'))
	time.sleep(5)
