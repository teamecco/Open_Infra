from subprocess import PIPE, Popen
import pika
import Adafruit_DHT
import time
from socket import *
import threading
import Queue
import spidev, os
from random import *
import json
import ast

#==============Server=============================================================
class Cserver(threading.Thread):
	
	data = Queue.Queue()

	def __init__(self,s_socket):
		threading.Thread.__init__(self)
		self.s_socket = s_socket

	def run(self):
		global index
		self.c_socket, addr = self.s_socket.accept()
		index = index + 1
		create_thread(self.s_socket)
		thread = threading.Thread(target = self.c_recv)
		thread.daemon = True
		thread.start()

	def c_recv(self):
		while True:
			self.data.put(self.c_socket.recv(1024))
		self.c_socket.close()

	def return_data(self):
		return self.data.get()

def create_thread(s_socket):
	global index
	t.append(Cserver(s_socket))
	t[index].daemon=True
	t[index].start()

t = []
index = 0
s_socket = socket(AF_INET, SOCK_STREAM)
host = '203.253.21.155'
port = 5001
s_socket.bind((host,port))
s_socket.listen(3)
create_thread(s_socket)

#==============Get CPU Data=============================================================

def get_cpu_temperature():
	"""get cpu temperature using vcgencmd"""
	process = Popen(['vcgencmd', 'measure_temp'], stdout=PIPE)
	output, _error = process.communicate()
	return float(output[output.index('=') + 1:output.rindex("'")])

#==============RabbitMQ info=============================================================

broker_address="106.10.38.29"
credentials = pika.PlainCredentials('admin','admin')
connection = pika.BlockingConnection(pika.ConnectionParameters(broker_address,5672,'/',credentials))
channel = connection.channel()
channel.queue_declare(queue='sensor')

#==============RabbitMQ info=============================================================

sensor = Adafruit_DHT.DHT11 # temp, humidity sensor
pin = 2

spi = spidev.SpiDev() # spi setting for analog sensor
spi.open(0,0)

def analog_read(channel):
	r = spi.xfer([1,(8+channel) << 4, 0])
	adc_out = ((r[1]&3) << 8)+r[2]
	return adc_out

#========================================================================================
try:
	while True:
		msg = '{'
		if index != 0:
			for i in (0,index-1):
				msg += t[i].return_data()+', '
		msg += "'pi3' : " +str(get_cpu_temperature())+", "
		h, temp = Adafruit_DHT.read_retry(sensor, pin)
		pre_value = analog_read(0)
		vib_value = analog_read(1) 
		voltage_value = analog_read(2)
		msg += "'humidity' : "+str(h)+", "
		msg += "'temp': "+str(temp)+", "
		msg += "'presure' : "+str(pre_value)+", "
		msg += "'vibrate' : "+str(vib_value)+", "
		msg += "'voltage' : "+str(voltage_value)+"}"

		json_msg = json.dumps(eval(msg))
		
		channel.basic_publish(exchange='', routing_key='sensor', body=json_msg)
		print(json_msg)
		time.sleep(5)

except KeyboardInterrupt:
	print('keyboard interrupt')
	connection.close()

s_socket.close()
