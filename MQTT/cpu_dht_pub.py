from subprocess import PIPE, Popen
import paho.mqtt.client as mqtt
import Adafruit_DHT
import time

def get_cpu_temperature():
    """get cpu temperature using vcgencmd"""
    process = Popen(['vcgencmd', 'measure_temp'], stdout=PIPE)
    output, _error = process.communicate()
    return float(output[output.index('=') + 1:output.rindex("'")])

count = 0
broker_address="203.253.21.147"
client = mqtt.Client("ClientPublisher")
client.connect(broker_address)	

sensor = Adafruit_DHT.DHT11
pin = 2

try:
	while True:
		h, t = Adafruit_DHT.read_retry(sensor, pin)
		if h is not None and t is not None:
			count += 1
			cpu = get_cpu_temperature()
			pub_data = "{0},{1},{2},{3:0.1f},{4:0.1f}".format(count, "pi3", cpu, t, h)
		client.publish("sensor1", pub_data)
		time.sleep(1)

except KeyboardInterrupt:
	print("Terminated by Keyboard")	
