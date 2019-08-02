from subprocess import PIPE, Popen
import paho.mqtt.client as mqtt
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

try:
	while True:
		count += 1
		cpu = get_cpu_temperature()
		pub_data = "{0},{1},{2}".format(count, "pi2", cpu)
		client.publish("sensor1", pub_data)
		time.sleep(1)

except KeyboardInterrupt:
	print("Terminated by Keyboard")	
