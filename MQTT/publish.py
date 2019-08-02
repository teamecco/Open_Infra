import datetime as dt
import paho.mqtt.client as mqtt

count = 0

broker_address="127.0.0.1"
client2 = mqtt.Client("ClientPublisher")
client2.connect(broker_address)

while True:
	count += 1
	time = dt.datetime.now().strftime("%M%S.%f")
	pub_data = "{0},{1}".format(count, time)

	client2.publish("sensor1", pub_data)
