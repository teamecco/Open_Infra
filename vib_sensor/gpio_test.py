import RPi.GPIO as GPIO
import time
import sys
import Adafruit_DHT

sensor = Adafruit_DHT.DHT11
pin = 2

GPIO.setmode(GPIO.BCM)

GPIO.setup(17, GPIO.IN)

try :
        while True:
		humidity, temperature = Adafruit_DHT.read_retry(sensor, pin)
                if GPIO.input(17)==1:
			print("Temp={0:0.1f}*	Humidity={1:0.1f}%".format(temperature, humidity)) 
		time.sleep(2)

                print "Press the button"

except KeyboardIntrrupt:
        GPIO.cleanup()

finally:
	sys.exit(1)
