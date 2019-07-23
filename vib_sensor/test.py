import spidev, time

spi = spidev.SpiDev()
spi.open(0,0)

def analog_read(channel):
	r = spi.xfer([1, (8 + channel) << 4, 0])
	adc_out = ((r[1]&3) << 8)+r[2]
	return adc_out

while True:
	reading = analog_read(1)
	voltage = reading * 5.0 / 1024
	print "Digital:", reading, " Voltage:", voltage
	time.sleep(0.3)


