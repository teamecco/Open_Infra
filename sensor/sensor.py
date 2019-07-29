import spidev, time, os

spi = spidev.SpiDev()
spi.open(0,0)

Vcc = 5.0
R1 = 1000

def fsr420_Registor(voltage):
	R = (R1 * Vcc)/voltage - R1
	return R

def analog_read(channel):
	r = spi.xfer([1,(8+channel) << 4, 0])
	adc_out = ((r[1]&3) << 8)+r[2]
	return adc_out
try :
	while True:
		pres_value = analog_read(0)
		vib_value = analog_read(1)
		print "Pres_value:", pres_value, " vib_value:", vib_value
		time.sleep(0.1)

except KeyboardInterrupt:
	print "Now Exit"

