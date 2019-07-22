#!/usr/bin/env python

import spidev
import time
import os

Vcc = 5.0
R1 = 1000

spi = spidev.SpiDev()
spi.open(0,0)

def fsr420_Registor(voltage):
	R = (R1 * Vcc)/voltage - R1
	return R

def ReadChannel(channel):
	adc = spi.xfer([1,(8+channel)<<4,0])
	data = ((adc[1]&3) << 8) + adc[2]
	return data

mcp3008_channel = 0

delay = 1

f = open('fsr402.dat', 'w')
index = 0

try:
	while True:
		analog_level = ReadChannel(mcp3008_channel)
		Vout = analog_level * Vcc / 1024.0
		if(Vout == 0.0):
			Vout = 0.001
		Rfsr = fsr420_Registor(Vout)
		print "Digital:", analog_level, " Voltage:", Vout, " R(K Ohm):", Rfsr / 1000.0
		data = "{} {} {} {}\n".format(index,analog_level,Vout,Rfsr/1000.0)
		f.write(data)
		time.sleep(delay)
		index += 1

except KeyboardInterrupt:
	print "Now Exit"
	f.close()
