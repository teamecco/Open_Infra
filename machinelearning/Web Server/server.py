# _*_ coding: utf-8 _*_

from flask import Flask, render_template, request
import datetime
import sys
import tensorflow as tf
import pandas as pd
import numpy as np
#import for rabbitmq consuming
import pika
import queue
import threading
import json
from consumer import Threaded_consumer  # thread consumer object
import global_data  # global queue

from inference import Threaded_Inference

#==========================================consumer code=============================================

td = Threaded_consumer()
td.setDaemon(True)
td.start()

#=====================================================================================================

app = Flask(__name__)

X1 = tf.placeholder(tf.float32, shape=[None,5])
Y1 = tf.placeholder(tf.float32, shape=[None,1])

X2 = tf.placeholder(tf.float32, shape=[None,3])
Y2 = tf.placeholder(tf.float32, shape=[None,1])


model = tf.global_variables_initializer()

serverweight = "./model/server.txt"
machineweight = "./model/machine.txt"

model = tf.global_variables_initializer()

sess = tf.Session()
sess.run(model)
sess2 = tf.Session()
sess2.run(model)

f = open(serverweight)
weight = np.ones((5,1))

for a in range(0,5):
    line = f.readline()
    line = float(line)
    weight[a] = line
line = f.readline()
bias = (float(line))
f.close()

f2 = open(machineweight)
m_weight = np.ones((3,1))

for x in range(0,3):
    temp = f2.readline()
    temp = float(temp)
    m_weight[x] = temp
temp = f2.readline()
m_bias = (float(temp))
f2.close()

W1 = tf.Variable(weight, dtype=tf.float32)
b1 = tf.Variable(bias, dtype=tf.float32)
W2 = tf.Variable(m_weight, dtype=tf.float32)
b2 = tf.Variable(m_bias, dtype=tf.float32)

hypothesis = tf.sigmoid(tf.matmul(X1,W1) + b1)
hypothesis2 = tf.sigmoid(tf.matmul(X2,W2) + b2)

sess = tf.Session()
model = tf.global_variables_initializer()
sess.run(model)

sess2 = tf.Session()
sess2.run(model)

train1_csv = pd.read_csv("../data/positive_train.csv")
train2_csv = pd.read_csv("../data/negative_train.csv")

machine_train1_csv = pd.read_csv("../data/positive.csv")
machine_train2_csv = pd.read_csv("../data/negative.csv")

train_csv = pd.concat([train1_csv,train2_csv], sort=False)
train_csv["pi"] = train_csv["pi"].astype(float)
train_csv["cpu"] = train_csv["cpu"].astype(float)
train_csv["ram"] = train_csv["ram"].astype(float)
train_csv["temp"] = train_csv["temp"].astype(float)
train_csv["humidity"] = train_csv["humidity"].astype(float)

machine_train_csv = pd.concat([machine_train1_csv,machine_train2_csv], sort=False)
machine_train_csv["vibrate"] = machine_train_csv["vibrate"].astype(float)
machine_train_csv["voltage"] = machine_train_csv["voltage"].astype(float)
machine_train_csv["pressure"] = machine_train_csv["pressure"].astype(float)
#=====================================================================================================

inferece_td = Threaded_Inference(train_csv, machine_train_csv, hypothesis, hypothesis2, X1, X2, sess, sess2)
inferece_td.setDaemon(True)
inferece_td.start()

#==========================================================================================================================

@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html')



if __name__ == '__main__':
    app.run(debug=True)
