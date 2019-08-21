# _*_ coding: utf-8 _*_

from flask import Flask, render_template, request
import datetime
import tensorflow as tf
import pandas as pd
from sklearn import preprocessing
import numpy as np

#import for rabbitmq consuming
import pika
import queue
import threading
import json

#==========================================consumer code=============================================

q = queue.Queue() # store IoT data from rabbitmq(raspberry pi)

class Threaded_consumer(threading.Thread):
    def callback(self, ch, method, properties, body):
        q.put(body)
        print(q.get())

    def __init__(self):
        threading.Thread.__init__(self)
        
        self.HOST = '106.10.38.29'
        self.PORT = 5672
        self.Virtual_Host = '/'
        self.credentials = pika.PlainCredentials('admin', 'admin')
        self.parameters = pika.ConnectionParameters(self.HOST, self.PORT,self.Virtual_Host, self.credentials)
        self.connection = pika.BlockingConnection(self.parameters)
        self.channel = self.connection.channel()

        self.channel.basic_consume(on_message_callback=self.callback,queue='sensor', auto_ack=True)

    def run(self):
        print('start consuming')
        self.channel.start_consuming()

td = Threaded_consumer()
td.setDaemon(True)
td.start()


#=====================================================================================================        

app = Flask(__name__)

X = tf.placeholder(tf.float32, shape=[None,5])
Y = tf.placeholder(tf.float32, shape=[None,1])
W = tf.Variable(tf.random_normal([5,1]), name="weight")
b = tf.Variable(tf.random_normal([1]), name="bias")

X2 = tf.placeholder(tf.float32, shape=[None,3])
Y2 = tf.placeholder(tf.float32, shape=[None,1])
W2 = tf.Variable(tf.random_normal([3,1]), name="weight")
b2 = tf.Variable(tf.random_normal([1]), name="bias")

hypothesis = tf.sigmoid(tf.matmul(X,W) + b)
hypothesis2 = tf.sigmoid(tf.matmul(X2,W2) + b2)

saver = tf.train.Saver()
saver2 = tf.train.Saver()

model = tf.global_variables_initializer()

sess = tf.Session()
sess.run(model)
sess2 = tf.Session()
sess2.run(model)

save_path = "./model/saved.cpkt"
saver.restore(sess,save_path)

save_path2 = "./model2/saved.cpkt"
saver2.restore(sess2,save_path2)

@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html')
    if request.method == 'POST':
        machine_temp = float(request.form['machine_temp'])
        cpu_usage = float(request.form['cpu_usage'])
        ram_usage = float(request.form['ram_usage'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        vibrate = float(request.form['vibrate'])
        voltage = float(request.form['voltage'])
        pressure = float(request.form['pressure'])

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

        input = pd.Series([machine_temp, cpu_usage, ram_usage, temperature, humidity], index = ['pi','cpu','ram','temp','humidity'])
        train_csv = train_csv.append(input, ignore_index=True)

        machine_input = pd.Series([vibrate, voltage, pressure], index = ['vibrate','voltage','pressure'])
        machine_train_csv = machine_train_csv.append(machine_input, ignore_index=True)

        std_scaler = preprocessing.StandardScaler().fit(train_csv[["pi","cpu","ram","temp","humidity"]])
        train_std = std_scaler.transform(train_csv[["pi","cpu","ram","temp","humidity"]])

        machine_std_scaler = preprocessing.StandardScaler().fit(machine_train_csv[["vibrate","voltage","pressure"]])
        machine_train_std = std_scaler.transform(machine_train_csv[["vibrate","voltage","pressure"]])

        data = ((train_std[-1][0],train_std[-1][1],train_std[-1][2],train_std[-1][3],train_std[-1][4]),(0,0,0,0,0))
        machine_data = ((train_std[-1][0],train_std[-1][1],train_std[-1][2]),(0,0,0))
        arr = np.array(data, dtype=np.float32)
        machine_arr = np.array(machine_data, dtype=np.float32)

        x_data = arr[0:5]
        machine_x_data = arr[0:3]

        dict = sess.run(hypothesis, feed_dict={X: x_data})
        dict2 = sess2.run(hypothesis, feed_dict={X2: machine_x_data})
        print(dict[0])
        if dict[0] > 0.5:
            if dict2[0] > 0.5:
                return render_template('index.html',status = "정상",status2 = "정상")
            if dict2[0] <= 0.5:
                return render_template('index.html',status = "정상",status2 = "고장")
        if dict[0] <= 0.5:
            if dict2[0] > 0.5:
                return render_template('index.html',status = "고장",status2 = "정상")
            if dict2[0] <= 0.5:
                return render_template('index.html',status = "고장",status2 = "고장")

if __name__ == '__main__':
    app.run(debug=True)
