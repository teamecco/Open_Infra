# _*_ coding: utf-8 _*_

from flask import Flask, render_template, request

import datetime
import tensorflow as tf
import pandas as pd
from sklearn import preprocessing
import numpy as np

app = Flask(__name__)

X = tf.placeholder(tf.float32, shape=[None,5])
Y = tf.placeholder(tf.float32, shape=[None,1])
W = tf.Variable(tf.random_normal([5,1]), name="weight")
b = tf.Variable(tf.random_normal([1]), name="bias")

hypothesis = tf.sigmoid(tf.matmul(X,W) + b)

saver = tf.train.Saver()
model = tf.global_variables_initializer()

sess = tf.Session()
sess.run(model)

save_path = "./model/saved.cpkt"
saver.restore(sess,save_path)


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

        train1_csv = pd.read_csv("../data/positive_train.csv")
        temp1_csv = train1_csv
        train2_csv = pd.read_csv("../data/negative_train.csv")
        temp2_csv = train2_csv

        train_csv = pd.concat([train1_csv,train2_csv], sort=False)
        train_csv["pi"] = train_csv["pi"].astype(float)
        train_csv["cpu"] = train_csv["cpu"].astype(float)
        train_csv["ram"] = train_csv["ram"].astype(float)
        train_csv["temp"] = train_csv["temp"].astype(float)
        train_csv["humidity"] = train_csv["humidity"].astype(float)




        label = 0
        input = pd.Series([machine_temp, cpu_usage, ram_usage, temperature, humidity], index = ['pi','cpu','ram','temp','humidity'])
        train_csv = train_csv.append(input, ignore_index=True)

        std_scaler = preprocessing.StandardScaler().fit(train_csv[["pi","cpu","ram","temp","humidity"]])
        train_std = std_scaler.transform(train_csv[["pi","cpu","ram","temp","humidity"]])


        data = ((train_std[-1][0],train_std[-1][1],train_std[-1][2],train_std[-1][3],train_std[-1][4]),(0,0,0,0,0))
        arr = np.array(data, dtype=np.float32)

        x_data = arr[0:5]
        dict = sess.run(hypothesis, feed_dict={X: x_data})
        print(dict[0])
        if dict[0] > 0.5:
            return render_template('index.html',status = "고장")
        if dict[0] <= 0.5:
            return render_template('index.html',status = "정상")

if __name__ == '__main__':
    app.run(debug=True)
