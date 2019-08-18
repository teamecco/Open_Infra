# _*_ coding: utf-8 _*_

from flask import Flask, render_template, request

import datetime
import tensorflow as tf
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

        label = 0
        data = ((machine_temp, cpu_usage, ram_usage, temperature, humidity),(0,0,0,0,0))
        arr = np.array(data, dtype=np.float32)

        x_data = arr[0:5]
        dict = sess.run(hypothesis, feed_dict={X: x_data})
        if dict[0] > 0.5:
            return render_template('index.html',status = "정상")
        if dict[0] <= 0.5:
            return render_template('index.html',status = "고장")

if __name__ == '__main__':
    app.run(debug=True)
