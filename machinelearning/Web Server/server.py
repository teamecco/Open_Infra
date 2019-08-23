# _*_ coding: utf-8 _*_

from flask import Flask, render_template, request
import datetime
import sys

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

app = Flask(__name__
#=====================================================================================================

inferece_td = Threaded_Inference(train_csv, machine_train_csv, hypothesis, hypothesis2, X1, X2, sess, sess2)
inferece_td.setDaemon(True)
inferece_td.start()

#==========================================================================================================================

@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html')
    if request.method == 'POST':


if __name__ == '__main__':
    app.run(debug=True)
