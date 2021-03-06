import global_data
import threading
import tensorflow as tf
import pandas as pd
from sklearn import preprocessing
import numpy as np
import json
from elasticsearch import Elasticsearch
from elasticsearch import helpers
import datetime

class Threaded_Inference(threading.Thread):
    def __init__(self, train_csv, machine_train_csv, hypothesis,hypothesis2, X1, X2, sess, sess2):
        threading.Thread.__init__(self)
        self.train_csv = train_csv
        self.machine_train_csv = machine_train_csv
        self.hypothesis = hypothesis
        self.hypothesis2 = hypothesis2
        self.X1 = X1
        self.X2 = X2
        self.sess = sess
        self.sess2 = sess2
        self.es = Elasticsearch("http://106.10.40.129:31032/")

    def run(self):
        print("start Inference")
        while True:
            if not global_data.q.empty():
                self.status = np.zeros((4,), dtype=int)  # [0] = device [1] = pi1 [2] = pi2 [3] = pi3
                self.val_dict = json.loads(global_data.q.get())
                self.input1 = pd.Series(
                        [float(self.val_dict['pi1_temp']),
                        float(self.val_dict['pi1_cpu']),
                        float(self.val_dict['pi1_ram']/9),
                        float(self.val_dict['temp']),
                        float(self.val_dict['humidity'])],
                        index = ['pi','cpu','ram','temp','humidity']
                        )
                self.input2 = pd.Series(
                        [float(self.val_dict['pi2_temp']),
                        float(self.val_dict['pi2_cpu']),
                        float(self.val_dict['pi2_ram']/9),
                        float(self.val_dict['temp']),
                        float(self.val_dict['humidity'])],
                        index = ['pi','cpu','ram','temp','humidity']
                        )

                self.input3 = pd.Series(
                        [float(self.val_dict['pi3_temp']),
                        float(self.val_dict['pi3_cpu']),
                        float(self.val_dict['pi3_ram']/9),
                        float(self.val_dict['temp']),
                        float(self.val_dict['humidity'])],
                        index = ['pi','cpu','ram','temp','humidity']
                        )

                self.machine_input = pd.Series(
                        [float(self.val_dict['vibrate']),
                        float(self.val_dict['voltage']),
                        float(self.val_dict['presure'])],
                        index = ['vibrate','voltage','pressure']
                        )

                self.train_csv = self.train_csv.append(self.input1, ignore_index=True)
                self.train_csv = self.train_csv.append(self.input2, ignore_index=True)
                self.train_csv = self.train_csv.append(self.input3, ignore_index=True)

                self.machine_train_csv = self.machine_train_csv.append(self.machine_input, ignore_index=True)
                self.machine_std_scaler = preprocessing.StandardScaler().fit(self.machine_train_csv[["vibrate","voltage","pressure"]])
                self.machine_train_std = self.machine_std_scaler.transform(self.machine_train_csv[["vibrate","voltage","pressure"]])
                self.machine_data = ((self.machine_train_std[-1][0],self.machine_train_std[-1][1],self.machine_train_std[-1][2]),(0,0,0))
                self.machine_arr = np.array(self.machine_data, dtype=np.float32)
                self.machine_x_data = self.machine_arr[0:3]
                self.dict2 = self.sess2.run(self.hypothesis2, feed_dict={self.X2:self.machine_x_data})
                if self.dict2[0] > 0.5:
                    self.status[0] = 1

                self.count = 0
                while self.count < 3 :
                    self.std_scaler = preprocessing.StandardScaler().fit(self.train_csv[["pi","cpu","ram","temp","humidity"]])
                    self.train_std = self.std_scaler.transform(self.train_csv[["pi","cpu","ram","temp","humidity"]])
                    self.data = ((self.train_std[self.count-3][0],
                                self.train_std[self.count-3][1],
                                self.train_std[self.count-3][2],
                                self.train_std[self.count-3][3],
                                self.train_std[self.count-3][4]),(0,0,0,0,0)
                            )

                    self.arr = np.array(self.data, dtype=np.float32)

                    self.x_data = self.arr[0:5]
                    self.dict = self.sess.run(self.hypothesis, feed_dict={self.X1: self.x_data})
                    if self.dict[0] > 0.5:
                        self.status[self.count+1] = 1
                    self.count = self.count + 1
                self.datetime = datetime.datetime.now()
                self.doc1 = {
                    "pi_temp" : self.val_dict['pi1_temp'],
                    "pi_cpu" : self.val_dict['pi1_cpu'],
                    "pi_ram" : self.val_dict['pi1_ram'],
                    "pi_status" : int(self.status[1]),
					"@timestamp" : self.datetime
                }

                self.doc2 = {
                    "pi_temp" : self.val_dict['pi2_temp'],
                    "pi_cpu" : self.val_dict['pi2_cpu'],
                    "pi_ram" : self.val_dict['pi2_ram'],
                    "pi_status" : int(self.status[2]),
					"@timestamp" : self.datetime
                }

                self.doc3 = {
                    "pi_temp" : self.val_dict['pi3_temp'],
                    "pi_cpu" : self.val_dict['pi3_cpu'],
                    "pi_ram" : self.val_dict['pi3_ram'],
                    "pi_status" : int(self.status[3]),
					"@timestamp" : self.datetime
                }
                self.doc4 = {
                    "presure" : self.val_dict['presure'],
                    "vibrate" : self.val_dict['vibrate'],
                    "voltage" : self.val_dict['voltage'],
                    "status" : int(self.status[0]),
					"@timestamp" : self.datetime
                }

                self.es.index(index='pi1', doc_type='pi', body=self.doc1)
                self.es.index(index='pi2', doc_type='pi', body=self.doc2)
                self.es.index(index='pi3', doc_type='pi', body=self.doc3)
                self.es.index(index='device', doc_type='device', body=self.doc4)
