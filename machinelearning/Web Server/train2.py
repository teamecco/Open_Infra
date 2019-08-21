import tensorflow as tf
import pandas as pd
from sklearn import preprocessing
tf.set_random_seed(777)  # for reproducibility

train1_csv = pd.read_csv("../data/negative.csv")
temp1_csv = train1_csv
train2_csv = pd.read_csv("../data/positive.csv")
temp2_csv = train2_csv

train_csv = pd.concat([train1_csv,train2_csv], sort=False)
train_csv["vibrate"] = train_csv["vibrate"].astype(float)
train_csv["voltage"] = train_csv["voltage"].astype(float)
train_csv["pressure"] = train_csv["pressure"].astype(float)

std_scaler = preprocessing.StandardScaler().fit(train_csv[["vibrate","voltage","pressure"]])

train_std = std_scaler.transform(train_csv[["vibrate","voltage","pressure"]])
# x_data = train_csv[["pi","cpu","ram","temp","humidity"]]
x_data = train_std
y_data = train_csv[["label"]]

# placeholders for a tensor that will be always fed.
X = tf.placeholder(tf.float32, shape=[None, 3],name="X")
Y = tf.placeholder(tf.float32, shape=[None, 1],name="Y")

W = tf.Variable(tf.random_normal([3, 1]), name='W')
b = tf.Variable(tf.random_normal([1]), name='b')

# Hypothesis using sigmoid: tf.div(1., 1. + tf.exp(tf.matmul(X, W)))
hypothesis = tf.sigmoid(tf.matmul(X, W) + b, name="h")

# cost/loss function
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) *
                       tf.log(1 - hypothesis))

train = tf.train.GradientDescentOptimizer(learning_rate=1e-3).minimize(cost)

# Accuracy computation
# True if hypothesis>0.5 else False
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

# Launch graph
with tf.Session() as sess:
    # Initialize TensorFlow variables
    sess.run(tf.global_variables_initializer())

    for step in range(30001):
        cost_val, _ = sess.run([cost, train], feed_dict={X: x_data, Y: y_data})
        if step % 500 == 0:
            print(step, cost_val)

    # Accuracy report
    h, c, a = sess.run([hypothesis, predicted, accuracy],
                       feed_dict={X: x_data, Y: y_data})
    print("\nHypothesis: ", h, "\nCorrect (Y): ", c, "\nAccuracy: ", a)
    print("\nWeight : \n",sess.run(W))
    final_weight = sess.run(W)
    final_bias = sess.run(b)
    filepath = "./model/machine.txt"
    file = open(filepath, 'w', encoding='utf8')
    for a in range(0,3):
        line = str(final_weight[a])[1:-1] + "\n"
        file.write(line)
    file.write(str(final_bias)[1:-1])
    file.close()


    print("\nBias : \n",sess.run(b))
