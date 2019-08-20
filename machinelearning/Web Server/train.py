import tensorflow as tf
import pandas as pd
from sklearn import preprocessing
tf.set_random_seed(777)  # for reproducibility

train1_csv = pd.read_csv("/nfs/data/negative_train.csv")
temp1_csv = train1_csv
train2_csv = pd.read_csv("/nfs/data/positive_train.csv")
temp2_csv = train2_csv

train_csv = pd.concat([train1_csv,train2_csv], sort=False)
train_csv["pi"] = train_csv["pi"].astype(float)
train_csv["cpu"] = train_csv["cpu"].astype(float)
train_csv["ram"] = train_csv["ram"].astype(float)
train_csv["temp"] = train_csv["temp"].astype(float)
train_csv["humidity"] = train_csv["humidity"].astype(float)

std_scaler = preprocessing.StandardScaler().fit(train_csv[["pi","cpu","ram","temp","humidity"]])

train_std = std_scaler.transform(train_csv[["pi","cpu","ram","temp","humidity"]])
# x_data = train_csv[["pi","cpu","ram","temp","humidity"]]
x_data = train_std
y_data = train_csv[["label"]]

# placeholders for a tensor that will be always fed.
X = tf.placeholder(tf.float32, shape=[None, 5])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([5, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# Hypothesis using sigmoid: tf.div(1., 1. + tf.exp(tf.matmul(X, W)))
hypothesis = tf.sigmoid(tf.matmul(X, W) + b)

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

    for step in range(10001):
        cost_val, _ = sess.run([cost, train], feed_dict={X: x_data, Y: y_data})
        if step % 500 == 0:
            print(step, cost_val)

    # Accuracy report
    h, c, a = sess.run([hypothesis, predicted, accuracy],
                       feed_dict={X: x_data, Y: y_data})
    print("\nHypothesis: ", h, "\nCorrect (Y): ", c, "\nAccuracy: ", a)
    saver = tf.train.Saver()
    save_path = saver.save(sess, "/nfs/model/saved.cpkt")
    builder = tf.saved_model.builder.SavedModelBuilder("/nfs/model/trained_model")
    builder.add_meta_graph_and_variables(sess,["serve"])
    builder.save()
