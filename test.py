
import random
import tensorflow as tf
import csv
import numpy as np
# Parameters
learning_rate = 0.1
num_steps = 100
batch_size = 100
display_step = 100

# Network Parameters
n_hidden_1 = 1024 # 1st layer number of neurons
n_hidden_2 = 1024 # 2nd layer number of neurons
num_input = 1024
num_classes = 2

# tf Graph input
X = tf.placeholder("float", [None, num_input])
Y = tf.placeholder("float", [None, num_classes])

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([num_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, num_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([num_classes]))
}


# Create model
def neural_net(x):

    layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights['h1']), biases['b1']))
    layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, weights['h2']), biases['b2']))
    out_layer = tf.add(tf.matmul(layer_2, weights['out']), biases['out'])
    return out_layer

# Construct model
logits = neural_net(X)
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
# optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
# train_op = optimizer.minimize(loss_op)

# Evaluate model
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()
saver = tf.train.Saver()
# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)
    saver.restore(sess, "logs/model.ckpt")

    data_set = []
    csv_read_file = csv.reader(open("processed_data_test.csv"))
    for row in csv_read_file :
        row = [int(x) for x in row]
        data_set.append(row)

    test_data = np.array(data_set)

    batch_x, y = test_data[:, :-1], test_data[:, -1]

    batch_y = np.zeros((batch_x.shape[0], 2))
    for i in range(batch_size) :
        if y[i] == 1 :
            batch_y[i, 0] = 1
        else :
            batch_y[i, 1] = 1

    pred, loss, acc = sess.run([prediction, loss_op, accuracy], feed_dict={X: batch_x,
                                                                 Y: batch_y})
    print("Loss= " + "{:.4f}".format(loss) + ", Testing Accuracy= " + "{:.3f}".format(acc))
    writer = csv.writer(open("output.csv", "w"))
    for i in range(pred.shape[0]):
        if round(pred[i,0]) :
            writer.writerow(["em"])
        else :
            writer.writerow(["wp"])
    print(pred)
    # print(pred)
    # save_path = saver.save(sess, "logs/model.ckpt")
    # print("Model saved in path: %s" % save_path)
    # print("Optimization Finished!")

    # # Calculate accuracy for MNIST test images
    # print("Testing Accuracy:", \
    #     sess.run(accuracy, feed_dict={X: mnist.test.images,
    #                                   Y: mnist.test.labels}))