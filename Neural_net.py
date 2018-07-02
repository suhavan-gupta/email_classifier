
import random
import tensorflow as tf
import csv
import numpy as np
# Parameters
learning_rate = 0.1
num_steps = 500
batch_size = 512
display_step = 100

# Network Parameters
n_hidden_1 = 1024 # 1st layer number of neurons
n_hidden_2 = 1024 # 2nd layer number of neurons
num_input = 1024 # size of the input instances
num_classes = 2 # em , wp

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

tf.summary.histogram("h1", weights['h1'])
tf.summary.histogram("b1", biases['b1'])

tf.summary.histogram("h2", weights['h2'])
tf.summary.histogram("b2", biases['b2'])

tf.summary.histogram("out_w", weights['out'])
tf.summary.histogram("out_b", biases['out'])


# Create model
def neural_net(x):
    # Hidden fully connected layer with 1024 neurons
    layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights['h1']), biases['b1']))
    tf.summary.histogram("layer1", layer_1)
    # Hidden fully connected layer with 1024 neurons
    layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, weights['h2']), biases['b2']))
    tf.summary.histogram("layer2", layer_2)
    # Output fully connected layer with a neuron for each class
    out_layer = tf.add(tf.matmul(layer_2, weights['out']), biases['out'])
    tf.summary.histogram("out_layer", out_layer)

    return out_layer

# Construct model
logits = neural_net(X)
prediction = logits

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)
tf.summary.scalar("loss", loss_op)

# Evaluate model
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
tf.summary.scalar("accuracy", accuracy)
# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()
saver = tf.train.Saver()
merged_summary = tf.summary.merge_all()
writer = tf.summary.FileWriter("logs/tensorboard/1")
# merged_summary = tf.summary.merge_all()
# writer = tf.summary.FileWriter("logs/tensorboard/0")
# writer.add_graph(sess.graph)

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)


    writer.add_graph(sess.graph)

    try:
        saver.restore(sess, "logs/model.ckpt")
    except:
        print("no models stored yet")

    data_set = []
    csv_read_file = csv.reader(open("processed_data.csv"))
    for row in csv_read_file :
        row = [int(x) for x in row]
        data_set.append(row)

    for step in range(1, num_steps+1):
        mini_batch_temp = random.sample(data_set, batch_size)
        mini_batch = np.array(mini_batch_temp)
        # print(mini_batch)
        batch_x, y = mini_batch[:, :-1], mini_batch[:, -1]

        batch_y = np.zeros((batch_size, 2))
        for i in range(batch_size):
            if y[i] == 1:
                batch_y[i, 0] = 1
            else:
                batch_y[i, 1] = 1

        # np.resize(batch_y, (batch_size, 2))
        # Run optimization op (backprop)
        # print(tf.shape(batch_y))
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
        graph_object = sess.run(merged_summary, feed_dict={X: batch_x, Y: batch_y})
        writer.add_summary(graph_object, step)
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                 Y: batch_y})
            print("Step " + str(step) + ", Minibatch Loss= " +
                  "{:.4f}".format(loss) + ", Training Accuracy= " +
                  "{:.3f}".format(acc))

    save_path = saver.save(sess, "logs/model.ckpt")
    print("Model saved in path: %s" % save_path)
    print("Optimization Finished!")
