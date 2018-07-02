import random
import tensorflow as tf
import csv
import numpy as np
import argparse
import os


class Model:
    def __init__(self, model_checkpoint_path, graph_checkpoint_path):

        # Parameters
        self.learning_rate = 0.1
        self.num_steps = 500
        self.batch_size = 512
        self.display_step = 100

        # Network Parameters
        self.n_hidden_1 = 1024  # 1st layer number of neurons
        self.n_hidden_2 = 1024  # 2nd layer number of neurons
        self.num_input = 1024  # size of the input instances
        self.num_classes = 2  # em , wp
        self.sess = tf.Session()
        self.model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
        self.model_name = "model.ckpt"
        self.graph_name = "tensorboard/1"
        self.set_path(model_checkpoint_path, graph_checkpoint_path)
        self.init()

    def set_path(self, model_checkpoint_path, graph_checkpoint_path):
        if model_checkpoint_path is not None:
            self.model_checkpoint = os.path.join(self.model_path, model_checkpoint_path)
        else:
            self.model_checkpoint = os.path.join(self.model_path, self.model_name)

        if graph_checkpoint_path is not None:
            self.graph_checkpoint = os.path.join(self.model_path, graph_checkpoint_path)
        else:
            self.graph_checkpoint = os.path.join(self.model_path, self.graph_name)

    def restore(self):
        self.saver = tf.train.Saver()
        if os.path.exists(os.path.join((os.path.dirname(self.model_checkpoint)), "checkpoint")):
            self.saver.restore(self.sess, self.model_checkpoint)
            print("model restored")
            return True
        else:
            print("No models found")
            return False

    def save_model(self):
        return self.saver.save(self.sess, self.model_checkpoint)

    def init(self):
        # tf Graph input
        self.X = tf.placeholder("float", [None, self.num_input])
        self.Y = tf.placeholder("float", [None, self.num_classes])

        # Store layers weight & bias
        self.weights = {
            'h1': tf.Variable(tf.random_normal([self.num_input, self.n_hidden_1])),
            'h2': tf.Variable(tf.random_normal([self.n_hidden_1, self.n_hidden_2])),
            'out': tf.Variable(tf.random_normal([self.n_hidden_2, self.num_classes]))
        }
        self.biases = {
            'b1': tf.Variable(tf.random_normal([self.n_hidden_1])),
            'b2': tf.Variable(tf.random_normal([self.n_hidden_2])),
            'out': tf.Variable(tf.random_normal([self.num_classes]))
        }

        tf.summary.histogram("h1", self.weights['h1'])
        tf.summary.histogram("b1", self.biases['b1'])

        tf.summary.histogram("h2", self.weights['h2'])
        tf.summary.histogram("b2", self.biases['b2'])

        tf.summary.histogram("out_w", self.weights['out'])
        tf.summary.histogram("out_b", self.biases['out'])

        # Hidden fully connected layer with 1024 neurons
        layer_1 = tf.nn.relu(tf.add(tf.matmul(self.X, self.weights['h1']), self.biases['b1']))
        tf.summary.histogram("layer1", layer_1)
        # Hidden fully connected layer with 1024 neurons
        layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, self.weights['h2']), self.biases['b2']))
        tf.summary.histogram("layer2", layer_2)
        # Output fully connected layer with a neuron for each class
        logits = tf.add(tf.matmul(layer_2, self.weights['out']), self.biases['out'])
        tf.summary.histogram("out_layer", logits)

        self.prediction = tf.nn.softmax(logits)

        self.loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=self.Y))
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_op = optimizer.minimize(self.loss_op)
        tf.summary.scalar("loss", self.loss_op)

        correct_pred = tf.equal(tf.argmax(self.prediction, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        tf.summary.scalar("accuracy", self.accuracy)

        self.sess.run(tf.global_variables_initializer())
        self.merged_summary = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(self.graph_checkpoint)

    def read_csv_file(self, file_name):

        data_set_list = []
        csv_read_file = csv.reader(open(file_name))

        for row in csv_read_file:
            row = [int(x) for x in row]
            data_set_list.append(row)

        dataset = np.array(data_set_list)
        # print(mini_batch)
        batch_x, y = dataset[:, :-1], dataset[:, -1]

        batch_y = np.zeros((dataset.shape[0], self.num_classes))
        for i in range(dataset.shape[0]):
            if y[i] == 1:
                batch_y[i, 0] = 1
            else:
                batch_y[i, 1] = 1

        # todo remove hard coding
        return np.append(batch_x, batch_y, axis=1)

    def train(self, file_name):

        self.restore()
        if file_name is None:
            file_name = "processed_data.csv"
        # Start training
        self.writer.add_graph(self.sess.graph)
        dataset = self.read_csv_file(file_name)


        for step in range(self.num_steps):
            np.random.shuffle(dataset)
            mini_batch = dataset[0:self.batch_size, :]

            # print(mini_batch)
            batch_x, batch_y = mini_batch[:, :-2], mini_batch[:, -2:]

            self.sess.run(self.train_op, feed_dict={self.X: batch_x, self.Y: batch_y})
            graph_object = self.sess.run(self.merged_summary, feed_dict={self.X: batch_x, self.Y: batch_y})
            self.writer.add_summary(graph_object, step)

            if step % self.display_step == 99 or step == 0:
                loss, acc = self.sess.run([self.loss_op, self.accuracy], feed_dict={self.X: batch_x, self.Y: batch_y})
                print("Step " + str(step) + ", Minibatch Loss= " + "{:.4f}".format(loss) + ", Training Accuracy= " +
                      "{:.3f}".format(acc))
                self.save_model()

        save_path = self.save_model()
        print("Model saved in path: %s" % save_path)
        print("Optimization Finished!")

    def test(self, test_file, output_file):

        if self.restore() is False:
            return

        if test_file is None:
            test_file = "processed_data_test.csv"
        if output_file is None:
            output_file = output_file = "output.csv"

        dataset = self.read_csv_file(test_file)
        batch_x, batch_y = dataset[:, :-2], dataset[:, -2:]
        pred, loss, acc = self.sess.run([self.prediction, self.loss_op, self.accuracy], feed_dict={self.X: batch_x,
                                                                                                   self.Y: batch_y})
        print("Loss= " + "{:.4f}".format(loss) + ", Testing Accuracy= " + "{:.3f}".format(acc))
        writer = csv.writer(open(output_file, "w"))

        for i in range(pred.shape[0]):
            if round(pred[i, 0]):
                writer.writerow(["em"])
            else:
                writer.writerow(["wp"])


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-train", help="by default testing mode. If train args is passed then training mode",
                        action="store_true")
    parser.add_argument("-modelpath", "-mp", help="path where the model will be stored. Path should be like logs/activation function/model.ckpt")
    parser.add_argument("-graphpath", "-gp", help="path where the graph data will be stored")
    parser.add_argument("-training_file", "-tf", help="name of the training file")
    parser.add_argument("-testing_file", "-testf", help="name of the testing file")
    parser.add_argument("-output_file", "-of", help="name of the output file in testing mode")
    args = parser.parse_args()

    model = Model(args.modelpath, args.graphpath)

    if args.train:
        model.train(args.training_file)
    else:
        model.test(args.testing_file, args.output_file)
