#System Imports
import os
import glob


#Math Imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import specgram
import sklearn as sk

#Data Analysis Imports
import scipy.io
import librosa
import librosa.display
%matplotlib inline
import pandas as pd

#NN Import
import tensorflow as tf


## Too be converted to a param in a pickle file using argparse.

ROOT_PATH = "/home/tim/Documents/Masters/Data"



def get_directories(ROOT_PATH, directory):

    directories = [d for d in os.listdir(directory)
                if os.path.isdir(os.path.join(directory, d))]
    print(directories)
    return directories

# def file_reader(file_paths):
#     """
#
#     :param file_paths:
#     :return:
#     """
#
#     items = os.listdir(file_paths)
#
#     # searches through the input file for any files
#     # named .wav and adds them to the list
#
#     files_list = []
#     for names in items:
#         if names.endswith(".wav"):
#             files_list.append(names)
#     return files_list


def load_sound_wave(parent_dir,sub_dirs,file_ext="*.wav"):
    """
    load_sound_wave extracts from the list the amplitude of the audio signal(x)
    and the sampling rate
    :param parent_dir: Location of Whale data for import
    sub_dirs
    file_ext="*.wav"
    :return: x: Audio Signal
             sr: Sampling Rate
    """

    for l, sub_dir in enumerate(sub_dirs):

        for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
            sound_clip, sr = librosa.load(fn)
            label = l




    return sound_clip, sr, label


def one_hot_encode(labels):
    n_labels = len(labels)
    print('labels ', n_labels)
    n_unique_labels = len(np.unique(labels))
    print('unique labels ', n_unique_labels)
    one_hot_encode = np.eye(n_unique_labels)
    print('one Hot', one_hot_encode)

    return one_hot_encode

def lstm_cell(n_hidden,state_is_tuple = True):
  return tf.contrib.rnn.BasicLSTMCell(n_hidden)


def RNN(x, weight, bias, number_of_layers):
    cell = tf.contrib.rnn.MultiRNNCell(
        [lstm_cell(n_hidden, state_is_tuple=True) for _ in range(number_of_layers)])
    output, state = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)
    output = tf.transpose(output, [1, 0, 2])
    last = tf.gather(output, int(output.get_shape()[0]) - 1)
    return tf.nn.softmax(tf.matmul(last, weight) + bias)



def main():
    """

    :return:
    """

    ROOT_PATH = "/home/tim/Documents/Masters/Data"

    train_data_directory = os.path.join(ROOT_PATH, "Autoencoder test/Training")
    test_data_directory = os.path.join(ROOT_PATH, "Autoencoder test/Testing")

    train_directories = get_directories(ROOT_PATH, train_data_directory)
    test_directories = get_directories(ROOT_PATH, test_data_directory)


    raw_sounds_tr, sr_tr, labels_tr  = load_sound_wave(train_data_directory,train_directories,file_ext="*.wav")
    raw_sounds_ts, sr_ts, labels_ts  = load_sound_wave(train_data_directory,train_directories,file_ext="*.wav")




######################################
    tr_labels = one_hot_encode(tr_labels)
    ts_labels = one_hot_encode(ts_labels)

    tf.reset_default_graph()

    learning_rate = 0.01
    training_iters = 1000
    batch_size = 50
    display_step = 200

    # Network Parameters
    n_input = 20
    number_of_layers = 2
    n_steps = 41
    n_hidden = 300
    n_classes = 2

    x = tf.placeholder("float", [None, n_steps, n_input], name='x')
    y = tf.placeholder("float", [None, n_classes], name='y')

    weight = tf.Variable(tf.random_normal([n_hidden, n_classes]))
    bias = tf.Variable(tf.random_normal([n_classes]))

    prediction = RNN(x, weight, bias, number_of_layers)

    # Define loss and optimizer
    loss_f = -tf.reduce_sum(y * tf.log(prediction))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_f)

    # Evaluate model
    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Initializing the variables
    init = tf.global_variables_initializer()

    with tf.Session() as session:
        session.run(init)

        for itr in range(training_iters):
            offset = (itr * batch_size) % (tr_labels.shape[0] - batch_size)
            batch_x = tr_features[offset:(offset + batch_size), :, :]
            batch_x = np.squeeze(batch_x)
            batch_y = tr_labels[offset:(offset + batch_size), :]
            _, c = session.run([optimizer, loss_f], feed_dict={x: batch_x, y: batch_y})

            if itr % display_step == 0:
                # Calculate batch accuracy
                acc = session.run(accuracy, feed_dict={x: batch_x, y: batch_y})
                # Calculate batch loss
                loss = session.run(loss_f, feed_dict={x: batch_x, y: batch_y})
                print("Iter " + str(epoch) + ", Minibatch Loss= " + \
                      "{:.6f}".format(loss) + ", Training Accuracy= " + \
                      "{:.5f}".format(acc))

        print('Test accuracy: ', round(session.run(accuracy, feed_dict={x: ts_features, y: ts_labels}), 3))


    return 0

if __name__ == "__main__":
    main()