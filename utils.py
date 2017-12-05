from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
import librosa
import numpy as np


import os


invalid_training_type = "Error invalid training Type"

FLAGS = None


class utils_cls():

    def __init__(self):
        self.name = FLAGS.TFRecordFile

        def train_checker(FLAGS):
            if FLAGS.train == 'Train':
                self.directory = FLAGS.training_file_dir
            elif FLAGS.train == 'Test':
                self.directory = FLAGS.testing_file_dir
            elif FLAGS.train == 'Validation':
                self.directory = FLAGS.Validation_file_dir
            else:
                raise invalid_training_type
            return directory



    def _int64_feature(self, value):
        return tf.train.Feature(int64_list = tf.train.Int64List(value=[value]))

    def Audio_feature_extractor(self):
        """
        Data is loaded into

        :return:
        """
        x = []
        sr = []
        mfcc = []
        label = []

        items = os.listdir(directory)
        print(items)
        for f in items:

            #       if f.endswith(".wav"):
            #           newlist.append(f)
            #           print('newlist = ' + str(newlist))

            # Loads the files found above in with librosa
            #        for fp in newlist:
            fp = os.path.join(directory, f)
            #            print('fp = ' + str(fp))
            x, sr = librosa.load(fp, 500, duration=5.0)

            if 'Noise' in fp:

                label = 0
            elif 'Minke' in fp:
                label = 1
            else:
                raise Exception('Error')

            mfcc = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=bands).T.flatten()[:, np.newaxis].T
        return x, sr, mfcc, label

    def RecordWriter(self, name, FLAGS):
        """
        Loads data from wave file and saves it as an iterator system
        args:
        name =

        return:

        """
        newlist = []
        dataarray = []
        features = []
        bands = 20
        frames = 41
        label = []
        # Code to initialise the TFRecordWriter API
        filename = os.path.join(name + '.tfrecords')
        writer = tf.python_io.TFRecordWriter(filename)


        print('hello')
        for f in items:
            print(items)
            print(f)
            example = tf.train.Example(features = tf.train.Features(feature={
                'x' : x[f],
                'mfcc': mfcc[f],
                'label': _int64_feature(label[f])

            }))




            writer.write(example.SerializeToString()
                         )


            #dataset = tf.data.Dataset.from_tensor_slices(x)
            #raw_sounds.append(x)

        return 0


    def recordReader(self, ):
        return 0





