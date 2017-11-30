from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
import librosa
import numpy as np


import os



FLAGS = None

class utils_cls(self, FLAGS):

    def __init__(self):
        self.name = name


    def _int64_feature(value):
        return tf.train.Feature(int64_list = tf.train.Int64List(value=[value]))

    def RecordWriter( name):
        """
        Loads data from wave file and saves it as an iterator system


        """
        items = os.listdir(FLAGS.training_file_dir)
        newlist = []
        dataarray = []
        features = []
        bands = 20
        frames = 41
        label = []

        #Code to initialise the TFRecordWriter API
        filename = os.path.join( name + '.tfrecords')
        writer = tf.python_io.TFRecordWriter(filename)

        print(items)
        for f in items:

            #       if f.endswith(".wav"):
            #           newlist.append(f)
            #           print('newlist = ' + str(newlist))

            # Loads the files found above in with librosa
            #        for fp in newlist:
            fp = os.path.join(FLAGS.training_file_dir, f)
            #            print('fp = ' + str(fp))
            x, sr = librosa.load(fp, 500, duration=5.0)


            if 'Noise' in fp:

                label = 0
            elif 'Minke'in fp:
                label = 1
            else:
                raise Exception('Error')

            mfcc = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=bands).T.flatten()[:, np.newaxis].T

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

    def recordReader():
        return 0





