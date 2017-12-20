
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#DATA ANALYSIS TOOLS
import tensorflow as tf
import librosa
import numpy as np


#FILE IMPORTERS
import csv
import dill

#SYSTEM TOOLS
import os


invalid_training_type = "Error invalid training Type"

FLAGS = None


class utils_cls():

    def __init__(self):
        self.name = FLAGS.TFRecordFile
        self.ROOT_PATH = "/home/tim/Documents/Masters/Data"
        self.invalid_training_type = invalid_training_type
        self.fn = ########



        if FLAGS.train == 'Train':
            self.directory = FLAGS.training_file_dir
        elif FLAGS.train == 'Test':
            self.directory = FLAGS.testing_file_dir
        elif FLAGS.train == 'Validation':
            self.directory = FLAGS.Validation_file_dir
        else:
            raise self.invalid_training_type




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




            writer.write(example.SerializeToString())


            #dataset = tf.data.Dataset.from_tensor_slices(x)
            #raw_sounds.append(x)

        return 0

    def recordReader(self, ):
        return 0

    def _convert_sound_waves(self, parent_dir, sub_dirs, file_ext="*.wav"):
        """
        convert_sound_waves extracts from the list the amplitude of the audio signal(x)
        and the sampling rate
        :param parent_dir: Location of Whale data for import
        sub_dirs
        file_ext="*.wav"
        :return: x: Audio Signal
                 sr: Sampling Rate
        """
        for l, sub_dir in enumerate(sub_dirs):

            for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
                self.fn = fn
                print("Current File = " + fn)
                sound_clip, sr = librosa.load(fn)
                self._convert_to_csv(sound_clip, fn)
                df = self._convert_to_dataframe(sound_clip, label)
                self._DIL_pickle_files(df, fn)
                label = l

        return 0

    def _convert_to_dataframe(self, sound_clip, label):
        df = DataFrame()
        df['sw'] = [sound_clip for soundclip in range(sound_clip)]
        df['label'] = label
        print(df)
        return df

    def _DIL_pickle_files(df, fn):
        fn = fn + 'DF'
        with open(fn, "w") as dill_file:
            dill.dump(df, dill_file)

    def _convert_to_csv(self, sound_clip, fn):
        fn = fn + ".csv"
        print(fn)
        with open(fn, 'w') as file:
            csv_file = csv.writer(file, delimiter=",")
            csv_file.writerow([sound_clip])
        return 0

    def _get_directories(self, ROOT_PATH, directory):

        directories = [d for d in os.listdir(directory)
                       if os.path.isdir(os.path.join(directory, d))]
        print(directories)
        return directories

    def convert_audio(self):
        """"
        convert_audio gets audio files and converts them to a dataframe and pickles the result
        """


        train_data_directory = os.path.join(self.ROOT_PATH, "Autoencoder test/Training")

        train_directories = self._get_directories(self.ROOT_PATH, train_data_directory)

        self._convert_sound_wave(train_data_directory, train_directories)

        return 0

    def load_data(self, fn):
        with open(fn, "r") as dill_file:
            files = dill.load( dill_file)
        return files


