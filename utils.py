
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import sys
import os

#DATA ANALYSIS TOOLS
import tensorflow as tf
import librosa
import numpy as np
import pandas as pd
import scipy


#FILE IMPORTERS
import csv
import dill

#SYSTEM TOOLS
import os
import glob
import tkinter as tk
from tkinter import filedialog as filedialog
import fnmatch
from itertools import compress


invalid_training_type = "Error invalid training Type"

FLAGS = None


class utils_cls:

    """
    Deprecated Class design
    """
    def __init__(self):

        self.ROOT_PATH = "D:\Masters\Data"
        self.invalid_training_type = invalid_training_type




        # if FLAGS.train == 'Train':
        #     self.directory = FLAGS.training_file_dir
        # elif FLAGS.train == 'Test':
        #     self.directory = FLAGS.testing_file_dir
        # elif FLAGS.train == 'Validation':
        #     self.directory = FLAGS.Validation_file_dir
        # else:
        #     raise self.invalid_training_type




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

    def load_sound_wave(self, parent_dir, sub_dirs, file_ext="*.wav"):
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
                print(len(sound_clip))
            label = l
            # df['t'] = [sound_clip for sound_clip in range(sound_clip)]
            # df['label'] = label
        print(type(sound_clip  ))
        return sound_clip, sr, label

    

        return sound_clip, sr

    # def _convert_to_dataframe(self, sound_clip, label):
    #     df = pd.DataFrame()
    #     df['sw'] =
    #     df['label'] = label
    #     print(df)
    #     return df

    def DILL_pickle_files(self, sounds, fn):

        with open(fn, "wb") as dill_file:
            dill.dump(sounds, dill_file)

    def DILL_unpickle_files(self, fn):
        with open(fn, "rb") as dill_file:
            pickled_sounds = dill.load(dill_file)
            return pickled_sounds

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

    
    #deprecated
    def convert_audio(self):
        """"
        convert_audio gets audio files and converts them to a dataframe and pickles the result
        """


        train_data_directory = os.path.join(self.ROOT_PATH, "Autoencoder test/Training")

        train_directories = self._get_directories(self.ROOT_PATH, train_data_directory)

        self._convert_sound_waves(train_data_directory, train_directories)

        return 0

    def load_data(self, fn):
        with open(fn, "r") as dill_file:
            files = dill.load( dill_file)
        return files

    def get_location():
        """
        Allows user to choose the log file that they wish to clean up
        """
            
        root = tk.Tk()
        dirname = filedialog.askopenfilename(parent=root,initialdir="/",title='Please select a file', filetypes = (("log files", "*.log"),("box files", "*.box"), ("all files", "*.*")))
        base = os.path.basename(dirname)
        name, ext = os.path.splitext(base)
        (filepath, _) = os.path.split(dirname)
        root.quit()
    
        return dirname, filepath , name

class logfile_reformatter:


    def __init__(self,
                 file_in = "/home/Documents/Masters/test4", 
                 file_out = "/home/Documents/Masters/test4", 
                 filepath = "/home/Documents/Masters/",
                 name = "dummy.txt"):

        self.file_in = file_in
        self.file_out = file_out
        self.filepath = filepath
        self.name = name
        
    def get_file_in(self):
        return self.file_in

    def get_file_out(self):
        return self.file_out

    def get_filepath(self):
        return self.filepath

    def get_name(self):
        return self.name

    def set_file_in(self, vals):
        self.file_in = vals

    def set_file_out(self, vals):
        self.file_out = vals    

    def set_filepath(self, vals):
        self.filepath = vals

    def set_name(self, vals):
        self.name = vals

    def reformatter(self):
        
        data = pd.read_csv(self.file_in, delim_whitespace = True, header = None, error_bad_lines=False)
        data = data.dropna(axis = 1, how = 'any')
        df = data.iloc[:,[0, 1]]

        
        #data.drop([3,4,5,6], axis = 1)
        #data.columns = ["start time", "end time"]"/home/Documents/Masters/
        try:
            a = df[1].str.contains("start")
            if a.empty == False:
                a = df.drop(df.index[[0]])
                df = a
        except:
            pass
        vals = list(df.columns.values)
    
        
        
        df.columns = ["start time", "end time"]
        
            
        df.to_csv(self.file_out, sep = ' ',index = False)
        return df
    
    def output_file_creator(self):
        file_out = []
        file_out = self.filepath + '/' + self.name + '.log'
        self.set_file_out(file_out)

    def clean_log_files(self):
        locat = filedialog.askdirectory()
        print(locat)
        for folders, dirs, files in os.walk(locat + "/"):
            print(folders)
            for file in glob.glob(folders + "/*"):
                print(file)
                if file.endswith(".log") or file.endswith(".box"):
                    base = os.path.basename(file)
                    name, _ = os.path.splitext(base)
                    self.set_name(name)
                    (filepath, _) = os.path.split(file)
                    self.set_filepath(filepath)
                    self.output_file_creator()
                    self.set_file_in(file)
                    
                    df = self.reformatter()
    
class audio_builder:
    def __init__(self, 
        filename =  "/home/Documents/Masters/test4/dummy.wav",  
        output_file = "/home/Documents/Masters/test4/dummy_mod.wav",
        output_file_noise ="/home/Documents/Masters/test4/dummy_mod_noise.wav"): 

        self.filename = filename
        self.output_file = output_file
        self.output_file_noise = output_file_noise

    def set_filename(self, vals):
        self.filename = vals
    def get_filename(self):
        return self.filename
    def set_output_file(self, vals):
        self.output_file = vals
    def get_output_file(self):
        return self.output_file
    def set_output_file_noise(self, vals):
        self.output_file_noise = vals
    def get_output_file_noise(self):
        return self.output_file_noise


    def get_and_save_audio(self, start, end):
        """
        Loads in audio snippets according to the log files and outputs snippets wave files.
        
        """      
        rate, _ = scipy.io.wavfile.read(self.get_filename())
        elapsed = end - start
        elapsed60percent = elapsed*.60
        start_edit = start - elapsed60percent
        duration = elapsed + elapsed60percent       
        data, sr = librosa.load(self.get_filename(), sr = rate,  offset = start_edit, duration = duration)
        data_noise, sr_noise = librosa.load(self.get_filename(), sr = rate, offset = end, duration = duration)
        librosa.output.write_wav(self.get_output_file(), data, sr)
        librosa.output.write_wav(self.get_output_file_noise(), data_noise, sr_noise)
        return 0

    def read_text_descriptor_file(self, text_file):
            """
            Reads log files related to the wave file
            Inputs:
            text_file
            
            Outputs:
            data
            """
            
            data = pd.read_csv(text_file, delim_whitespace = True)
            data = data.dropna(axis = 1, how = 'any')
            
            #data = data.drop(labels=["start time", "end time"], axis = 0, inplace = True)
            
        
            
            return data

    def audio_creation(self, outputfilename, df3):
        """
        Reads in the start and end time of each snap shot gets the audio snippet and saves the output to a new audio file.

        """
        output_file_list = []
        output_file_list_noise = []
        for start, end in zip(df3["start time"], df3["end time"]):
        #start_vals.append(start)
            
            
            dirname_strip = outputfilename.strip('.wav')
            
            output_file_list.append(dirname_strip)
            output_file_list.append("_")
            output_file_list.append(str(start))
            output_file_list.append('.wav')
            output_file = ''.join(output_file_list)
            self.set_output_file(output_file)
            output_file_list = []
            
            output_file_list.append(dirname_strip)
            output_file_list.append("_")
            output_file_list.append(str(end))
            output_file_list.append('_noise.wav')
            output_file_noise = ''.join(output_file_list)
            self.set_output_file_noise(output_file_noise)
            output_file_list = []        



            self.get_and_save_audio(start, end)
        return 0


    def audio_stripper(self):
    
        location = filedialog.askdirectory()
        exclude = set(['modified'])
        #allows exclusion of bad folders and modified files
        for root, dirs, files in os.walk(location, topdown=True):
            [dirs.remove(d) for d in list(dirs) if d in exclude]                    
            for file in files:
                #print(os.path.join(root, file))
                if file.endswith(".wav"):
                    filename = os.path.join(root, file)
                    self.set_filename(filename)
                    outputfolder = root + '/modified'
                    if not os.path.exists(outputfolder):
                        os.makedirs(outputfolder)

                    outputfilename = os.path.join(outputfolder, file)
                    directory = os.path.dirname(self.get_filename())
  
                            
                    dirname_remove = filename.split('/')[-1]
                    #print("dir name = ", dirname_remove)
                    name = dirname_remove.split('.') #filename without extension
                    filepath = directory  

                    #Variable initialisation
                    a = []
                    b = []
                    c = []
                    header = ["start time", "end time"]

                    df3 = pd.DataFrame()
                    df = pd.DataFrame()

                    #pattern to match wave file against        
                    pattern = name[0] + '*.log'
                    


                    for f in os.listdir(filepath):   
                        a.append(fnmatch.fnmatch(f, pattern))       
                        b.append(f)
                        
                    #merges match and filename to one df
                    df = pd.DataFrame({'match': a,
                                        'filename': b})  
                    #finds all files in df that pattern match
                    vals = df.loc[(df["match"]==True)]
                    #print(vals)
                    #reads text in items and appends it to dataframe.
                    
                    for items in vals["filename"]:
                        text_file = filepath + '/' + items
                        c.append(self.read_text_descriptor_file(text_file))
                    df3 = df3.append(c)   
                    df3.columns = header
                            #match_df = match_text(directory, name2)
                    self.audio_creation(outputfilename, df3)








