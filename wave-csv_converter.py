
import os
import glob

import csv
import librosa
import librosa.display
from Pandas import DataFrame
import dill

import os

def convert_sound_waves(parent_dir,sub_dirs,file_ext="*.wav"):
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
            print("Current File = " + fn)
            sound_clip, sr = librosa.load(fn)
            convert_to_csv(sound_clip, fn)
            df = convert_to_dataframe(sound_clip, label)
            DIL_pickle_files(df,fn)
            label = l




    return 0

def convert_to_dataframe(sound_clip, label):
    df = DataFrame()
    df['sw'] = [sound_clip for soundclip in range(sound_clip)]
    df['label']= label
    print(df)
    return df

def DIL_pickle_files(df, fn):
    fn = fn + 'DF'
    with open(fn, "w") as dill_file:
        dill.dump(df, dill_file)


def convert_to_csv(sound_clip, fn):
    fn = fn + ".csv"
    print(fn)
    with open(fn, 'w') as file:
        csv_file = csv.writer(file, delimiter=",")
        csv_file.writerow([sound_clip])
    return 0

def get_directories(ROOT_PATH, directory):

    directories = [d for d in os.listdir(directory)
                if os.path.isdir(os.path.join(directory, d))]
    print(directories)
    return directories


def main():



    ROOT_PATH = "/home/tim/Documents/Masters/Data"

    train_data_directory = os.path.join(ROOT_PATH, "Autoencoder test/Training")

    train_directories = get_directories(ROOT_PATH, train_data_directory)


    convert_sound_wave(train_data_directory, train_directories)


    return 0


if __name__ == "__main__":
    main()