# def parse_audio_files(parent_dir,sub_dirs,file_ext='*.wav'):
#     features, labels = np.empty((0, 193)), np.empty(0)

   
#     for label, sub_dir in enumerate(sub_dirs):
         
#         items = os.listdir(os.path.join(parent_dir, sub_dir))
#         labels = labels.itemset(label)
    
#         #searches through the input file for any files 
#         #named .wav and adds them to the list
    
#         files = []
#         for names in items:
            
#             if names.endswith(".wav"):
#                 #loc = os.path.join(items[1], names)
                
#                 files.append(names)
               
#                 #print(files)
        
                
#         for fn in files:
            
#             file = os.path.join(parent_dir, sub_dir, fn)
            
#             mfccs, chroma, mel, contrast,tonnetz = extract_feature(file)
#             ext_features = np.hstack([mfccs,chroma,mel,contrast,tonnetz])
#             features = np.vstack([features,ext_features])
        

        
#     return np.array(features), np.array(labels, dtype = np.float)



# def load_data(data_directory):
#     
#     """
#     Returns the features and labels of the wave data. 
#     """
#     directories = [d for d in os.listdir(data_directory) 
#                    if os.path.isdir(os.path.join(data_directory, d))]
# 
#     features, labels = np.empty((0, 193)), []
#     for d in directories:
#         label_directory = os.path.join(data_directory, d)
#         file_names = [os.path.join(label_directory, f) 
#                       for f in os.listdir(label_directory) 
#                       if f.endswith(".wav")]
#         
#         for f in file_names:
# #             images.append(skimage.data.imread(f))
#             X, sample_rate, mfccs = extract_feature(f)    #chroma, mel, mfccs, contrast,tonnetz - items removed for now.
#             ext_features = np.hstack([X,mfccs])
#             features = np.vstack([ext_features])
#             labels.append(int(d))
#             features = np.hstack(X, mfccs)
#               
#     return features, labels



def extract_features(parent_dir,sub_dirs,file_ext="*.wav",bands = 20, frames = 41):
    window_size = 512 * (frames - 1)

    mfccs = []
    labels = []

    for l, sub_dir in enumerate(sub_dirs):
        
        for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
            
            sound_clip, s = librosa.load(fn)
            label = l
            
            for (start, end) in windows(sound_clip, window_size):

                if(len(sound_clip[start:end]) == window_size):
                    signal = sound_clip[start:end]
                    mfcc = librosa.feature.mfcc(y=signal, sr=s, n_mfcc = bands).T.flatten()[:, np.newaxis].T
                    
                    
            mfccs.append(mfcc)
            labels.append(l)
            
    features = np.asarray(mfccs).reshape(len(mfccs),frames,bands)
    return np.array(features), np.array(labels,dtype = np.int)



def wave_batch_generator(batch_size=10,source=Source.DIGIT_WAVES,target=Target.digits): #speaker
    maybe_download(source, DATA_DIR)
    if target == Target.speaker: speakers=get_speakers()
    batch_waves = []
    labels = []
    # input_width=CHUNK*6 # wow, big!!
    files = os.listdir(path)
    while True:
        shuffle(files)
        print("loaded batch of %d files" % len(files))
        for wav in files:
            if not wav.endswith(".wav"):continue
            if target==Target.digits: labels.append(dense_to_one_hot(int(wav[0])))
            elif target==Target.speaker: labels.append(one_hot_from_item(speaker(wav), speakers))
            elif target==Target.first_letter:  label=dense_to_one_hot((ord(wav[0]) - 48) % 32,32)
            else: raise Exception("todo : Target.word label!")
            chunk = load_wav_file(path+wav)
            batch_waves.append(chunk)
            # batch_waves.append(chunks[input_width])
            if len(batch_waves) >= batch_size:
                yield batch_waves, labels
                batch_waves = []  # Reset for next batch
                labels = []

