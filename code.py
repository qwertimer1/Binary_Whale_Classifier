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