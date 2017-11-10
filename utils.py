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


