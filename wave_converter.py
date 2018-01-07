import utils



def main():

    audio_converter = utils.utils_cls()
    sound_clip, sr = audio_converter.convert_audio()
    return 0


if __name__ == "__main__":
    main()