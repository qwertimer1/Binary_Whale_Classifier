
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory',
                        type = str,
                        default = '/home/tim/Documents/Masters/Data/tfRecord Writer',
                        help = 'Directory to download data files and write the converted result')

    parser.add_argument('--validation_size',
                        type = int,
                        default = 50,
                        help = """
                        Number of examples left for validation
                        """)
    parser.add_argument('--training_file_dir',
                        type = str,
                        default = '/home/tim/Documents/Masters/Data/Autoencoder test/Training/Training',
                        help = 'Location of Data for Training')
    parser.add_argument('--testing_file_dir',
                        type=str,
                        default='/home/tim/Documents/Masters/Data/Autoencoder test/Testing/',
                        help='Location of Data for Testing')
    parser.add_argument('--Validation_file_dir',
                        type=str,
                        default='/home/tim/Documents/Masters/Data/Autoencoder test/Validation/',
                        help='Location of Data for Validation')
    parser.add_argument('--file_dir',
                        type=str,
                        default='/home/tim/Documents/Masters/Data/Autoencoder test/',
                        help='Location of Data')

    FLAGS, unparsed = parser.parse_known_args()


    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
