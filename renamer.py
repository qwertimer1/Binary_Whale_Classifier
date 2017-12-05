import os
import argparse


def file_renamer(args_parsed):

    """
    File renamer loads the filename in from the directory given.
    The file is then renamed to include the species type of the whale.


    :return:
    """
    files = args_parsed.file_loc

    print(type(files))
    for root, dirs, files in os.walk(files):
        print(files)
        if not files:
            continue
        prefix = '.wav'
        for f in files:
            print('hello')
            os.rename(os.path.join(root, f), os.path.join(root, "{}{}".format(f, prefix)))

    return 0

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("file_loc",
                        type=str,
                        default='/home/tim/Documents/Masters/Data/Autoencoder test/Training/',
                        help='Location of data for conversion')

    args_parsed = parser.parse_args()
    file_renamer(args_parsed)


if __name__ == '__main__':

    main()


