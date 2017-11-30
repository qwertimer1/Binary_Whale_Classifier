import os


def main():
    files =  '/home/tim/Documents/Masters/Data/Autoencoder test/Training/'
    for root, dirs, files in os.walk(files):
        if not files:
            continue
        prefix = '.wav'
        for f in files:
            os.rename(os.path.join(root, f), os.path.join(root, "{}_{}".format(f, prefix)))




if __name__ == '__main__':
    main()

