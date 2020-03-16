import librosa
import numpy as np
import os
from datetime import datetime
import pandas as pd
import sys
import inspect
import errno
import logging
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split


def reshape_x(x_data):
    def load_into(_filename, _x):
        _x.append(_filename)

    x = []
    for element in x_data:
        load_into(element, x)
    x = np.array(x)

    return np.expand_dims(x, axis=2)


def reshape_y(y_data):
    def load_into(_filename, _y):
        _y.append(_filename)

    y = []
    for element in y_data:
        load_into(element, y)

    return np.array(y)


def extract_features(file_name, nbr_mfcc):

    """
        1. Noramlizes audio properties into sampling rate of 22.05 KHz, bit-depth between -1 and 1, and flatten
        the adutio channel into mono.

        2.) Use MFCC to extract features in speech

        Inputs:
        ---------------
        file_name:          Source of raw wav signal str
        nbr_mfcc:           Number of mfcc to return


        Outputs:
        ---------------
        mfccsscaled:        np.ndarray [shape=(n_mfcc, t)]

    """

    try:
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=nbr_mfcc)
        mfccsscaled = np.mean(mfccs.T ,axis=0)

    except Exception as e:
        print("Error encountered while parsing file: ", file)
        print(e)
        return None

    return mfccsscaled

def extract_simple_features(file_name, sample_frequency, duration):

    """
        1. Noramlizes audio properties into sampling rate of 22.05 KHz, bit-depth between -1 and 1, and flatten
        the adutio channel into mono.


        Inputs:
        ---------------
        file_name:          Source of raw wav signal str
        nbr_mfcc:           Number of mfcc to return


        Outputs:
        ---------------
        mfccsscaled:        np.ndarray [shape=(n_mfcc, t)]

    """

    try:
        audio, sample_rate = librosa.load(file_name, sr=sample_frequency, offset=1.0, duration=duration)


    except Exception as e:
        print("Error encountered while parsing file: ", file)
        print(e)
        return None

    return audio


def return_label(file_path):

    """
        Extract label from filename

        Inputs:
        ---------------
        file_name:  Source of raw wav signal str


        Outputs:
        ---------------
        y:          target as string

    """

    if "silence" in file_path.lower():
        y = 'silence'
    elif "song" in file_path.lower():
        y = 'singing'
    else:
        y = 'speaking'
    return y


def create_train_test_data(df_features, partition=0.2):

    """
        Perform train/test split on data

        Inputs:
        ---------------
        df_features:  Two dimensional data frame with columns feautre and label


        Outputs:
        ---------------
        x_train:          x-train partition
        y_train:          y-train partition
        x_test:           x-test partition
        y_test:           y-test partition


    """

    X = np.array(df_features.feature.tolist())
    y = np.array(df_features.class_label.tolist())

    # Encode the classification labels
    le = LabelEncoder()
    yy = to_categorical(le.fit_transform(y))


    return train_test_split(X, yy, test_size=partition, random_state=42)


def main():

    #Parameters to add to config file
    sample_frequency = 8000
    duration = 2



    logging.basicConfig(format = '%(asctime)s %(levelname)s:%(message)s', level = logging.INFO)

    # Add parent directory to sys.path
    current_directory = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    parent_directory = os.path.dirname(current_directory)
    sys.path.insert(0, parent_directory)

    data_path = os.path.join(parent_directory, "data", "raw")

    #Create date for unique folder name
    logging.info("Creating directory.")

    now = datetime.now()
    timestamp = datetime.timestamp(now)
    features = []
    directory_path = os.path.join(parent_directory, "data", "processed", str(timestamp))

    try:
        os.makedirs(directory_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    logging.info("Extracting features from audio signal.")

    for subdir, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith(".wav"):
                data = extract_simple_features(os.path.join(subdir, file), sample_frequency, duration)
                class_label = return_label(subdir)
                if len(data) != sample_frequency * duration:
                    continue
                features.append([data, class_label])
            else:
                continue

    df_features = pd.DataFrame(features, columns=['feature', 'class_label'])
    x_train, x_test, y_train, y_test = create_train_test_data(df_features, partition=0.2)

    logging.info("Saving files.")

    np.save(os.path.join(directory_path, "x_train.npy"), reshape_x(x_train))
    np.save(os.path.join(directory_path, "y_train.npy"), reshape_y(y_train))
    np.save(os.path.join(directory_path, "x_test.npy"), reshape_x(x_test))
    np.save(os.path.join(directory_path, "y_test.npy"), reshape_y(y_test))



if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.error(" Caught CONTROL-C. Exiting.")
        sys.exit()

