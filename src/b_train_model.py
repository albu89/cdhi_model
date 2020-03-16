import numpy as np
import logging
import os,sys,inspect
import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau
import tensorflow.keras.backend as K
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Sequential
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint
from datetime import datetime
import errno
from tensorflow.keras.models import load_model



def conv_model_m5(num_classes):

    """
        This architecture is based on the following paper
        -> https://arxiv.org/pdf/1610.00087.pdf
        and the implementation has been branched from the following repositroy
        --> https://github.com/philipperemy/very-deep-convnets-raw-waveforms

        Inputs:
        ---------------
        num_classes:  The number of to be predicted categories


        Outputs:
        ---------------
        m:           Keras model


    """



    m = Sequential()
    m.add(Conv1D(128,
                 input_shape=[16000, 1],
                 kernel_size=80,
                 strides=4,
                 padding='same',
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=regularizers.l2(l=0.0001)))
    m.add(BatchNormalization())
    m.add(Activation('relu'))
    m.add(MaxPooling1D(pool_size=4, strides=None))
    m.add(Conv1D(128,
                 kernel_size=3,
                 strides=1,
                 padding='same',
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=regularizers.l2(l=0.0001)))
    m.add(BatchNormalization())
    m.add(Activation('relu'))
    m.add(MaxPooling1D(pool_size=4, strides=None))
    m.add(Conv1D(256,
                 kernel_size=3,
                 strides=1,
                 padding='same',
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=regularizers.l2(l=0.0001)))
    m.add(BatchNormalization())
    m.add(Activation('relu'))
    m.add(MaxPooling1D(pool_size=4, strides=None))
    m.add(Conv1D(512,
                 kernel_size=3,
                 strides=1,
                 padding='same',
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=regularizers.l2(l=0.0001)))
    m.add(BatchNormalization())
    m.add(Activation('relu'))
    m.add(MaxPooling1D(pool_size=4, strides=None))
    m.add(Lambda(lambda x: K.mean(x, axis=1)))  # Same as GAP for 1D Conv Layer
    m.add(Dense(num_classes, activation='softmax'))
    return m


def main():

    #parameters
    dataset = "1584182554.771545"
    num_classes = 3
    batch_size = 64
    epochs = 40
    learning_rate = 0.000001
    model_name = "Conv Model M5"

    logging.basicConfig(format = '%(asctime)s %(levelname)s:%(message)s', level = logging.INFO)

    # Add parent directory to sys.path
    current_directory = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    parent_directory = os.path.dirname(current_directory)
    sys.path.insert(0, parent_directory)

    directory = os.path.join(parent_directory, "data", "processed")

    # sort directories by date and select the most recent
    datasets = [data for data in os.listdir(directory) if data != "README.md"]
    dataset_name = max(datasets)

    data_directory = os.path.join(directory, dataset_name)


    now = datetime.now()
    timestamp = datetime.timestamp(now)
    features = []
    model_directory = os.path.join(parent_directory, "models",str(timestamp))

    try:
        os.makedirs(model_directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise




    logging.info("Loading input and target data...")

    x_train = np.load(os.path.join(data_directory, "x_train.npy"), allow_pickle=True)
    y_train = np.load(os.path.join(data_directory, "y_train.npy"), allow_pickle=True)
    x_test = np.load(os.path.join(data_directory, "x_test.npy"), allow_pickle=True)
    y_test = np.load(os.path.join(data_directory, "y_test.npy"), allow_pickle=True)



    checkpoint = ModelCheckpoint(os.path.join(model_directory, "best_model.hdf5"), monitor='val_accuracy', verbose=1,
                                 save_best_only=True, mode='auto', period=1)



    model = conv_model_m5(num_classes=num_classes)

    if model is None:
        exit('Something went wrong!!')

    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    print(model.summary())


    # reduce learning rate if no improvements after 3 epochs
    reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=3, min_lr=learning_rate, verbose=1)
    model.fit(x=x_train,
              y=y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              shuffle=True,
              validation_data=(x_test, y_test),
              callbacks=[reduce_lr,checkpoint])


    model = load_model(os.path.join(model_directory, 'best_model.hdf5'))

    # Evaluating the model on the training and testing set
    acc_train = model.evaluate(x_train, y_train, verbose=0)
    logging.info("Training Accuracy: {}".format(acc_train[1]))

    acc_test = model.evaluate(x_test, y_test, verbose=0)
    logging.info("Testing Accuracy: {}".format(acc_test[1]))

    logging.info("Converting model into tflite...")

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    open(os.path.join(model_directory, model_name+".tflite"), "wb").write(tflite_model)

    model_config = open(os.path.join(model_directory,"experiment_summary.txt"), "w")
    L = ["Dataset: {}\n".format(dataset), "Model name: {}\n".format(model_name),
            "Epochs: {}\n".format(epochs), "Batch size: {}\n".format(batch_size),
                "Learning rate: {}\n".format(learning_rate), "Train Performance: {}\n".format(acc_train[1]),
                        "Test Performance: {}\n".format(acc_test[1])]


    model_config.writelines(L)
    model_config.close()




if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.error(" Caught CONTROL-C. Exiting.")
        sys.exit()
