import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
import matplotlib.pyplot as plt

# path to json file that stores MFCCs and genre labels for each processed segment
DATA_PATH = "/content/drive/MyDrive/data_10.json"


def load_data(data_path):
    """Loads training dataset from json file.

        :param data_path (str): Path to json file containing data
        :return X (ndarray): Inputs
        :return y (ndarray): Targets
    """

    with open(data_path, "r") as fp:
        data = json.load(fp)

    # convert lists to numpy arrays
    X = np.array(data["mfcc"])
    y = np.array(data["labels"])

    print("Data successfully loaded!")

    return X, y


def plot_history(histories, activation_functions):
    """Plots accuracy and error for training/testing set as a function of the epochs
    for each activation function.

    :param histories: List of training histories for each activation function
    :param activation_functions: List of activation function names
    :return:
    """
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Plot accuracy
    for i, activation_name in enumerate(activation_functions):
        axs[0].plot(histories[i].history['val_accuracy'], label='{}'.format(activation_name))

    axs[0].set_title('Accuracy (Testing Set)')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Accuracy')
    axs[0].legend(loc='lower right')

    # Plot error
    for i, activation_name in enumerate(activation_functions):
        axs[1].plot(histories[i].history['val_loss'], label='{}'.format(activation_name))

    axs[1].set_title('Error (Testing Set)')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Error')
    axs[1].legend(loc='upper right')

    plt.tight_layout()
    plt.show()





if __name__ == "__main__":

    # load data
    X, y = load_data(DATA_PATH)

    # create train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # define activation function names
    activation_functions = ['relu', 'sigmoid', 'tanh', 'softmax', 'linear']

    # list to store training histories
    histories = []

    for activation_name in activation_functions:
        # build network topology
        model = keras.Sequential([
            # input layer
            keras.layers.Flatten(input_shape=(X.shape[1], X.shape[2])),
            # 1st dense layer
            keras.layers.Dense(512, activation=activation_name, kernel_regularizer=keras.regularizers.l2(0.001)),
            keras.layers.Dropout(0.3),
            # 2nd dense layer
            keras.layers.Dense(256, activation=activation_name, kernel_regularizer=keras.regularizers.l2(0.001)),
            keras.layers.Dropout(0.3),
            # 3rd dense layer
            keras.layers.Dense(64, activation=activation_name, kernel_regularizer=keras.regularizers.l2(0.001)),
            keras.layers.Dropout(0.3),
            # output layer
            keras.layers.Dense(10, activation='softmax')
        ])

        # compile model
        optimiser = keras.optimizers.Adam(learning_rate=0.0001)
        model.compile(optimizer=optimiser,
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        model.summary()

        # train model
        history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=32, epochs=100)
        histories.append(history)

    # plot accuracy and error for all activation functions
    plot_history(histories, activation_functions)
	
