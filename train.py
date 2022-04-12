from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt
from praudio import utils


def build_perceptron(output_size: int, shape_size, dense1=512, dropout1=0, dense2=0, dropout2=0, dense3=0, learning_rate=0.0001):
    """
    It creates a model with the given parameters.

    :param output_size: The number of classes in the dataset
    :type output_size: int
    :param shape_size: The shape of the input data
    :param dense1: the number of hidden units in the first layer, defaults to 512 (optional)
    :param dropout1: 0.3, defaults to 0 (optional)
    :param dense2: the number of units in the second dense layer, defaults to 0 (optional)
    :param dropout2: 0.3, defaults to 0 (optional)
    :param dense3: The number of neurons in the second dense layer, defaults to 0 (optional)
    :param learning_rate: How quickly to adjust the cost function
    :return: A model object.
    """
    model = keras.Sequential()

    model.add(keras.layers.Flatten(
        input_shape=shape_size[1:]))

    # dense 90
    model.add(keras.layers.Dense(dense1, activation='relu',
              kernel_initializer='he_uniform'))

    # dropout 0.3
    # dense 90
    if dense2:
        model.add(keras.layers.Dropout(dropout1))
        model.add(keras.layers.Dense(dense2, activation='relu',
                  kernel_initializer='he_uniform'))

    # dropout 0.3
    # dense 90
    if dense3:
        model.add(keras.layers.Dropout(dropout2))
        model.add(keras.layers.Dense(dense3, activation='relu',
                  kernel_initializer='he_uniform'))

    model.add(keras.layers.Dense(output_size, activation='softmax'))

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def train_model(model, epochs, batch_size, patience, X_train, y_train, X_validation, y_validation, verbose=2):
    """
    Train the model using the training data and the validation data

    :param model: the model to train
    :param epochs: The number of epochs to train for
    :param batch_size: The number of samples the model trains on at a time
    :param patience: Number of epochs with no improvement after which training will be stopped
    :param X_train: The training data
    :param y_train: The training labels
    :param X_validation: The validation data
    :param y_validation: The validation labels
    :return: The history object is a record of training loss values and metrics values at successive
    epochs, as well as validation loss values and validation metrics values.
    """

    earlystop_callback = tf.keras.callbacks.EarlyStopping(
        monitor="accuracy", min_delta=0.001, patience=patience)

    # train model
    history = model.fit(X_train,
                        y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(X_validation, y_validation),
                        callbacks=[earlystop_callback],
                        verbose=verbose)
    return history
