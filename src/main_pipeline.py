"""
CNN classification model for mri images on Alzheimer's disease
used mri images were provided by ADNI
this script was build for usage on my personal computer for testing functionality
"""
import utils
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Dropout
from keras.utils import to_categorical


def prepare(path1: str, path2: str):
    adcoronalimages = utils.load_npy(path1)
    cncoronalimages = utils.load_npy(path2)

    return adcoronalimages, cncoronalimages


def trainnn_fullysupervised_3d(adcoronal, cncoronal):
    """
    Train neural network fully supervised
    :param adcoronal:
    :param cncoronal:
    :return:
    """

    # Combine the AD and CN data along with their labels
    x = adcoronal + cncoronal
    y = [1] * len(adcoronal) + [0] * len(cncoronal)

    # control shapes
    for item in x[:10]:
        print(item.shape)

    print("Padding images")
    x = utils.pad_images(x)

    for item in x[:10]:
        print(item.shape)

    # Convert data and labels to numpy arrays
    x = np.array(x)
    y = np.array(y)

    # Normalize images
    print("Before normalization")
    print('Data type: %s' % x.dtype)
    print('Min: %.3f, Max: %.3f' % (x.min(), x.max()))

    # Calculate the minimum and maximum pixel values in your dataset
    min_val = np.min(x)
    max_val = np.max(x)

    # Normalize the data using Min-Max scaling
    x_normalized = (x - min_val) / (max_val - min_val)

    print("After normalization")
    print('Data type: %s' % x_normalized.dtype)
    print('Min: %.3f, Max: %.3f' % (x_normalized.min(), x_normalized.max()))

    del adcoronal
    del cncoronal
    del x

    # Split the data into training and testing sets
    print("Splitting the data")
    x_train, x_test, y_train, y_test = train_test_split(x_normalized, y, test_size=0.2, random_state=42)

    del x_normalized

    # Reshape the data
    x_train = x_train.reshape(x_train.shape + (1,))
    x_test = x_test.reshape(x_test.shape + (1,))
    image_shape = x_train[0].shape

    print(f"Dataset shapes:\n"
          f"-> x_train: {x_train.shape} | y_train: {y_train.shape}\n"
          f"-> x_test:  {x_test.shape}  | y_test:  {y_test.shape}")


    # Convert labels to categorical one-hot encoding
    y_train = to_categorical(y_train, num_classes=2)
    y_test = to_categorical(y_test, num_classes=2)

    # Define a simple convolutional neural network
    model = Sequential()
    model.add(Conv3D(32, kernel_size=(3, 3, 3), activation='relu', input_shape=image_shape,
                     data_format='channels_last'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv3D(64, kernel_size=(3, 3, 3), activation='relu'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    print("Start of training")
    # Train the model
    model.fit(x_train, y_train, epochs=50, batch_size=10, validation_data=(x_test, y_test), verbose=1, workers=6)

    # Evaluate the model
    print("Evaluating model")
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1, workers=6)
    print(f"Test Accuracy: {test_acc}")


def trainnn_semisupervised_self_learning(adcoronal, cncoronal):
    """
    attempt at semisupervised learning
    :param adcoronal:
    :param cncoronal:
    :return:
    """

    # Combine the AD and CN data along with their labels
    x = adcoronal + cncoronal
    y = [1] * len(adcoronal) + [0] * len(cncoronal)

    # control shapes
    for item in x[:10]:
        print(item.shape)

    print("Padding images")
    x = utils.pad_images(x)

    for item in x[:10]:
        print(item.shape)

    # Convert data and labels to numpy arrays
    x = np.array(x)
    y = np.array(y)

    # Normalize images
    print("Before normalization")
    print('Data type: %s' % x.dtype)
    print('Min: %.3f, Max: %.3f' % (x.min(), x.max()))

    # Calculate the minimum and maximum pixel values in your dataset
    min_val = np.min(x)
    max_val = np.max(x)

    # Normalize the data using Min-Max scaling
    x_normalized = (x - min_val) / (max_val - min_val)

    print("After normalization")
    print('Data type: %s' % x_normalized.dtype)
    print('Min: %.3f, Max: %.3f' % (x_normalized.min(), x_normalized.max()))

    del x
    del adcoronal
    del cncoronal

    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x_normalized, y, test_size=0.2, random_state=42)

    x_labelled, x_unlabelled, y_labelled, y_unlabelled = train_test_split(x_train, y_train, train_size=0.1,
                                                                          random_state=42)

    del x_normalized
    del x_train
    del y
    del y_train

    # Reshape the data
    x_labelled = x_labelled.reshape(x_labelled.shape + (1,))
    x_unlabelled = x_unlabelled.reshape(x_unlabelled.shape + (1,))
    x_test = x_test.reshape(x_test.shape + (1,))
    image_shape = x_labelled[0].shape

    # model def
    model = Sequential()
    model.add(Conv3D(16, kernel_size=(3, 3, 3), activation='relu', data_format='channels_last',
                     input_shape=image_shape))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv3D(32, kernel_size=(3, 3, 3), activation='relu', data_format='channels_last'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(32, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation="softmax"))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    print(model.summary())

    # Train the model iteratively
    for epoch in range(1, 5):
        print(f'epoch {epoch}')
        # Train the model on labeled data
        model.fit(x_labelled, y_labelled, epochs=10, verbose=1, batch_size=10, workers=-1)
        # for data in x_labelled:
        #     i = 0
        #     with tf.GradientTape() as tape:
        #         predictions = model(data)
        #         loss = keras.losses.BinaryCrossentropy(y_labelled[i], predictions)
        #
        #     gradients = tape.gradient(loss, model.trainable_variables)
        #     keras.optimizers.Adam.apply_gradients(zip(gradients, model.trainable_variables))

        # Generate pseudo-labels and confidence scores for unlabeled data
        filtered_pictures, filtered_pseudo_labels = utils.generate_and_filter_pseudo_labels_with_conf_score(model, x_unlabelled)

        # Add filtered pseudo-labels to the labeled data
        x_labelled = np.concatenate((x_labelled, filtered_pictures), axis=0)
        y_labelled = np.concatenate((y_labelled, filtered_pseudo_labels))

        # Train the model on labeled and filtered pseudo-labeled data
        model.fit(x_labelled, y_labelled, epochs=10, verbose=1, batch_size=10, workers=-1)

    # Evaluate the model
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1, workers=6)
    print(f"Test Accuracy: {test_acc}")


def main():
    adcoronal, cncoronal = prepare(
                     path1="../Daten/AD_coronal_cube/*.npy",
                     path2="../Daten/CN_coronal_cube/*.npy")

    trainnn_semisupervised_self_learning(adcoronal=adcoronal, cncoronal=cncoronal)


if __name__ == "__main__":
    main()
