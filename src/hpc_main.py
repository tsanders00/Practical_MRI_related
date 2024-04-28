import glob
import numpy as np
import keras
from keras.losses import BinaryCrossentropy
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Dropout
from keras.utils import to_categorical
import tensorflow as tf

def load_npy(directory):
    images = []
    npy_files = glob.glob(directory)

    for file in npy_files:
        array = np.load(file)
        images.append(array)

    return images

def pad_images(list):
    """
    function to pad images to the same size
    :param list: images you want to pad
    :return: padded images
    """
    max_shape = np.max([arr.shape for arr in list], axis=0)
    padded_images = []

    for arr in list:
        pad_width = [(0, max_dim - cur_dim) for max_dim, cur_dim in zip(max_shape, arr.shape)]
        padded_arr = np.pad(arr, pad_width, mode='minimum')
        padded_images.append(padded_arr)

    return padded_images

def generate_and_filter_pseudo_labels_with_conf_score(model, dataset):
    """

    :param model:
    :param dataset:
    :return:
    """
    filtered_pictures = []
    filtered_predictions = []
    filtered_confidence_scores = []

    for data in dataset:
        prediction = model(data[0])
        prediction_probs = tf.nn.softmax(prediction)
        prediction = tf.argmax(prediction)
        confidence_score = tf.reduce_max(prediction_probs)
        if confidence_score >= 0.8:
            filtered_predictions.append(prediction)
            filtered_confidence_scores.append(confidence_score)
            filtered_pictures.append(data)
    filtered_pictures = tf.convert_to_tensor(filtered_pictures)
    filtered_predictions = tf.convert_to_tensor(filtered_predictions)
    filtered_confidence_scores = tf.convert_to_tensor(filtered_confidence_scores)

    return filtered_pictures, filtered_predictions, filtered_confidence_scores


def prepare(path1: str, path2: str, path3: str, path4: str, path5: str, path6: str):
    with tf.device('CPU'):
        adcoronal = load_npy(path1)
        cncoronal = load_npy(path2)
        adsagittal = load_npy(path3)
        cnsagittal = load_npy(path4)
        adaxial = load_npy(path5)
        cnaxial = load_npy(path6)

    return adcoronal, cncoronal, adsagittal, cnsagittal, adaxial, cnaxial


def filter_arrays_for_black_cube(arrays: list):
    result = []
    for arr in arrays:
        if np.max(arr) > 1:
            result.append(arr)
    return result


def trainnn_fullysupervised(adcoronal, cncoronal, adsagittal, cnsagittal, adaxial, cnaxial):
    # Combine the AD and CN data along with their labels
    with tf.device('CPU'):
        x = adcoronal + cncoronal + adsagittal + cnsagittal + adaxial + cnaxial
        y = ([1] * len(adcoronal) + [0] * len(cncoronal) + [1] * len(adsagittal) + [0] * len(cnsagittal) +
             [1] * len(adaxial) + [0] * len(cnaxial))

    # control shapes
    for item in x[:10]:
        print(item.shape)

    print("Padding images")
    x = pad_images(x)

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
    del adsagittal
    del cnsagittal
    del adaxial
    del cnaxial

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
    model.add(Conv3D(32, kernel_size=(3, 3, 3), activation='relu', input_shape=image_shape, data_format='channels_last'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv3D(64, kernel_size=(3, 3, 3), activation='relu'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    print("Start of training")
    model.fit(x_train, y_train, epochs=50, batch_size=10, validation_data=(x_test, y_test), verbose=2, workers=-1)

    # Evaluate the model
    print("Evaluating model")
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2, workers=-1)
    print(f"Test Accuracy: {test_acc}")


def trainnn_semisupervised_self_learning(adcoronal, cncoronal):

    # Combine the AD and CN data along with their labels
    x = adcoronal + cncoronal
    y = [1] * len(adcoronal) + [0] * len(cncoronal)

    x = pad_images(x)

    # Convert data and labels to numpy arrays
    x = np.array(x)
    y = np.array(y)

    # Normalize images
    print('Data type: %s' % x.dtype)
    print('Min: %.3f, Max: %.3f' % (x.min(), x.max()))

    # Calculate the minimum and maximum pixel values in your dataset
    min_val = np.min(x)
    max_val = np.max(x)

    # Normalize the data using Min-Max scaling
    x_normalized = (x - min_val) / (max_val - min_val)

    print('Data type: %s' % x_normalized.dtype)
    print('Min: %.3f, Max: %.3f' % (x_normalized.min(), x_normalized.max()))

    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x_normalized, y, test_size=0.2, random_state=42)

    x_labelled, x_unlabelled, y_labelled, y_unlabelled = train_test_split(x_train, y_train, train_size=0.1,
                                                                          random_state=42)
    # model def
    model = Sequential()
    model.add(Conv3D(32, kernel_size=(3, 3, 3), activation='relu', data_format='channels_last'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(Conv3D(64, kernel_size=(3, 3, 3), activation='relu', data_format='channels_last'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(Flatten())
    model.add(Dense(64, activation="relu"))
    model.add(Dense(2, activation="softmax"))

    model.compile(loss=keras.losses.BinaryCrossentropy, optimizer=keras.optimizers.Adam,
                  metrics=keras.metrics.BinaryAccuracy)

    # Train the model iteratively
    for epoch in range(5):
        print(f'epoch {epoch}')
        # Train the model on labeled data
        for data in x_labelled:
            i = 0
            with tf.GradientTape() as tape:
                predictions = model(data[i])
                loss = keras.losses.BinaryCrossentropy(y_labelled[i], predictions)

            gradients = tape.gradient(loss, model.trainable_variables)
            keras.optimizers.Adam.apply_gradients(zip(gradients, model.trainable_variables))

        # Generate and filter pseudo-labels and confidence scores for unlabeled data
        filtered_pictures, filtered_pseudo_labels, filtered_confidence_scores = (
            generate_and_filter_pseudo_labels_with_conf_score(model, x_unlabelled))

        # Add filtered pseudo-labels to the labeled data
        x_labelled = tf.concat((x_labelled, filtered_pictures), 0)
        y_labelled = np.concatenate((y_labelled, filtered_pseudo_labels))

        # Train the model on labeled and filtered pseudo-labeled data
        for data in x_labelled:
            j = 0
            prediction = model(data[j])
            loss = keras.losses.BinaryCrossentropy(y_labelled[j], prediction)

            gradients = tape.gradient(loss, model.trainable_variables)
            keras.optimizers.Adam.apply_gradients(zip(gradients, model.trainable_variables))

    # Evaluate the model
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1, workers=6)
    print(f"Test Accuracy: {test_acc}")


def trainnn_fully_partial(path1: str, path2: str, path3: str, path4: str,
                          path5: str, path6: str, path7: str, path8: str,
                          path9: str, path10: str, path11: str, path12: str):

    print('Define and compile model')
    # Define a simple convolutional neural network
    model = Sequential()
    model.add(Conv3D(16, kernel_size=(3, 3, 3), activation='relu', input_shape=(256, 256, 256, 1), data_format='channels_last'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv3D(32, kernel_size=(3, 3, 3), activation='relu'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model.summary()

    path_arr = [path1, path2, path3, path4, path5, path6, path7, path8, path9, path10, path11, path12]

    print('Start of processing')
    j = 1

    for i in range(0, len(path_arr) - 1, 2):

        print(f'Iteration number: {j} / {len(path_arr)/2}')

        with tf.device('CPU'):
            ad = load_npy(path_arr[i])
            cn = load_npy(path_arr[i+1])

            # ad = filter_arrays_for_black_cube(ad)
            # cn = filter_arrays_for_black_cube(cn)

            x = ad + cn
            y = [1] * len(ad) + [0] * len(cn)

            del ad
            del cn

            # control shapes
            for item in x[:10]:
                print(item.shape)

            print("Padding images")
            x = pad_images(x)

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

            del x

            print("After normalization")
            print('Data type: %s' % x_normalized.dtype)
            print('Min: %.3f, Max: %.3f' % (x_normalized.min(), x_normalized.max()))

        # Split the data into training and testing sets
        print("Splitting the data")
        x_train, x_test, y_train, y_test = train_test_split(x_normalized, y, test_size=0.2, random_state=42)

        with tf.device('CPU'):
            del x_normalized

        # Reshape the data
        x_train = x_train.reshape(x_train.shape + (1,))
        x_test = x_test.reshape(x_test.shape + (1,))

        print(f"Dataset shapes:\n"
              f"-> x_train: {x_train.shape} | y_train: {y_train.shape}\n"
              f"-> x_test:  {x_test.shape}  | y_test:  {y_test.shape}")

        y_train = y_train.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)

        # Train the model
        print("Start of training")
        model.fit(x_train, y_train, epochs=20, batch_size=1, validation_data=(x_test, y_test), verbose=2, workers=-1)

        del x_train

        # Evaluate the model
        print("Evaluating model")
        test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2, workers=-1)
        print(f"Test Accuracy: {test_acc}")

        del x_test

        j += 1

    model.save(filepath='/home/tcsn39/adni/adni_cnn_model_full.keras')


def main():
    print(tf.config.get_visible_devices())

    trainnn_fully_partial(path1="/home/tcsn39/adni/Datenneu/AD_coronal_256_part1/*.npy",
        path2="/home/tcsn39/adni/Datenneu/CN_coronal_256_part1/*.npy",
        path3="/home/tcsn39/adni/Datenneu/AD_coronal_256_part2/*.npy",
        path4="/home/tcsn39/adni/Datenneu/CN_coronal_256_part2/*.npy",
        path5="/home/tcsn39/adni/Datenneu/AD_sagittal_256_part1/*.npy",
        path6="/home/tcsn39/adni/Datenneu/CN_sagittal_256_part1/*.npy",
        path7="/home/tcsn39/adni/Datenneu/AD_sagittal_256_part2/*.npy",
        path8="/home/tcsn39/adni/Datenneu/CN_sagittal_256_part2/*.npy",
        path9="/home/tcsn39/adni/Datenneu/AD_axial_256_part1/*.npy",
        path10="/home/tcsn39/adni/Datenneu/CN_axial_256_part1/*.npy",
        path11="/home/tcsn39/adni/Datenneu/AD_axial_256_part2/*.npy",
        path12="/home/tcsn39/adni/Datenneu/CN_axial_256_part2/*.npy")

    # print('Loading images')
    #
    # (adcoronal, cncoronal, adsagittal, cnsagittal, adaxial, cnaxial) = prepare(
    #     path1="/home/tcsn39/adni/Datenneu/AD_coronal_16/*.npy",
    #     path2="/home/tcsn39/adni/Datenneu/CN_coronal_16/*.npy",
    #     path3="/home/tcsn39/adni/Datenneu/AD_sagittal_16/*.npy",
    #     path4="/home/tcsn39/adni/Datenneu/CN_sagittal_16/*.npy",
    #     path5="/home/tcsn39/adni/Datenneu/AD_axial_16/*.npy",
    #     path6="/home/tcsn39/adni/Datenneu/CN_axial_16/*.npy")
    #
    # print('Loaded images')
    #
    # trainnn_fullysupervised(adcoronal=adcoronal, cncoronal=cncoronal, adsagittal=adsagittal, cnsagittal=cnsagittal,
    #                         adaxial=adaxial, cnaxial=cnaxial)


if __name__ == "__main__":
    main()
