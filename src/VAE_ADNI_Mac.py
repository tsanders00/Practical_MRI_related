"""
VAE implementation for mri images
inspiration from https://github.com/rcantini/CNN-VAE-MNIST
this script was build for usage on my personal laptop to test the functionality
the script that was used to actually generate results is 'VAE_ADNI.py'
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras import backend as K
from keras import layers
from keras.models import Model
import tensorflow as tf
from sklearn.model_selection import train_test_split
import glob

def plot_history(history, fname):
    history = pd.DataFrame(history)
    history.plot(figsize=(16, 10))
    plt.grid(True)
    plt.gca().set_ylim(0, 1000)
    plt.xlabel("epoch")
    plt.ylabel("metrics")
    plt.title("metrics over epochs", loc="center")
    plt.savefig(f"{fname}.pdf")


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


def separate_and_stack_slices(array_list):
    # Check if the input list is not empty
    if not array_list:
        raise ValueError("Input list is empty")

    # Check if all arrays in the list have the same shape
    array_shape = array_list[0].shape
    if any(arr.shape != array_shape for arr in array_list):
        raise ValueError("All arrays in the list must have the same shape")

    # Separate and stack slices
    stacked_array = np.stack([arr[i, :, :] for arr in array_list for i in range(array_shape[0])], axis=0)

    return stacked_array

def filter_arrays_for_black_cube(arrays: list):
    result = []
    for arr in arrays:
        if np.max(arr) > 5:
            result.append(arr)
    return result


def sampling(args):
    mu, sigma = args
    batch = K.shape(mu)[0]
    dim = K.int_shape(mu)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return mu + K.exp(0.5 * sigma) * epsilon


def create_compiled_model(latent_dim, image_shape):
    # VAE model = encoder + decoder

    # build encoder model
    inputs = layers.Input(shape=image_shape, name='encoder_input')
    x = inputs
    x = layers.Conv3D(32, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.MaxPooling3D(pool_size=(2, 2, 2))(x)
    x = layers.Conv3D(64, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.MaxPooling3D(pool_size=(2, 2, 2))(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation='relu')(x)
    mu = layers.Dense(latent_dim, name='mu')(x)
    sigma = layers.Dense(latent_dim, name='sigma')(x)

    # use reparameterization trick
    z = layers.Lambda(sampling, name='z')([mu, sigma])
    # instantiate encoder model
    encoder = Model(inputs, [mu, sigma, z], name='encoder')
    encoder.summary()

    # build decoder model
    latent_inputs = layers.Input(shape=(latent_dim,), name='z_sampling')
    x = layers.Dense(64, activation='relu')(latent_inputs)
    x = layers.Dense(8 * 8 * 8 * 64, activation='relu')(x)
    x = layers.Reshape((8, 8, 8, 64))(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Conv3DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
    outputs = layers.Conv3DTranspose(1, 3, activation="sigmoid", strides=2, padding="same")(x)
    outputs = layers.Reshape(image_shape)(outputs)

    # instantiate decoder model
    decoder = Model(latent_inputs, outputs, name='decoder')
    decoder.summary()

    # instantiate VAE model
    outputs = decoder(encoder(inputs)[2])
    vae = Model(inputs, outputs, name='CVAE')
    vae.summary()
    models = (encoder, decoder)
    # VAE loss = rec_loss + kl_loss
    reconstruction_loss = tf.reduce_sum(tf.square(inputs - outputs), axis=[1, 2, 3])
    kl_loss = 1 + sigma - K.square(mu) - K.exp(sigma)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam')
    return vae, models

def evaluate_model(vae, models, x_test):
    encoder, decoder = models
    # Reconstruction accuracy
    reconstruction_loss = []
    latent_representations = []
    for batch in x_test:
        batch_reshaped = np.expand_dims(batch, axis=0)
        batch_reshaped = np.expand_dims(batch_reshaped, axis=-1)
        mu, sigma, z = encoder(batch_reshaped)
        x_recon = decoder(z)
        rec_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(batch, x_recon))
        reconstruction_loss.append(rec_loss.numpy())
        latent_representations.append(mu.numpy())

    avg_rec_loss = np.mean(reconstruction_loss)
    print(f'Average Reconstruction Loss on Test Set: {avg_rec_loss}')

    # Generating samples from the latent space
    num_samples = 3
    x_test_npy = np.stack(list(x_test))

    # Plotting original and reconstructed images
    for i in range(num_samples):
        plt.figure(figsize=(40, 16))  # Adjust the overall figure size

        x_test_reshaped = np.expand_dims(x_test_npy[np.random.randint(0, len(x_test_npy))], axis=0)
        generated_image = vae.predict(x_test_reshaped, verbose=2)

        for j in range(x_test_npy.shape[-2]):
            # Original Images
            plt.subplot(2, x_test_npy.shape[-2], j + 1)
            plt.imshow(x_test_npy[i, :, :, j], cmap='gray')  # Assuming you want to visualize the first depth slice
            plt.axis('off')

        for j in range(x_test_npy.shape[-2]):
            # Reconstructed Images
            plt.subplot(2, x_test_npy.shape[-2], x_test_npy.shape[-2] + j + 1)
            gen_image = np.squeeze(generated_image[:, :, :, j])
            plt.imshow(gen_image, cmap='gray')  # Assuming you want to visualize the first depth slice
            plt.axis('off')

        plt.tight_layout()
        plt.show()
        # plt.savefig(fname=f"/home/tcsn39/vae/plots/original_recon_images_{i}.pdf")


def generate_and_test_images_with_cnn(vae, x_test, number_of_pictures, save_path, model_path):
    # load trained model
    cnn_model = tf.keras.models.load_model(model_path)
    cnn_model.summary()
    # generate new images
    vae_images = []
    for i in range(number_of_pictures):
        x_test_reshaped = np.expand_dims(x_test[np.random.randint(0, len(x_test))], axis=0)
        generated_image = vae.predict(x_test_reshaped, verbose=2)
        vae_images.append(generated_image.astype(np.float64))
        np.save(file=f'{save_path}/vae_image{i}.npy', arr=generated_image)
    vae_images = np.array(vae_images)
    vae_images = vae_images.reshape(vae_images.shape[0], 32, 32, 32, 1)
    print(vae_images.shape)
    # create labels for new images
    test_labels = [1] * number_of_pictures
    test_labels = np.array(test_labels)
    print(test_labels.shape)
    print(f'dtype of vae images: {vae_images[0].dtype}')
    # predict new VAE images with cnn
    loss, acc = cnn_model.evaluate(vae_images, test_labels, verbose=2)

    print(f'Accuracy of trained cnn on vae generated images: {acc*100:.2f}%')


if __name__ == '__main__':
    # network parameters
    batch_size = 2
    latent_dim = 50
    epochs = 10
    image_shape = (32, 32, 32, 1)

    print("Loading data")
    images = load_npy(directory='/Users/Torben/Desktop/Bioinformatik_Master/3.Semester'
                                '/Praktikum_Andreas/Daten/AD_coronal_cube/*.npy')
    print(f'Number of images in original dataset and before filtering: {len(images)}')
    images = filter_arrays_for_black_cube(images)
    print(f'Number of images after filtering: {len(images)}')
    images = pad_images(images)

    # Convert data and labels to numpy arrays
    x = np.array(images)
    y = [0] * len(x)

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

    x_train, x_test, y_train, y_test = train_test_split(x_normalized, y, test_size=0.2, random_state=42)

    # Reshape the data
    x_train = x_train.reshape(x_train.shape + (1,))
    x_test = x_test.reshape(x_test.shape + (1,))
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    print(f"Dataset shapes:\n"
          f"-> x_train: {x_train.shape} | y_train: {y_train.shape}\n"
          f"-> x_test:  {x_test.shape}  | y_test:  {y_test.shape}")

    # train the autoencoder
    vae, models = create_compiled_model(latent_dim=latent_dim, image_shape=image_shape)

    history = vae.fit(x_train, x_train,
            epochs=epochs,
            batch_size=batch_size,
            verbose=2,
            workers=-1)

    plot_history(history, fname="../")

    evaluate_model(vae=vae, models=models, x_test=x_test)

    # generate_and_test_images_with_cnn(vae=vae, x_test=x_test, number_of_pictures=10,
    #                                  save_path="/home/tcsn39/vae/VAE_gen_img",
    #                                  model_path="/home/tcsn39/adni/adni_cnn_model.keras")