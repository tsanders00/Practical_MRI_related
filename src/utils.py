"""

"""
import glob
import os
import dicom2nifti
import nibabel as nib
import numpy
import numpy as np
from PIL import Image
from dipy.segment.tissue import TissueClassifierHMRF
import tensorflow as tf


def convdcmtonii(dicom_directory, output_folder):
    dicom2nifti.convert_directory(dicom_directory, output_folder, compression=True, reorient=True)


def remove_black_slices(data):

    index = []

    # Find slices that are not completely black
    for i in range(data.shape[2]):

        array2d = data[:, :, i]

        if np.all(array2d == 0):
            index.append(i)
            continue

    data = np.delete(data, index, axis=0)

    return data

def convert_nii_to_npy(input_dir, output_dir):

    nii_files = glob.glob(input_dir)
    npy_arrays = []
    # Loop through .nii files found
    for nii_path in nii_files:
        try:
            # Load the .nii file using nibabel
            nii_image = nib.load(nii_path)

            # Get data array from the .nii file
            nii_data = nii_image.get_fdata()

            # Remove completely black slices
            # nii_data = remove_black_slices(nii_data)
            volume_data = nii_data

            # Normalize the volume data
            volume_data = (volume_data - np.min(volume_data)) / (np.max(volume_data) - np.min(volume_data))
            volume_data = (volume_data * 255).astype(np.uint8)  # Convert to uint8

            output_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(nii_path))[0]}.npy")

            numpy.save(file=output_file, arr=volume_data)
            print('Saved cube')
            npy_arrays.append(volume_data)

        except Exception as e:
            print(f"Error processing {os.path.basename(nii_path)}: {str(e)}")

    return npy_arrays

def convert_nii_to_single_tiff(input_dir, output_dir):

    # Find all .nii files in the input directory
    nii_files = glob.glob(input_dir)

    # Loop through .nii files found
    for nii_path in nii_files:
        try:
            # Load the .nii file using nibabel
            nii_image = nib.load(nii_path)

            # Get data array from the .nii file
            nii_data = nii_image.get_fdata()

            # Remove completely black slices
            nii_data = remove_black_slices(nii_data)
            volume_data = nii_data

            # Normalize the volume data
            volume_data = (volume_data - np.min(volume_data)) / (np.max(volume_data) - np.min(volume_data))
            volume_data = (volume_data * 255).astype(np.uint8)  # Convert to uint8

            # Reshape the volume to combine all slices into a single 2D array
            combined_2d_slices = np.concatenate(np.rollaxis(volume_data, axis=-1), axis=1)

            # Convert the combined 2D array into a PIL Image
            pil_image = Image.fromarray(combined_2d_slices)

            # Construct the output file path and name
            output_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(nii_path))[0]}.tiff")

            # Save the PIL Image as a TIFF file
            pil_image.save(output_file)
            # print(f"Converted {os.path.basename(nii_path)} to {output_file}")

        except Exception as e:
            print(f"Error processing {os.path.basename(nii_path)}: {str(e)}")


def convert_nii_to_multi_page_tiff(input_dir, output_dir):

    # Find all .nii files in the input directory
    nii_files = glob.glob(input_dir)

    # Loop through .nii files found
    for nii_path in nii_files:
        try:
            # Load the .nii file using nibabel
            nii_image = nib.load(nii_path)

            # Get data array from the .nii file
            nii_data = nii_image.get_fdata()

            # Normalize the volume data
            nii_data = (nii_data - np.min(nii_data)) / (np.max(nii_data) - np.min(nii_data))
            nii_data = (nii_data * 255).astype(np.uint8)  # Convert to uint8

            # Loop through each slice and save as a TIFF image
            for i in range(nii_data.shape[-1]):
                slice_data = nii_data[:, :, i]
                # Convert the slice to a PIL Image
                pil_image = Image.fromarray(slice_data)

                # Construct the output file path and name
                output_file = os.path.join(output_dir,
                                           f"{os.path.splitext(os.path.basename(nii_path))[0]}_{i + 1}.tiff")

                # Save the PIL Image as a TIFF file
                pil_image.save(output_file)
                # print(f"Converted slice {i + 1} of {os.path.basename(nii_path)} to {output_file}")

        except Exception as e:
            print(f"Error processing {os.path.basename(nii_path)}: {str(e)}")


def loadnii(directory):
    """
    function to load .nii files and import the actual image and the data
    :param directory: place of your .nii files
    :return: images and imagedata
    """
    nii_files = glob.glob(directory)
    images = []
    imagedata = []
    for file in nii_files:
        image = nib.load(file)
        imagedata.append(image.get_fdata())
        images.append(np.array(image.dataobj))

    return images, imagedata


def load_tiff_images(directory):
    loaded_images = []
    tiff_files = glob.glob(directory)

    # Loop through files in the directory
    for file_name in tiff_files:
        try:
            # Open the TIFF file using PIL
            img = Image.open(file_name)
            loaded_images.append(img)
            print(f"Loaded {file_name}")
        except Exception as e:
            print(f"Error loading {file_name}: {str(e)}")

    return loaded_images

def load_npy(directory):
    images = []
    npy_files = glob.glob(directory)

    for file in npy_files:
        array = np.load(file)
        images.append(array)

    return images


def tissue_classification(images):
    """
    tissue is seperated into corticospinal fluid, white and gray matter (nlasses=3)
    beta is smoothness factor of segmentation (value should be between 0 and 0.5)
    """
    hmrf = TissueClassifierHMRF()
    nclasses = 3
    beta = 0.1
    tissue_classified_images = []

    for image in images:
        initial_segmentation, final_segmentation, pve = hmrf.classify(image, nclasses, beta, max_iter=10)
        tissue_classified_images.append(final_segmentation)

    return tissue_classified_images


def pad_images(list):
    """
    function to pad images to the same size
    :param images: images you want to pad
    :return: padded images
    """
    max_shape = np.max([arr.shape for arr in list], axis=0)
    padded_images = []

    for arr in list:
        pad_width = [(0, max_dim - cur_dim) for max_dim, cur_dim in zip(max_shape, arr.shape)]
        padded_arr = np.pad(arr, pad_width, mode='minimum')
        padded_images.append(padded_arr)

    return padded_images


def remove_zero_slices(list_of_arrays):
    """
    function to remove slices containing only zeros from 3D NumPy arrays
    :param list_of_arrays: list containing your ndarrays
    :return: list of new ndarrys without slices containing all zeros
    """
    array_sliced = []
    index = []

    for array3d in list_of_arrays:
        for i in range(array3d.shape[2]):

            array2d = array3d[:, :, i]

            if np.all(array2d == 0):
                index.append(i)
                continue

        array3d = np.delete(array3d, index, axis=0)
        array_sliced.append(array3d)

    return array_sliced


def get_dataset(train_images, train_labels, batch_size, buffer_size):
    train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    train_dataset = train_dataset.shuffle(buffer_size=buffer_size, seed=42).batch(batch_size)

    return train_dataset


def train_step(images, labels, model, optimizer):

    with tf.GradientTape() as tape:
        # Forward pass
        predictions = model(images)
        # Calculate loss
        loss = tf.losses.categorical_crossentropy(labels, predictions)
    # Calculate gradients
    gradients = tape.gradient(loss, model.trainable_variables)
    # Apply gradients
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss


def train_model(epochs, train_dataset, model):

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        total_loss = 0
        for batch_images, batch_labels in train_dataset:
            # Perform one training step
            batch_loss = train_step(batch_images, batch_labels, model=model)
            total_loss += batch_loss
        avg_loss = total_loss / len(train_dataset)
        print(f"Average Loss: {avg_loss.numpy():.4f}")


def generate_cubes_from_3d_arrays(arrays_list, cube_x_size, cube_y_size, cube_z_size, output_dir):
    for idx, array in enumerate(arrays_list):
        # Get the shape of the current 3D array
        array_shape = array.shape

        # Determine the maximum dimensions (X, Y, Z) of the 3D array
        max_x, max_y, max_z = array_shape

        print(f"Processing Array {idx + 1}/{len(arrays_list)}")
        # print(f"Max Dimensions: ({max_x}, {max_y}, {max_z})")
        # print(f"Cube Sizes: ({cube_x_size}, {cube_y_size}, {cube_z_size})")

        # Generate cubes within the array
        for x_idx in range(0, max_x, cube_x_size):
            for y_idx in range(0, max_y, cube_y_size):
                for z_idx in range(0, max_z, cube_z_size):
                    # Define the ranges for the current cube
                    x_end = min(x_idx + cube_x_size, max_x)
                    y_end = min(y_idx + cube_y_size, max_y)
                    z_end = min(z_idx + cube_z_size, max_z)

                    # Extract the cube from the 3D array
                    cube = array[x_idx:x_end, y_idx:y_end, z_idx:z_end]

                    # Save or process the cube as needed
                    # Example: Saving as a .npy file
                    output_file = os.path.join(output_dir, f"cube_{idx + 1}_{x_idx}_{y_idx}_{z_idx}.npy")
                    np.save(file=output_file, arr=cube)

                    # print(f"Saved Cube {x_idx}_{y_idx}_{z_idx} of Array {idx + 1}")


def generate_and_filter_pseudo_labels_with_conf_score(model, dataset):
    """

    :param model:
    :param dataset:
    :return:
    """

    # Make predictions for the entire dataset
    predictions = model.predict(dataset)

    # Apply softmax to get probabilities for all predictions
    prediction_probs = tf.nn.softmax(predictions)

    # Calculate confidence scores for each prediction
    confidence_scores = tf.reduce_max(prediction_probs, axis=1)

    # Filter predictions based on the confidence score threshold (0.8)
    high_confidence_indices = tf.where(confidence_scores > 0.8)[:, 0]
    high_confidence_predictions = tf.gather(dataset, high_confidence_indices)
    high_confidence_pictures = tf.gather(dataset, high_confidence_indices)
    high_confidence_predictions = np.array(high_confidence_predictions)
    high_confidence_pictures = np.array(high_confidence_pictures)

    return high_confidence_pictures, high_confidence_predictions



def generate_and_filter_pseudo_labels_with_entropy(model, x_unlabelled):
    """

    :param model:
    :param x_unlabelled:
    :return:
    """
    pseudo_labels = []
    pictures = []
    for data in x_unlabelled:
        data = data.reshape(1, 32, 32, 32, 1)
        prediction = model(data)
        prediction_probability = tf.nn.softmax(prediction)
        entropy = tf.reduce_sum(-prediction_probability * tf.math.log(prediction_probability), axis=-1)
        if entropy <= 0.2:
            pictures.append(data)
            pseudo_labels.append(prediction)

    return pictures, pseudo_labels
