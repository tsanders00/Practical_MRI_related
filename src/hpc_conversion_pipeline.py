import numpy as np
import glob
import os
import nibabel as nib

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

            np.save(file=output_file, arr=volume_data)
            print(f"Saved cube")
            npy_arrays.append(volume_data)

        except Exception as e:
            print(f"Error processing {os.path.basename(nii_path)}: {str(e)}")

    return npy_arrays

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


def main(input_directory, output_directory):
    arr = convert_nii_to_npy(input_dir=input_directory, output_dir=output_directory)


if __name__ == "__main__":
    print("Starting HPC Conversion Pipeline")
    main(input_directory="/home/tcsn39/adni/Datenneu/ADNI3_CN_Sagittal_T1_nii_processed/*.nii",
         output_directory="/home/tcsn39/adni/Datenneu/CN_sagittal_256")
    print("Done")