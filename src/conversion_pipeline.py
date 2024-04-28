import utils

def main(input_directory, output_directory):
    utils.convert_nii_to_npy(input_dir=input_directory, output_dir=output_directory)


if __name__ == "__main__":
    main(input_directory="/Users/Torben/Desktop/Bioinformatik_Master/3.Semester/Praktikum_Andreas/Daten/ADNI3_AD_Coronal_T1_nii_processed/*.nii",
         output_directory="/Users/Torben/Desktop/Bioinformatik_Master/3.Semester/Praktikum_Andreas/Daten/AD_coronal_256")

    