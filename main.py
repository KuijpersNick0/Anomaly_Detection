import os
import sys

# Define paths
image_folder = 'data/data_defaults'
image_folder_default_processed = 'data/data_default_processed'
# List all files in the image folder
image_files = os.listdir(image_folder)

def execute_image_extraction(image_folder, image_folder_default_processed):
    # Import and execute component_matching.py
    import image_extraction
    image_extraction.main(image_folder, image_folder_default_processed)

def execute_train_CNN(image_path):
    # Import and execute component_matching.py
    import train_CNN
    train_CNN.main(image_path)

if __name__ == "__main__":
    # Camera calibration
    # Not possible with these images

    # Define with true which method to execute
    execute_image_extraction_flag = False 
    execute_train_CNN_flag = False 

    # Determine which scripts to execute based on conditions
    if execute_image_extraction_flag:
        execute_image_extraction(image_folder, image_folder_default_processed)

    # if execute_train_CNN_flag:
    #     execute_train_CNN(image_path)