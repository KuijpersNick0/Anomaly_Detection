import os
import shutil

# Function to rename and move images
def rename_and_move_images(src_dir, dest_dir):
    for root, dirs, files in os.walk(src_dir): 
        # Skip the additional name folder if it exists
        if len(dirs) == 1 and dirs[0] in ['2F2', '2F3', '2E4']:
            continue
        for file in files:
            # if file.startswith("pp (1)") or file.startswith("pp (2)") or file.startswith("PP (1)") or file.startswith("PP (2)"):
            if "pp (1)" in file or "pp (2)" in file or "PP (1)" in file or "PP (2)" in file:
                folder_name = os.path.basename(os.path.dirname(root)) 
                parent_folder_name = os.path.basename(os.path.dirname(os.path.dirname(root)))
                position = "Bottom" if "bottom" in root.lower() else "Top"
                extension = os.path.splitext(file)[-1].lower()
                new_name = f"{parent_folder_name}_{folder_name}_{position}_PP{file.split('(')[-1].split(')')[0]}{extension}"
                src_path = os.path.join(root, file)
                dest_path = os.path.join(dest_dir, new_name)
                shutil.copy(src_path, dest_path)
                print(f"Renamed: {file} -> {new_name}")

# Source and destination directories
src_directory = "/home/nick-kuijpers/Documents/Railnova/data/data_default/New"
dest_directory = "/home/nick-kuijpers/Documents/Railnova/data/data_images_processed"

# Create the destination directory if it doesn't exist
os.makedirs(dest_directory, exist_ok=True)

# Call the function to rename and move the images
rename_and_move_images(src_directory, dest_directory)