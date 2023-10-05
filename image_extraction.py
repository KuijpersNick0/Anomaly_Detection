import cv2
import numpy as np
import os 

import matplotlib.pyplot as plt
import json

# List of top components to be used for the top images
top_components = ["2F3_U904Ux", "2F3_U911Ux", "U500", "J2", "U701"]

# Histogram template matching saving data
file_path = '../data/histogram_template_matching.json'

def find_match(image, template, threshold=0.69):
    # Apply normalized cross-correlation between the ROI and the template
    result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    # cv2.normalize( result, result, 0, 1, cv2.NORM_MINMAX, -1 )
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    matchLoc = max_loc    
    # img_display = image.copy()

    print(max_val)
    if max_val >= threshold:
        # cv2.rectangle(img_display, matchLoc, (matchLoc[0] + template.shape[0], matchLoc[1] + template.shape[1]), (0,0,0), 2, 8, 0 ) 
        img_component = image[matchLoc[1]:matchLoc[1] + template.shape[0], matchLoc[0]:matchLoc[0] + template.shape[1]] 
        # cv2.namedWindow("Display window", cv2.WINDOW_NORMAL)
        # cv2.namedWindow("Display result", cv2.WINDOW_NORMAL)

        # cv2.imshow("Display window", image)
        # cv2.waitKey(0)
        # cv2.imshow("Display result", img_component)
        # cv2.waitKey(0)
        print("Components matching found")
        return [img_component, max_val]
    else:
        return ["No", max_val]

def is_top_image(image_path):
    return True if "top" in image_path.lower() else False

def is_top_component(template_image_path):
    return True if template_image_path in top_components else False

def save_image(img_component, template_image_path, image_path, image_file, destination_folder):
    if img_component[0] != "No":
        # Output path should be linked to the template image that was used but also some information of image_file 
        image_info = os.path.splitext(os.path.basename(template_image_path))[0] + "_" + os.path.basename(image_path)
        output_path = os.path.join(destination_folder, image_info)
        cv2.imwrite(output_path , img_component[0]) 
        print(f"Saved: {image_file} -> {output_path}") 
        save_histogram(True, img_component[1])
    else:
        print(f"Components matching not found for {image_file} with component {os.path.splitext(os.path.basename(template_image_path))[0]}")
        save_histogram(False, img_component[1])

def save_histogram(flag, max_val): 
    # Function to save the histogram data
    data = {
        "saved_flag": flag,
        "threshold": max_val
    }
    with open(file_path, 'w') as file:
        json.dump(data, file)
        file.write('\n')  # Add a newline to separate entries

def filter_2E4(image_file):
    # If 2.E.4 is in the image path, then it should not be used for now
    # It needs a distinct template from 2F2 and 2F3
    return True if "2.e.4" in image_file.lower() else False

def image_extraction(image_folder, destination_folder):
    # Load template images
    template_folder = '../data/template_images'
    template_files = os.listdir(template_folder)
    # List all files in the image folder
    image_files = os.listdir(image_folder)

    for index, image_file in enumerate(image_files):
        print(f"Processing image {index + 1}/{len(image_files)}: {image_file}")
        # Filter out 2.E.4 images
        if filter_2E4(image_file):
            continue
        for template_image in template_files:
            template_image_path = os.path.join(template_folder, template_image)
            # Should make a check here for the top and bottom components
            if is_top_image(image_file):
                if is_top_component(os.path.splitext(os.path.basename(template_image_path))[0]):
                    image_path = os.path.join(image_folder, image_file)
                    image = cv2.imread(image_path, 0)
                    template = cv2.imread(template_image_path, 0)
                    img_component = find_match(image, template)
                    save_image(img_component, template_image_path, image_path, image_file, destination_folder)
                else:
                    print("Top image but not top component")
            else:
                if not is_top_component(os.path.splitext(os.path.basename(template_image_path))[0]):
                    image_path = os.path.join(image_folder, image_file)
                    image = cv2.imread(image_path, 0)
                    template = cv2.imread(template_image_path, 0)
                    img_component = find_match(image, template)
                    save_image(img_component, template_image_path, image_path, image_file, destination_folder)
                else:
                    print("Bottom image but not bottom component")

def main(image_folder, destination_folder):
    image_extraction(image_folder, destination_folder)
    return None

if __name__ == "__main__": 
    image_folder = '../data/data_default_processed'
    destination_folder = '../data/data_extracted'
    main(image_folder, destination_folder)