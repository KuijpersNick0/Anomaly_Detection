import cv2
import numpy as np
import os 

def find_match(image, template, threshold=0.9):
    # Apply normalized cross-correlation between the ROI and the template
    result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    cv2.normalize( result, result, 0, 1, cv2.NORM_MINMAX, -1 )
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    matchLoc = max_loc    
    # img_display = image.copy()

    if max_val >= threshold:
        # cv2.rectangle(img_display, matchLoc, (matchLoc[0] + template.shape[0], matchLoc[1] + template.shape[1]), (0,0,0), 2, 8, 0 ) 
        img_component = image[matchLoc[1]:matchLoc[1] + template.shape[0], matchLoc[0]:matchLoc[0] + template.shape[1]] 
        cv2.namedWindow("Display window", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Display result", cv2.WINDOW_NORMAL)

        cv2.imshow("Display window", image)
        cv2.waitKey(0)
        cv2.imshow("Display result", img_component)
        cv2.waitKey(0)
        print("Components matching found")
        return img_component
    else:
        return "Components matching not found"

def image_extraction(image_folder, destination_folder):
    # Load template images
    template_folder = '/templates_images'
    template_files = os.listdir(template_folder)
    # List all files in the image folder
    image_files = os.listdir(image_folder)

    for template_image in template_files:
        template_image_path = os.path.join(template_folder, template_image)
        template = cv2.imread(template_image_path, 0)
        for image_file in image_files:
            image_path = os.path.join(image_folder, image_file)
            image = cv2.imread(image_path, 0)
            img_component = find_match(image, template)
            if img_component != "Components matching not found":
                # Output path should be linked to the template image that was used but also some information of image_file 
                output_path = os.path.join(destination_folder, image_file)
                cv2.imwrite(output_path , img_component) 
                print(f"Saved: {image_file} -> {output_path}")


def main(image_folder, destination_folder):
    image_extraction(image_folder, destination_folder)
    return None

if __name__ == "__main__": 
    image_folder = '../data/data_default'
    destination_folder = '../data/data_default_proccessed'
    main(image_folder, destination_folder)