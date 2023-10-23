import cv2
import numpy as np
import os 
import re
import matplotlib.pyplot as plt
import json
from tqdm.notebook import tqdm
import pandas as pd
 
default_csv_path = '../data/default_excell/Default.csv'
csv_file = pd.read_csv(default_csv_path)

def find_match(image, template, threshold=0.20):
    # Apply normalized cross-correlation between the ROI and the template
    result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    # cv2.normalize( result, result, 0, 1, cv2.NORM_MINMAX, -1 )
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    matchLoc = max_loc     

    print(max_val)
    if max_val >= threshold:
        # cv2.rectangle(img_display, matchLoc, (matchLoc[0] + template.shape[0], matchLoc[1] + template.shape[1]), (0,0,0), 2, 8, 0 ) 
        img_component = image[matchLoc[1]:matchLoc[1] + template.shape[0], matchLoc[0]:matchLoc[0] + template.shape[1]] 
        # cv2.namedWindow("Display result", cv2.WINDOW_NORMAL)
        # cv2.imshow("Display window", image)
        # cv2.waitKey(0)
        # cv2.imshow("Display result", img_component)
        # cv2.waitKey(0)
        print("Components matching found")
        return img_component
    else:
        return "Error finding matching"

def find_match_U911(image, template, version, threshold=0.30):
    # Apply normalized cross-correlation between the ROI and the template
    result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    # cv2.normalize( result, result, 0, 1, cv2.NORM_MINMAX, -1 )
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    matchLoc = min_loc     
    min_x = matchLoc[0]

    print("U911U5")
    if version != "2E4":
        if max_val >= threshold:
            for loc in zip(*np.where(result >= threshold)[::-1]):
                if loc[0] < min_x:
                    min_x = loc[0]
                    matchLoc = loc

            # cv2.rectangle(img_display, matchLoc, (matchLoc[0] + template.shape[0], matchLoc[1] + template.shape[1]), (0,0,0), 2, 8, 0 ) 
            img_component = image[matchLoc[1]:matchLoc[1] + template.shape[0], matchLoc[0]:matchLoc[0] + template.shape[1]] 
            # cv2.namedWindow("Display window", cv2.WINDOW_NORMAL)
            # cv2.namedWindow("Display result", cv2.WINDOW_NORMAL)

            # cv2.imshow("Display window", image)
            # cv2.waitKey(0)
            # cv2.imshow("Display result", img_component)
            # cv2.waitKey(0)
            print("Components matching found")
            return img_component
        else:
            return "Error finding matching for U911U5"
    else:
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
            return img_component
        else:
            return "Error finding matching for U911U5 version 2E4"

def find_match_U904(image, template, threshold=0.30):
    # Apply normalized cross-correlation between the ROI and the template
    result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    # cv2.normalize( result, result, 0, 1, cv2.NORM_MINMAX, -1 )
    loc = np.where(result >= threshold)
    rectangles = []
    print("U904U3")
    for pt in zip(*loc[::-1]):
        rectangles.append(image[pt[1]:pt[1]+template.shape[0], pt[0]:pt[0]+template.shape[1]])

    return rectangles

def save_image(new_image, component, version, board_id):
    # Based on CSV file, save the image in the right folder 
    component = str(component)
    version = str(version)
    board_id = 'G' + str(board_id)

    # Check if component and version match in the CSV file
    match = csv_file[(csv_file['Component'] == component) & (csv_file['Version'] == version) & (csv_file['Id'] == board_id)]
    print(match)
    if not match.empty:
        folder = component+'_Def'
    else:
        folder = component
    print(folder + " folder")
    # Save the image in the appropriate folder
    save_path = f'../data/CNN_images/Run1/{folder}/{component}_{version}_{board_id}.jpg'
    cv2.imwrite(save_path, new_image) 
    return "Saved" 

def get_template_image(component):
    # Load template images
    template_folder = '../data/template_images/matching_templates'
    template_files = os.listdir(template_folder)
    for file in template_files:
        if component in file:
            template_path = os.path.join(template_folder, file)
            template_image = cv2.imread(template_path)
            # cv2.namedWindow("Display window", cv2.WINDOW_NORMAL)
            # cv2.imshow("Display window", template_image)
            # cv2.waitKey(0)
            return template_image

def image_extraction(image_path):
    pattern = re.compile(r'(\d+[fFeE]\d+)/.*_(\w+)_(\d+)_(bottom|top)_(\d+)\.jpg$')
    # List all files in the image folder 
    for folder in os.listdir(image_path):
        folder_path = os.path.join(image_path, folder)
        if os.path.isdir(folder_path): 
            for file in os.listdir(folder_path): 
                file_path = os.path.join(folder_path, file)
                if os.path.isfile(file_path):
                    # print(file_path)        
                    match = pattern.search(file_path)
                    if match:
                        version = match.group(1)
                        component = match.group(2) 
                        orientation = match.group(4)
                        board_id = match.group(5)
                        # print(f"Version: {version}, Component: {component}, Orientation: {orientation}, Board id: {board_id}")
                        template = get_template_image(component)
                        image = cv2.imread(file_path)
                        if (component == "U911U5"):
                            # continue
                            # Chop 1 le plus à gauche (4 pins alors qu'2E4 pas 4 pins plus à gauche)
                            new_image = find_match_U911(image, template, version)
                            print(save_image(new_image, component, version, board_id))
                        if (component == "U904U3"):
                            # continue
                            # Chop les 3 et les mettre ds le bon ordre ds le folder
                            new_image_list = find_match_U904(image, template)
                            for i, new_image in enumerate(new_image_list):
                                print(save_image(new_image, component, version, board_id)) 
                        else:
                            new_image = find_match(image, template)
                            print(save_image(new_image, component, version, board_id))
    return None
    


def main(image_folder):
    image_extraction(image_folder)
    return None

if __name__ == "__main__": 
    image_path = '../data/area_extraction_results'
    main(image_path)