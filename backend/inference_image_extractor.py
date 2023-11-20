import sys
import os
import re
import json 
import os.path 

import pandas as pd
from tqdm.notebook import tqdm
from datetime import datetime 

import numpy as np 
import cv2   

# intermediate solution
from template_matcher import find_template_match

df_main_board = pd.read_csv('../../data/board_info_csv/processed/board_data_2F3.csv')

output_folder = '/home/nick-kuijpers/Documents/Railnova/Python/backend/all_components/2F3/'

# Define the template matching method
def get_matching_method():
    methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
               'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
    return eval(methods[1])

# Load an image and a pattern for template matching
def load_images(image_path, pattern_path):
    image = cv2.imread(image_path, 1)
    pattern_check = cv2.imread(pattern_path, 1)
    pattern_check_gray = cv2.cvtColor(pattern_check, cv2.COLOR_BGR2GRAY)
    return image, pattern_check_gray

# Perform template matching and return the matched results
def perform_template_matching(image, pattern_check_gray):
    method = get_matching_method()

    gray = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
    res = cv2.matchTemplate(gray, pattern_check_gray, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    # Centering the matched result on component and scaling it to mm
    matched_result = [(max_loc[0] + (pattern_check_gray.shape[1]/2)), (max_loc[1] + (pattern_check_gray.shape[0]/2))]
    
    # print(f"Matched Result: {matched_result}")    
    # cv2.rectangle(image, max_loc, (max_loc[0] + pattern_check_gray.shape[1], max_loc[1] + pattern_check_gray.shape[0]), (0,0,0), 2, 8, 0 )
    # cv2.imwrite('all_images_with_match/image_matched_{}.jpg'.format(board_id), image)

    return matched_result

# Fit the matched results to the csv data
def fit_results(matched_result_comp1, matched_result_comp2, matched_result_comp3, comp1, comp2, comp3, df_main_board): 
    # Coord of csv
    component_comp1 = df_main_board[df_main_board["Designator"]==comp1]
    x_comp1 = component_comp1['Ref-X(mm)'].values[0]
    y_comp1 = component_comp1['Ref-Y(mm)'].values[0] 
    
    if comp2 == "J703_J702":
        component_comp2 = df_main_board[df_main_board["Designator"]=="J703_1"]
        x_comp2 = component_comp2['Ref-X(mm)'].values[0] - 8
        y_comp2 = component_comp2['Ref-Y(mm)'].values[0] 
    else: 
        component_comp2 = df_main_board[df_main_board["Designator"]==comp2]
        x_comp2 = component_comp2['Ref-X(mm)'].values[0]
        y_comp2 = component_comp2['Ref-Y(mm)'].values[0] 

    component_comp3 = df_main_board[df_main_board["Designator"]==comp3]
    x_comp3 = component_comp3['Ref-X(mm)'].values[0]
    y_comp3 = component_comp3['Ref-Y(mm)'].values[0] 
    
    csv_points = np.array([[x_comp1, y_comp1], [x_comp2, y_comp2], [x_comp3, y_comp3]], dtype=np.float32)

    # Coord of template
    template_points = np.array([matched_result_comp1, matched_result_comp2, matched_result_comp3], dtype=np.float32)

    # Calculate transformation matrix
    transformation_matrix, _ = cv2.estimateAffine2D(csv_points, template_points)
    
    # Two things to do:
        # 1. Change J703_J702 to other less "variable component"
        # 2. Change the transformation matrix to a more robust one 

    # print(f'Transformation Matrix: {transformation_matrix}')

    return transformation_matrix

def process_components(image, components_to_process, df_main_board, transformation_matrix, board_id, orientation):
    components_processed = []
    for component in components_to_process:
        # Get the component's X and Y coordinates from the CSV file
        component_data = df_main_board[df_main_board["Designator"] == component]
        x = component_data['Ref-X(mm)'].values[0]
        y = component_data['Ref-Y(mm)'].values[0]

        new_point = np.array([[x, y]], dtype=np.float32)
        new_camera_point = cv2.transform(new_point.reshape(1, -1, 2), transformation_matrix).squeeze()

        # print(f"New Camera Point: {new_camera_point}")

        # Load the pattern image
        modified_comp_string = component.replace("_1", "")
        pattern_path = f'../../data/template_images/area_extraction_templates/{modified_comp_string}.jpg'
        pattern_check_component = cv2.imread(pattern_path, 1)

        Width = int(pattern_check_component.shape[1] / 2)
        Height = int(pattern_check_component.shape[0] / 2)

        # cv2.rectangle(image, (int(new_camera_point[0]) - Width - 500, int(new_camera_point[1]) - Height - 500),
        #               (int(new_camera_point[0]) + Width + 500, int(new_camera_point[1]) + Height + 500), (255, 0, 0),
        #               2, 8, 0)

        # With better transformation matrix, I can remove the 500 and 400 and be more precise

        # Define the region of interest (ROI) coordinates
        roi_top_left = (int(new_camera_point[0]) - Width - 400, int(new_camera_point[1]) - Height - 400)
        roi_bottom_right = (int(new_camera_point[0]) + Width + 400, int(new_camera_point[1]) + Height + 400)

        # Check if ROI is within image bounds
        if (0 <= roi_top_left[0] < image.shape[1] and 0 <= roi_top_left[1] < image.shape[0] and
                0 <= roi_bottom_right[0] < image.shape[1] and 0 <= roi_bottom_right[1] < image.shape[0]):

            # Extract the region of interest (ROI) corresponding to the rectangle
            roi = image[roi_top_left[1]:roi_bottom_right[1], roi_top_left[0]:roi_bottom_right[0]]

            # Create a unique filename for each component
            filename = f'{output_folder}/image_smaller_2F3_{modified_comp_string}_{orientation}_{board_id}.jpg'

            # Save the ROI with the rectangle
            cv2.imwrite(filename, roi)
            print("Saved smaller not cropped tottally image to {}".format(filename))
            components_processed.append([modified_comp_string, filename, roi_top_left, roi_bottom_right])
        else:
            print(f"Warning: ROI for component {component} is outside image bounds.")

    return components_processed


# Process the matched results and create component rectangles
def process_matched_results(image, matched_results_comp1, matched_results_comp2, matched_results_comp3, df_main_board, orientation, board_id, PP1): 
    match orientation:
        case 'Top':
            transformation_matrix = fit_results(matched_results_comp1, matched_results_comp2, matched_results_comp3, "J2_1", "J704_1", "BT100_1", df_main_board) 
            if PP1: 
                # U904U5_1 va avoir la taille de _1 + ... + U5 et au template matching j'en extrait 5
                # U911Ux petit soucis car composants identique entre Ux
                components_to_process = ["J2_1", "U701_1", "U911U5_1","U904U3_1"]
            else:
                components_to_process = ["U500_1"]
        case 'Bottom':
            transformation_matrix = fit_results(matched_results_comp1, matched_results_comp2, matched_results_comp3, "U604_1", "J703_J702", "U7_1", df_main_board) 
            if PP1: 
                components_to_process = ["U600_1", "U608_1", "Q102_1", "U100_1"]
            else:
                # AL va avoir en template la taille de AL + AH, au template matching je dois extraire les 2
                components_to_process = ["U901AL_1"] 
            
        case _:
            print("Unknown orientation.")

    # Process components and draw rectangles
    components_processed = process_components(image, components_to_process, df_main_board, transformation_matrix, board_id, orientation)

    # Save the image with the rectangles
    # cv2.imwrite(f'all_images_with_match/image_matched_2F2_{board_id}_{orientation}.jpg', image)
    return components_processed

# Main function for processing images
def process_images(image_path, df_main_board): 
    pattern = r'(?P<orientation>Top|Bottom)_(?P<board_id>[A-Z0-9]+)'
    # extract orientation and board_id from image_path
    match = re.search(pattern, image_path)
    if match:
        orientation = match.group('orientation')
        board_id = match.group('board_id')
        PP1 = False
        
        match orientation:
            case 'Top':    
                image, pattern_check_gray_J2 = load_images(image_path, '../../data/template_images/area_extraction_templates/J2.jpg') 
                _, pattern_check_gray_J704 = load_images(image_path, '../../data/template_images/area_extraction_templates/J704.jpg')
                _, pattern_check_gray_BT100 = load_images(image_path, '../../data/template_images/area_extraction_templates/BT100.jpg')
                matched_result_J2 = perform_template_matching(image, pattern_check_gray_J2)
                matched_result_J704 = perform_template_matching(image, pattern_check_gray_J704)
                matched_result_BT100 = perform_template_matching(image, pattern_check_gray_BT100)
                if (matched_result_J2[1] < 4500): 
                    PP1 = True
                components_processed = process_matched_results(image, matched_result_J2, matched_result_J704, matched_result_BT100, df_main_board, orientation, board_id, PP1)   
            case 'Bottom':
                image, pattern_check_gray_U604 = load_images(image_path, '../../data/template_images/area_extraction_templates/U604.jpg') 
                _, pattern_check_gray_J703_J702 = load_images(image_path, '../../data/template_images/area_extraction_templates/J703_J702.jpg')
                _, pattern_check_gray_U7 = load_images(image_path, '../../data/template_images/area_extraction_templates/U7.jpg')
                matched_result_U604 = perform_template_matching(image, pattern_check_gray_U604)
                matched_result_J703_J702 = perform_template_matching(image, pattern_check_gray_J703_J702)
                matched_result_U7 = perform_template_matching(image, pattern_check_gray_U7)

                if (matched_result_U604[1] > 4000):
                    PP1 = True
                components_processed = process_matched_results(image, matched_result_U604, matched_result_J703_J702, matched_result_U7, df_main_board, orientation, board_id, PP1) 

            case _:
                print("Unknown orientation.")
    else:
        print(f"No match found for pattern {pattern} in image_path {image_path}")

    return components_processed





# Main function for processing images
def main(image_path, df_main_board):
    cropped_images = []
    component_processed = process_images(image_path, df_main_board)
    for component in component_processed: 
        input_image_path = component[1]
        print(component)
        template_image_path = f'../../data/template_images/matching_templates/{component[0]}.jpg'
        to_save_im_path = f'{output_folder}/cropped_images/{component[0]}.jpg'
        find_template_match(input_image_path, template_image_path, to_save_im_path)
        print("Saved cropped image to {}".format(to_save_im_path))
        cropped_images.append([to_save_im_path, component[0]])
    
    return cropped_images

if __name__ == '__main__':
    image_path = '../../data/data_default_processed/2.F.3_G015_Top_PP1.jpg'
    main(image_path, df_main_board)
