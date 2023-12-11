import sys
import os
import re
import json
from pathlib2 import Path
import os.path  
import pandas as pd 
from datetime import datetime 
import numpy as np 
import cv2    
from PIL import Image

# bottom_pp1 = ["U600_1", "U608_1", "Q102_1", "U100_1", "U101_1", "Q400_1", "C401_1", "L400_1","C408_1", "L500_1", "C525_1", "C526_1", "Q204_1", "U4_1"]

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

# Perform template matching and return the approximate result
def perform_broad_template_matching_with_roi(image, pattern_check_gray, roi_top_left, roi_bottom_right):
    method = get_matching_method() 

    gray = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)

    # Create a rectangle to represent the ROI
    roi = gray[roi_top_left[1]:roi_bottom_right[1], roi_top_left[0]:roi_bottom_right[0]] 
    # Perform template matching and return the matched results
    res = cv2.matchTemplate(roi, pattern_check_gray, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    top_left = max_loc
    h, w = pattern_check_gray.shape[:2]
    bottom_right = (top_left[0] + w, top_left[1] + h)

    new_top_left = top_left[0] + roi_top_left[0], top_left[1] + roi_top_left[1]
    new_bottom_right = bottom_right[0] + roi_top_left[0], bottom_right[1] + roi_top_left[1]

    # cv2.rectangle(gray, new_top_left, new_bottom_right, (0,0,0), 2, 8, 0 )
    # cv2.namedWindow("Display window 2 ", cv2.WINDOW_NORMAL)
    # cv2.imshow("Display window 2 ", gray)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return new_top_left, new_bottom_right

def perform_template_matching_with_roi(image, pattern_check_gray, roi_top_left, roi_bottom_right):
    method = get_matching_method() 
    
    # img to gray
    gray = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
    
    # Create a rectangle to represent the ROI
    roi = gray[roi_top_left[1]:roi_bottom_right[1], roi_top_left[0]:roi_bottom_right[0]] 
    # Perform template matching and return the matched results
    res = cv2.matchTemplate(roi, pattern_check_gray, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    top_left = max_loc
    h, w = pattern_check_gray.shape[:2]
    bottom_right = (top_left[0] + w, top_left[1] + h)

    # # Draw a rectangle around the matched area in the original image
    # cv2.rectangle(gray, (top_left[0] + roi_top_left[0], top_left[1] + roi_top_left[1]),
    #             (bottom_right[0] + roi_top_left[0], bottom_right[1] + roi_top_left[1]), (0, 255, 0), 2)

    # Centering the matched result on component and scaling
    matched_result = [(top_left[0] + roi_top_left[0] + (pattern_check_gray.shape[1]/2)), (top_left[1] + roi_top_left[1] + (pattern_check_gray.shape[0]/2))]
    

    # print(f"Matched Result: {(top_left[0] + roi_top_left[0],top_left[1] + roi_top_left[1]), max_val}")    
    # cv2.namedWindow("Display window 2 ", cv2.WINDOW_NORMAL)
    # cv2.imshow("Display window 2 ", gray)
    # cv2.waitKey(0)

    return matched_result

# Fit the matched results to the csv data
def fit_results(matched_result_comp1, matched_result_comp2, matched_result_comp3, comp1, comp2, comp3, df_main_board): 
    # Coord of csv
    component_comp1 = df_main_board[df_main_board["Designator"]==comp1]
    x_comp1 = component_comp1['Ref-X(mm)'].values[0]
    y_comp1 = component_comp1['Ref-Y(mm)'].values[0] 

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
    
    # print(f'Transformation Matrix: {transformation_matrix}')

    return transformation_matrix

def process_components(image, components_to_process, df_main_board, transformation_matrix, board_id, orientation, top_folder): 
    for component in components_to_process: 
        
        #  Get the component's X and Y coordinates from the CSV file
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

        # cv2.rectangle(image, (int(new_camera_point[0]) - Width, int(new_camera_point[1]) - Height),
        #               (int(new_camera_point[0]) + Width, int(new_camera_point[1]) + Height), (255, 0, 0),
        #               5, 8, 0)

        # Define the region of interest (ROI) coordinates
        roi_top_left = (int(new_camera_point[0]) - Width -10, int(new_camera_point[1]) - Height -10)
        roi_bottom_right = (int(new_camera_point[0]) + Width +10, int(new_camera_point[1]) + Height +10)

        # print(roi_top_left, roi_bottom_right)
 
        # Extract the region of interest (ROI) corresponding to the rectangle
        roi = image[roi_top_left[1]:roi_bottom_right[1], roi_top_left[0]:roi_bottom_right[0]]
        
        save_and_check(roi, modified_comp_string, board_id, orientation, top_folder)
    
    # cv2.namedWindow("Display window", cv2.WINDOW_NORMAL)  
    # cv2.imshow("Display window", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return image

# Process the matched results and create component rectangles
def process_matched_results(image, df_main_board, orientation, board_id, PP1, transformation_matrix, top_folder): 
    # Process components and draw rectangles
    components_to_process = []  # Define components_to_process list
    match orientation:
        case 'top':
            if PP1: 
                # U904U5_1 va avoir la taille de _1 + ... + U5 et au template matching j'en extrait 5
                # U911Ux petit soucis car composants identique entre Ux
                components_to_process = ["J2_1", "U701_1", "U911U1_1", "U911U2_1", "U911U3_1", "U911U4_1", "U911U5_1", "U904U1_1", "U904U2_1", "U904U3_1", "U904U4_1", "U904U5_1"]
            else:
                components_to_process = ["U500_1"]
        case 'bottom': 
            if PP1: 
                components_to_process = ["U600_1", "U608_1", "Q102_1", "U100_1"]
            else:
                # AL va avoir en template la taille de AL + AH, au template matching je dois extraire les 2
                components_to_process = ["U901AL_1", "U901AH_1"] 
            
        case _:
            print("Unknown orientation.") 

    image = process_components(image, components_to_process, df_main_board, transformation_matrix, board_id, orientation, top_folder) 


def save_and_check(roi, component, board_id, orientation, top_folder): 
    # Read the defects CSV file
    defects_df = pd.read_csv("/home/nick-kuijpers/Documents/Railnova/data/default_excell/Default_V2.csv")
    
    parts = top_folder.split('.') 
    # Remove empty parts and concatenate them
    top_folder = ''.join(parts)  

    # Check if the component is "bad"
    is_bad = (defects_df["Version"] == top_folder) & (defects_df["ID"] == board_id) & (defects_df["Designator"] == component)

    if is_bad.any():
        # Save the component in the "bad" folder
        folder_name = f"/home/nick-kuijpers/Documents/Railnova/data/CNN_images/Run2/{component}_bad"
    else:
        # Save the component in the normal folder
        folder_name = f"/home/nick-kuijpers/Documents/Railnova/data/CNN_images/Run2/{component}"
    
    # Create the folder if it doesn't exist
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    
    # Save the image in the folder
    cv2.imwrite(os.path.join(folder_name, f"{top_folder}_{component}_{board_id}_{orientation}.jpg"), roi)
    print(f"Saved {component} in {folder_name}")


# Main function for processing images
def process_images(image_path, df_main_board, orientation , clean_g_code, top_folder):  
    
    # bottom pp1 3:4
    # bottom pp2 5:6

    # Probleme 6:7 top U500 out of bound alors que très peu bougé.

    # top pp1 10:11

    PP1 = False 
    
    if 'top' in orientation.lower():
        orientation = 'top'
    else:
        orientation = 'bottom'
    
    board_id = clean_g_code
     
    match orientation:
        case 'top':    
            image, pattern_check_gray_BT100 = load_images(image_path, '../../data/template_images/area_extraction_templates/smaller/BT100.jpg')
            _, pattern_check_gray_J704 = load_images(image_path, '../../data/template_images/area_extraction_templates/smaller/J704.jpg')
            matched_result_BT100 = perform_template_matching_with_roi(image, pattern_check_gray_BT100, (4500, 800), (9000, 5000))
            matched_result_J704 = perform_template_matching_with_roi(image, pattern_check_gray_J704, (100, 1500), (3000, 5000))
            if (matched_result_BT100[1] < 2000): 
                PP1 = True
                _, pattern_check_gray_U701 = load_images(image_path, '../../data/template_images/area_extraction_templates/smaller/U701.jpg') 
                matched_result_U701 = perform_template_matching_with_roi(image, pattern_check_gray_U701, (100, 3000), (4000, 8000))
                
                # Matrice de transfo sur base du matching de 3 composants
                transformation_matrix = fit_results(matched_result_U701, matched_result_J704, matched_result_BT100, "U701_1", "J704_1", "BT100_1", df_main_board) 
                process_matched_results(image, df_main_board, orientation, board_id, PP1, transformation_matrix, top_folder)   
            else :
                # print("PP2")
                _, pattern_check_gray_U500 = load_images(image_path, '../../data/template_images/area_extraction_templates/smaller/U500.jpg') 
                matched_result_U500 = perform_template_matching_with_roi(image, pattern_check_gray_U500, (6500, 700), (10000, 3200))

                # Matrice de transfo sur base du matching de 3 composants
                transformation_matrix = fit_results(matched_result_U500, matched_result_J704, matched_result_BT100, "U500_1", "J704_1", "BT100_1", df_main_board) 
                process_matched_results(image, df_main_board, orientation, board_id, PP1, transformation_matrix, top_folder)   

        
        case 'bottom':
            image, pattern_check_gray_U604_broad = load_images(image_path, '../../data/template_images/area_extraction_templates/U604.jpg') 
            top_left, bottom_right = perform_broad_template_matching_with_roi(image, pattern_check_gray_U604_broad,(3000, 600),(7800, 7000))
            _, pattern_check_gray_U604 = load_images(image_path, '../../data/template_images/area_extraction_templates/smaller/U604.jpg')
            matched_result_U604 = perform_template_matching_with_roi(image, pattern_check_gray_U604,top_left, bottom_right)
            if (matched_result_U604[1] > 4000):
                PP1 = True
                _, pattern_check_gray_Q400 = load_images(image_path, '../../data/template_images/area_extraction_templates/smaller/Q400.jpg')
                matched_result_Q400 = perform_template_matching_with_roi(image, pattern_check_gray_Q400, (5000,500), (7700,3000))
                _, pattern_check_gray_U100 = load_images(image_path, '../../data/template_images/area_extraction_templates/smaller/U100.jpg')
                matched_result_U100 = perform_template_matching_with_roi(image, pattern_check_gray_U100, (1000,300), (4100,3000))
                
                # Matrice de transfo sur base du matching de 3 composants
                transformation_matrix = fit_results(matched_result_U604, matched_result_Q400, matched_result_U100, "U604_1", "Q400_1", "U100_1", df_main_board) 
                process_matched_results(image, df_main_board, orientation, board_id, PP1, transformation_matrix, top_folder) 

            else:
                # print("PP2")
                _, pattern_check_gray_U707_Meas27 = load_images(image_path, '../../data/template_images/area_extraction_templates/U707_Meas27.jpg')
                top_left, bottom_right = perform_broad_template_matching_with_roi(image, pattern_check_gray_U707_Meas27,(7500, 4500),(10500, 7200))
                _, pattern_check_gray_MeasurePoint27 = load_images(image_path, '../../data/template_images/area_extraction_templates/smaller/MeasurePoint.jpg')
                matched_result_MeasurePoint27 = perform_template_matching_with_roi(image, pattern_check_gray_MeasurePoint27, top_left, bottom_right)
                _, pattern_check_gray_MeasurePoint17 = load_images(image_path, '../../data/template_images/area_extraction_templates/smaller/MeasurePoint.jpg')
                matched_result_MeasurePoint17 = perform_template_matching_with_roi(image, pattern_check_gray_MeasurePoint17, (4100, 4800), (6100, 8000))
                
                # Matrice de transfo sur base du matching de 3 composants
                transformation_matrix = fit_results(matched_result_U604, matched_result_MeasurePoint27, matched_result_MeasurePoint17, "U604_1", "Meas27_1", "Meas17_1", df_main_board)
                process_matched_results(image, df_main_board, orientation, board_id,PP1, transformation_matrix, top_folder)
                

        case _:
            print("Unknown orientation.") 

# Stopped here
# Processing /home/nick-kuijpers/Documents/Railnova/data/data_default/new/2.F.2/G48 - 6DCD535C35/Bottom/pp (1).jpg
def iterate_images():
    # Define the paths to the top folders
    top_folders = ['2.F.2', '2.F.3', '2.E.4']
    start = False
    # Iterate through each top folder
    for top_folder in top_folders:
        # Get the full path to the top folder
        top_folder_path = os.path.join('/home/nick-kuijpers/Documents/Railnova/data/data_default/new/', top_folder)
        
        # Import the corresponding csv file based on the top folder
        if top_folder == '2.F.2':
            df_main_board = pd.read_csv('../../data/board_info_csv/processed/board_data_2F2.csv')
        elif top_folder == '2.F.3':
            df_main_board = pd.read_csv('../../data/board_info_csv/processed/board_data_2F3.csv')
        elif top_folder == '2.E.4':
            df_main_board = pd.read_csv('../../data/board_info_csv/processed/board_data_2E4.csv')
        else:
            # Handle the case when the top folder is not recognized
            print("Invalid top folder:", top_folder)
            continue 

        # Iterate through each board folder
        for board_folder in sorted(os.listdir(top_folder_path)):
            # Check if the board folder is a string representation of a board ID
            if re.match('^G[0-9]{2,3}', board_folder): 
                # Extract the G code from the board folder
                g_code = re.search('^G[0-9]{2,3}', board_folder).group(0)
                # Get the full path to the board folder
                board_folder_path = os.path.join(top_folder_path, board_folder)
                clean_g_code = g_code.strip('- ')
                # print(clean_g_code)
                # Iterate through the two valid orientations (Top, Bottom) and (top, bottom)
                for orientation in ['Top', 'Bottom', 'top', 'bottom']:
                    # Check if the orientation folder exists before accessing it
                    if os.path.exists(os.path.join(board_folder_path, orientation)):
                        # Get the full path to the orientation folder
                        orientation_folder_path = os.path.join(board_folder_path, orientation)  
                        # Iterate through each image file (PP_1, PP_2)
                        for image_file in sorted(os.listdir(orientation_folder_path)): 
                            # Check if the image file is a pp image
                            if re.match(r'\b[ppPP]', image_file, flags=re.IGNORECASE): 
                                # Get the full path to the image file
                                image_path = os.path.join(orientation_folder_path, image_file) 
                                print(f"Processing {image_path}") 
                                if start == True:
                                    process_images(image_path, df_main_board, orientation, clean_g_code, top_folder)
                                if image_path=="/home/nick-kuijpers/Documents/Railnova/data/data_default/new/2.F.2/G54 - 5BC472F23B/top/pp (1).jpg":
                                    start = True

                            else:
                                # Skip the iteration if the image file is not a pp image 
                                continue
                    else:
                        # Skip the iteration if the orientation folder doesn't exist
                        continue
            else:
                # Skip the iteration if the board folder is not a string representation of a board ID
                print("Board folder is not a string representation of a board ID")
                continue
 

def main():
    iterate_images()

if __name__ == "__main__":
    main()
