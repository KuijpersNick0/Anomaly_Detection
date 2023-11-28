import cv2
import numpy as np
import os

# image_path
img_path="../../data/data_default_processed/2.F.2_G296 - 5EF56B3888_Top_PP1.jpg"

# read image
img_raw = cv2.imread(img_path)
img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
cv2.namedWindow("Display", cv2.WINDOW_NORMAL)
# Adjust the dimensions as needed
# resized_image = cv2.resize(img_raw, (800, 600))  

# select ROIs function
ROIs = cv2.selectROIs("Display", img_raw, True)

# print rectangle points of selected roi
print(ROIs)

#counter to save image with different name
# crop_component = "J2"
# crop_component = "U500"
# crop_component = "U600"
# crop_component = "U701"
# crop_component = "2F3_U904Ux"
# crop_component = "2F3_U911Ux"
# crop_component = "Q102"
# crop_component = "U901_AH"
# crop_component = "U608"
# crop_component = "U103"
# crop_component = "J704"
# crop_component = "BT100"
# crop_component = "U911U5"
# crop_component = "U904U5"
# crop_component = "U604"
# crop_component = "J703"
# crop_component = "U7"
# crop_component = "J703_J702"
# crop_component = "J702"

# crop_component = "U101"
# crop_component = "Q400"
# crop_component = "C401"
# crop_component = "L400"
# crop_component = "C408"
# crop_component = "L500"
# crop_component = "C525"
# crop_component = "C526"
# crop_component = "Q204"
# crop_component = "U4"

# crop_component = "U604"
# crop_component = "U7"
# crop_component = "J703"

# crop_component = "MeasurePoint"

crop_component = "U904U1"

#loop over every bounding box save in array "ROIs"
for rect in ROIs:
    x1=rect[0]
    y1=rect[1]
    x2=rect[2]
    y2=rect[3]

    #crop roi from original image
    img_crop = img_raw[y1:y1+y2,x1:x1+x2] 

    #show cropped image
    cv2.imshow("crop"+str(crop_component),img_crop)

    #save cropped image 
    output_folder = '../../data/template_images/area_extraction_templates/'
    output_path = os.path.join(output_folder, str(crop_component) + '.jpg')
    cv2.imwrite(output_path , img_crop) 

#hold window
cv2.waitKey(0)