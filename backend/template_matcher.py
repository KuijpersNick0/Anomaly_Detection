import cv2
import numpy as np

def find_template_match(input_image_path, template_image_path, to_save_im_path):
    # Load input and template images
    input_image = cv2.imread(input_image_path)
    template_image = cv2.imread(template_image_path)

    # Convert images to grayscale
    input_gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template_image, cv2.COLOR_BGR2GRAY)

    # Get dimensions of template image
    template_height, template_width = template_gray.shape[::-1]

    # Perform template matching
    result = cv2.matchTemplate(input_gray, template_gray, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    # Get coordinates of best match
    top_left = max_loc
    bottom_right = (top_left[0] + template_height, top_left[1] + template_width)

    # Crop the matched region from the input image
    cropped_image = input_image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

    # Display input image with best match highlighted
    # cv2.rectangle(input_image, top_left, bottom_right, (0, 0, 255), 2)
    # cv2.imshow('Input Image', input_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    cv2.imwrite(to_save_im_path, cropped_image)

    # Return cropped image of best match
    return cropped_image
