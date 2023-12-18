from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS, cross_origin
import json 
import torch
import pandas as pd
import os

from inference import predict
from inference_image_extractor_v2 import process_image

app = Flask(__name__)
CORS(app)

def clean_comp_name(component_name):
    if component_name.startswith('U901'):
        return 'U901'
    elif component_name.startswith('U904'):
        return 'U904'
    elif component_name.startswith('U911'):
        return 'U911'
    else:
        return component_name

@app.route('/analyze', methods=['POST'])
@cross_origin() 
def analyze_image():
    predictions = []
    data = request.get_json() 
    image_path = data.get('path').get('path')
    orientation = data.get('path').get('orientation')
    clean_g_code = data.get('path').get('board_id')
    top_folder = data.get('path').get('top_folder')

    print("Image path: ", image_path)
    print("Orientation: ", orientation)
    print("Clean G Code: ", clean_g_code)
    print("Top folder: ", top_folder)

    # 1. Find image
    # folder_name = image_path.split('_')[0]
    base_path = '../../data/data_default_processed/'
    real_image_path = base_path + image_path

    # 2. Execute images extraction 
    # Returns [image_path_cropped_to_component, component_name]
    cropped_images_path = process_image(real_image_path, orientation, clean_g_code, top_folder)
  
    # 3. Load model and execute inference 
        # Should be on each image from step 2, append to a list the results
    all_cropped_images = []
    for cropped_image_path in cropped_images_path:
        print(cropped_image_path)
        component_name = cropped_image_path[1]
        component_name = clean_comp_name(component_name)
        if component_name == 'U500':
            predictions.append(["U500", cropped_image_path[1]])
        else:
            model_string = '/home/nick-kuijpers/Documents/Railnova/Python/backend/models/trained_model_CNN_' + component_name + '.pt' 
            model = torch.load(model_string, map_location='cpu')
            all_cropped_images.append(cropped_image_path[0])
            # cropped_image_path[0] = '/home/nick-kuijpers/Documents/Railnova/Python/backend/all_components/2F3/cropped_images/' + cropped_image_path[0]
            predictions.append([predict(model, cropped_image_path[0], component_name), cropped_image_path[1]])
        
    
    # 4. Return results
    # analysis_results = {'result': 'Analysis complete', 'originalImagePath': image_path, 'predictions': predictions, 'analyzedImagePath': all_cropped_images}
    analysis_results = {
        'result': 'Analysis complete',
        'originalImagePath': image_path,
        'predictions': predictions,
        'analyzedImagePath': all_cropped_images
    }
    return jsonify(analysis_results)
     
# Serve original images route
@app.route('/images/<path:filename>')
def serve_original_image(filename): 
    return send_from_directory(os.path.join('/home/nick-kuijpers/Documents/Railnova/data/data_default_processed/'), filename)

# Serve cropped images route
@app.route('/cropped_images/<path:filename>')
def serve_cropped_image(filename):  
    return send_from_directory(os.path.join('/'), filename)

@app.route('/send-feedback', methods=['POST'])
def send_feedback():
    try:
        data = request.json
        confirmations = data.get('confirmations')
        analysis_data = data.get('analysisData')

        # Save the updated data to a JSON file
        feedback_dir = 'feedback'
        if not os.path.exists(feedback_dir):
            os.makedirs(feedback_dir)
        feedback_file = os.path.join(feedback_dir, 'feedback.json')
        with open(feedback_file, 'w') as f:
            json.dump({'confirmations': confirmations, 'analysis_data': analysis_data}, f)

        return jsonify({'message': 'Feedback received successfully'})
    except Exception as e:
        print(f"Error sending feedback: {str(e)}")
        return jsonify({'error': 'Failed to send feedback'}), 500


if __name__ == '__main__':
    app.run(debug=True)
