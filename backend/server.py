from flask import Flask, request, jsonify
from flask_cors import CORS
import json 
import torch
import pandas as pd

from inference import predict
from inference_image_extractor import main
import os

app = Flask(__name__)
CORS(app)

@app.route('/analyze', methods=['POST'])
def analyze_image():
    predictions = []
    data = request.get_json()
    image_path = data.get('path').get('path')
    
    # 1. Find image
    # folder_name = image_path.split('_')[0]
    base_path = '../../data/data_default_processed/'
    real_image_path = base_path + image_path

    print(real_image_path)

    # 2. Execute images extraction
        # To be modified
        # Returns new image path (and bounding boxes?)
    df_main_board = pd.read_csv('../../data/board_info_csv/processed/board_data_2F3.csv')
    cropped_images_path = main(real_image_path, df_main_board)

    print(cropped_images_path)

    # 3. Load model and execute inference 
        # Should be on each image from step 2, append to a list the results
    model = torch.load('/home/nick-kuijpers/Documents/Railnova/Python/backend/models/trained_model.pt', map_location='cpu')
    for cropped_image_path in cropped_images_path:
        print(cropped_image_path)
        predictions.append([predict(model, cropped_image_path[0]), cropped_image_path[1]])
        
    
    # 4. Return results
    analysis_results = {'result': 'Analysis complete', 'image_path': image_path, 'predictions': predictions}
    return jsonify(analysis_results)


@app.route('/send-feedback', methods=['POST'])
def send_feedback():
    try:
        data = request.json
        confirmation = data.get('confirmation')
        analysis_data = data.get('analysisData')

        # Save the updated data to a JSON file
        feedback_dir = 'feedback'
        if not os.path.exists(feedback_dir):
            os.makedirs(feedback_dir)
        feedback_file = os.path.join(feedback_dir, 'feedback.json')
        with open(feedback_file, 'w') as f:
            json.dump({'confirmation': confirmation, 'analysis_data': analysis_data}, f)

        return jsonify({'message': 'Feedback received successfully'})
    except Exception as e:
        print(f"Error sending feedback: {str(e)}")
        return jsonify({'error': 'Failed to send feedback'}), 500


if __name__ == '__main__':
    app.run(debug=True)
