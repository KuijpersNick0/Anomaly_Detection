from flask import Flask, request, jsonify
from flask_cors import CORS
import json 
import torch

from inference import predict

app = Flask(__name__)
CORS(app)

@app.route('/analyze', methods=['POST'])
def analyze_image():
    data = request.get_json()
    image_path = data.get('path').get('path')
    
    # 1. Find image
    folder_name = image_path.split('_')[0]
    base_path = f'../../data/CNN_images/Run1/{folder_name}/'
    real_image_path = base_path + image_path

    # 2. Execute images extraction
        # To be modified
        # Returns new image path (and bounding boxes?)

    # 3. Load model and execute inference 
        # Should be on each image from step 2, append to a list the results
    model = torch.load('/home/nick-kuijpers/Documents/Railnova/Python/backend/models/trained_model.pt', map_location='cpu')
    predictions = predict(model, real_image_path)
    
    # 4. Return results
    analysis_results = {'result': 'Analysis complete', 'image_path': image_path, 'predictions': predictions}
    return jsonify(analysis_results)

if __name__ == '__main__':
    app.run(debug=True)
