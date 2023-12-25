import React, { useState }  from 'react';
import api from '../services/api';
import './AnalysisResults.css'; 

const AnalysisResults = ({ analysisData }) => {
    const [confirmations, setConfirmations] = useState(Array((analysisData?.analyzedImagePath || []).length).fill('yes'));
    
    // pathToSaveImage = f'{output_folder}{top_folder}/{board_id}_{orientation}_{modified_comp_string}.jpg'

    const handleConfirmationChange = (value, index) => {
        const newConfirmations = [...confirmations];
        newConfirmations[index] = value;
        setConfirmations(newConfirmations);
    };

    const handleSendFeedback = async () => {
        try {
            // Assuming analysisData includes an "id" field for identification
            // const response = await api.sendFeedback(analysisData.id, confirmation);
            const response = await api.sendFeedback(confirmations, analysisData);
            console.log('Feedback sent:', response);
            // Optionally, you can reset the confirmation state after sending feedback
            setConfirmations(Array((analysisData?.analyzedImagePath || []).length).fill('yes')); 
             
        } catch (error) {
            console.error('Error sending feedback:', error);
        }
    };


    return analysisData ? (
        <div className="container">

        {/* Display original uploaded image */}
        <div className="OriginalImage">
            <h2>Original Image</h2>
            <img src={`http://localhost:5000/images/${encodeURIComponent(analysisData.originalImagePath)}`} alt="Original"   />
        </div>

        {/* Display analyzed images and predictions */}
        <div className="AnalysisResults"> 
            {(analysisData.analyzedImagePath || []).map((analyzedImage, index) => (
                <div key={index}>
                    <img src={`http://localhost:5000/cropped_images/${encodeURIComponent(analyzedImage)}`} alt={`Analyzed ${index + 1}`} />
                    <pre>{(analysisData.predictions || []).map(pred => pred.join('\n'))[index]}</pre>
                    
                    {/* Confirmation dialog */} 
                    <p>Confirm analysis?</p>
                    <label>
                    <input
                        type="radio"
                        name={`confirmation-${index}`}
                        value="yes" 
                        onChange={() => handleConfirmationChange('yes', index)}
                        // checked={confirmations[index] === 'yes'}
                        checked={true }
                    />
                    Yes
                    </label>
                    <label>
                    <input
                        type="radio"
                        name={`confirmation-${index}`}
                        value="no"
                        onChange={() => handleConfirmationChange('no', index)}
                        checked={confirmations[index] === 'no'}
                    />
                    No
                    </label> 
                </div>
            ))}
        </div>

        {/* Send Feedback button */}
        {confirmations && (
            <div className='Feedback'>
            <button onClick={handleSendFeedback}>Send Feedback</button>
            </div>
        )}
    
    </div>     
) : null;
};

export default AnalysisResults;
        