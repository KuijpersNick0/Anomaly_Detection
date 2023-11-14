import React, { useState }  from 'react';
import api from '../services/api';

const AnalysisResults = ({ analysisData }) => {
    const [confirmation, setConfirmation] = useState(null); 
    
    const handleConfirmationChange = (value) => {
        setConfirmation(value);
    };

    const handleSendFeedback = async () => {
        try {
            // Assuming analysisData includes an "id" field for identification
            // const response = await api.sendFeedback(analysisData.id, confirmation);
            const response = await api.sendFeedback(confirmation, analysisData);
            console.log('Feedback sent:', response);
            // Optionally, you can reset the confirmation state after sending feedback
            setConfirmation(null);
        } catch (error) {
            console.error('Error sending feedback:', error);
        }
    };


    return analysisData && (
    <div>
        
        {/* Analysis results */}
        <h2>Analysis Results</h2>
        <pre>{JSON.stringify(analysisData, null, 2)}</pre>
    
        {/* Confirmation dialog */}
        <div>
            <p>Do you confirm the analysis?</p>
            <label>
            <input
                type="radio"
                name="confirmation"
                value="yes"
                onChange={() => handleConfirmationChange('yes')}
                checked={confirmation === 'yes'}
            />
            Yes
            </label>
            <label>
            <input
                type="radio"
                name="confirmation"
                value="no"
                onChange={() => handleConfirmationChange('no')}
                checked={confirmation === 'no'}
            />
            No
            </label>
        </div>

        {/* Send Feedback button */}
        {confirmation && (
            <button onClick={handleSendFeedback}>Send Feedback</button>
        )}
    
    </div>
);
};

export default AnalysisResults;
