const BASE_URL = 'http://localhost:5000'; // Adjust to Flask backend URL

const api = {
  // Analyze an image and return the analysis results
  analyzeImage: async (path, orientation, boardId, topFolder) => {
    const response = await fetch(`${BASE_URL}/analyze`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ 
        path: path,
        orientation: orientation,
        board_id: boardId,
        top_folder: topFolder,
      }), // Send the image path and other parameters to the backend
    });

    if (!response.ok) {
      throw new Error('Image analysis failed');
    }

    return response.json();
  },

  // Send feedback to the backend about the analysis results
  // sendFeedback: async (analysisId, confirmation) => {
  sendFeedback: async (confirmations, analysisData) => {
    const response = await fetch(`${BASE_URL}/send-feedback`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({confirmations, analysisData }),
    });

    if (!response.ok) {
      throw new Error('Failed to send feedback');
    }

    return response.json();
  },
};

export default api;
