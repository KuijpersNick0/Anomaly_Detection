const BASE_URL = 'http://localhost:5000'; // Adjust to Flask backend URL

const api = {
  analyzeImage: async (path) => {
    const response = await fetch(`${BASE_URL}/analyze`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ path }), // Send the path in the request body
    });

    if (!response.ok) {
      throw new Error('Image analysis failed');
    }

    return response.json();
  },

  // Add other API methods as needed
};

export default api;
