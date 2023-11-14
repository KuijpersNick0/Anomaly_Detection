import React, { useState } from 'react';
import api from '../services/api';

const UploadImage = ({ onFileUpload, onUploadComplete }) => {
  const [selectedFile, setSelectedFile] = useState(null);

  const handleFileChange = (event) => {
    const fileInput = event.target;
    const selectedFile = fileInput.files[0];
    setSelectedFile(selectedFile);

    // Reset analysis results when a new file is chosen
    onUploadComplete(null);
  };

  const handleConfirm = async () => {
    if (selectedFile) {
      onFileUpload();
      try {
        const response = await api.analyzeImage({ path: selectedFile.name });
        onUploadComplete(response);
      } catch (error) {
        console.error('Error analyzing image:', error);
      }
    }
  };

  return (
    <div>
      <input type="file" onChange={handleFileChange} />
      {selectedFile && (
        <button onClick={handleConfirm}>Confirm</button>
      )}
    </div>
  );
};

export default UploadImage;
