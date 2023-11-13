import React, { useState } from 'react';
import api from '../services/api';

const UploadImage = ({ onUploadComplete }) => {
  const handleFileChange = async (event) => {
    const fileInput = event.target;
    const fileName = fileInput.files[0]?.name;

    try {
      const response = await api.analyzeImage({ path: fileName });
      onUploadComplete(response);
    } catch (error) {
      console.error('Error analyzing image:', error);
    }
  };

  return (
    <div>
      <input type="file" onChange={handleFileChange} />
    </div>
  );
};

export default UploadImage;
