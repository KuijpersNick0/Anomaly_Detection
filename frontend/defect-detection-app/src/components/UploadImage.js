import React, { useState } from 'react';
import api from '../services/api';

const UploadImage = ({ onFileUpload, onUploadComplete }) => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [orientation, setOrientation] = useState('');
  const [boardId, setBoardId] = useState('');
  const [topFolder, setTopFolder] = useState('');

  const handleFileChange = (event) => {
    const fileInput = event.target;
    const selectedFile = fileInput.files[0];
    setSelectedFile(selectedFile);

    // Reset analysis results when a new file is chosen
    onUploadComplete(null);
  };

  const handleOrientationChange = (event) => {
    setOrientation(event.target.value);
  };

  const handleBoardIdChange = (event) => {
    setBoardId(event.target.value);
  };

  const handleTopFolderChange = (event) => {
    setTopFolder(event.target.value);
  };

  const handleConfirm = async () => {
    if (selectedFile) {
      onFileUpload();
      try {
        const response = await api.analyzeImage({
          path: selectedFile.name,
          orientation: orientation,
          board_id: boardId,
          top_folder: topFolder,
        });
        onUploadComplete(response);
      } catch (error) {
        console.error('Error analyzing image:', error);
      }
    }
  };

  return (
    <div>
      
      <img src="https://www.railnova.eu/wp-content/uploads/2017/09/Railnova-logo-color-1.jpg" alt="Railnova logo" />
      <input type="file" onChange={handleFileChange} />
      <div>
        <label>Orientation: </label>
        <input type="text" value={orientation} onChange={handleOrientationChange} />
      </div>
      <div>
        <label>Board ID: </label>
        <input type="text" value={boardId} onChange={handleBoardIdChange} />
      </div>
      <div>
        <label>Top Folder: </label>
        <input type="text" value={topFolder} onChange={handleTopFolderChange} />
      </div>
      {selectedFile && (
        <button onClick={handleConfirm}>Confirm</button>
      )}
    </div>
  );
};

export default UploadImage;
