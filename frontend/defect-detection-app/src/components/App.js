import React, { useState } from 'react';
import UploadImage from './UploadImage';
import LoadingScreen from './LoadingScreen';
import AnalysisResults from './AnalysisResults';
import './App.css'; 

const App = () => {
  const [analysisData, setAnalysisData] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleAnalysisComplete = (data) => {
    setAnalysisData(data);
    setLoading(false);
  };

  const handleFileUpload = () => {
    setLoading(true);
    setAnalysisData(null)
  };

  return (
    <div className="App">
      <div className="Input">
        <h1>Railnova Defect Detection Application</h1>
        <UploadImage onFileUpload={handleFileUpload} onUploadComplete={handleAnalysisComplete} />
        
      </div>

      <LoadingScreen loading={loading} />

      <AnalysisResults analysisData={analysisData} />
    </div>
  );
};

export default App;
