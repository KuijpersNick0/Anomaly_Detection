import React, { useState } from 'react';
import UploadImage from './UploadImage';
import LoadingScreen from './LoadingScreen';
import AnalysisResults from './AnalysisResults';

const App = () => {
  const [analysisData, setAnalysisData] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleAnalysisComplete = (data) => {
    setAnalysisData(data);
    setLoading(false);
  };

  const handleFileUpload = () => {
    setLoading(true);
  };

  return (
    <div className="App">
      <h1>Railnova PCB defect detection app</h1>

      <UploadImage onFileUpload={handleFileUpload} onUploadComplete={handleAnalysisComplete} />

      <LoadingScreen loading={loading} />

      <AnalysisResults analysisData={analysisData} />
    </div>
  );
};

export default App;
