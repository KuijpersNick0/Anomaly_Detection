import React, { useState } from 'react';
import UploadImage from './UploadImage'; 

const App = () => {
  const [analysisData, setAnalysisData] = useState(null);

  // Function to handle analysis completion
  const handleAnalysisComplete = (data) => {
    setAnalysisData(data);
  };

  return (
    <div className="App">
      <h1>Railnova PCB defect detection app</h1>

      {/* UploadImage Component */}
      <UploadImage onUploadComplete={handleAnalysisComplete} />

      {/* Display analysis results if available */}
      {analysisData && (
        <div>
          <h2>Analysis Results</h2>
          <pre>{JSON.stringify(analysisData, null, 2)}</pre>
        </div>
      )}

    </div>
  );
};

export default App;
