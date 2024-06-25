// src/App.tsx
import React from 'react';
import WebcamCapture from './WebCamCapture';

const App: React.FC = () => {
  return (
    <div className="App">
      <header className="App-header">
        <h1>Face Detection App</h1>
        <WebcamCapture />
      </header>
    </div>
  );
};

export default App;
