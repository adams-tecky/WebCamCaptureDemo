// src/App.tsx
import React, { useState } from "react";
import WebcamCapture from "./WebCamCapture";
import ManualCapture from "./ManualCapture";
import "./CamCapture.css";

const App: React.FC = () => {
  const [show, setShow] = useState("manual");
  return (
    <div className="App">
      <header className="App-header">
        <h1>Face Detection App</h1>
        <button onClick={() => setShow("manual")}>Manual</button>
        <button onClick={() => setShow("auto")}>Auto</button>
        {show === "manual" ? <ManualCapture /> : <WebcamCapture />}
      </header>
    </div>
  );
};

export default App;
