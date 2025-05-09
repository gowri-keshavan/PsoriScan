import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [image, setImage] = useState(null);
  const [result, setResult] = useState(null);

  const handleImageChange = (e) => {
    setImage(e.target.files[0]);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!image) {
      alert("Please select an image.");
      return;
    }
  
    const formData = new FormData();
    formData.append("image", image);
  
    try {
      const response = await fetch("http://localhost:5000/predict", {
        method: "POST",
        body: formData,
      });
  
      if (!response.ok) throw new Error("Upload failed");
  
      const data = await response.json();
      console.log(data);
      setResult(data);
    } catch (err) {
      alert("Upload failed.");
      console.error("Error:", err);
    }
  };
  
  
  return (
    <div className="app-container">
      <div className="card">
        <h2>🧬 Psoriasis Detection Tool</h2>
        <form onSubmit={handleSubmit}>
          <input type="file" onChange={handleImageChange} accept="image/*" />
          <button type="submit">Analyze Image</button>
        </form>

        {result && (
          <div className="results">
            <h3>Result: <span className="label">{result.label}</span></h3>
            <p>Confidence: <strong>{result.confidence}%</strong></p>
            <div className="images">
              <img src={`http://localhost:5000/${result.original_img}`} alt="Original" />
              <img src={`http://localhost:5000/${result.output_img}`} alt="Processed" />
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
