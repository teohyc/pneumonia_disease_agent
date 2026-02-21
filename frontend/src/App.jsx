import { useState } from "react";
import "./App.css";

function App() {
  const [selectedImage, setSelectedImage] = useState(null);
  const [heatmap, setHeatmap] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [report, setReport] = useState("");
  const [loading, setLoading] = useState(false);

  const handleUpload = async () => {
    if (!selectedImage) {
      alert("Please select an image.");
      return;
    }

    setLoading(true);

    const formData = new FormData();
    formData.append("file", selectedImage);

    try {
      const response = await fetch("http://localhost:8000/diagnose", {
        method: "POST",
        body: formData
      });

      const data = await response.json();

      setPrediction(data.prediction);
      setReport(data.report);
      setHeatmap("data:image/jpeg;base64," + data.heatmap_base64);

    } catch (err) {
      alert("Error connecting to backend.");
    }

    setLoading(false);
  };

  return (
  <div className="dashboard">
    <header className="header">
      <h1>AI Chest X-Ray Diagnostic System</h1>
    </header>

    <div className="upload-bar">
      <input
        type="file"
        accept="image/*"
        onChange={(e) => setSelectedImage(e.target.files[0])}
      />
      <button onClick={handleUpload}>
        {loading ? "Analyzing..." : "Analyze X-Ray"}
      </button>
    </div>

    <div className="content-grid">

      {/* LEFT PANEL - IMAGES */}
      <div className="image-panel">

        {selectedImage && (
          <div className="card">
            <h3>Original X-Ray</h3>
            <img
              src={URL.createObjectURL(selectedImage)}
              alt="original"
              className="medical-image"
            />
          </div>
        )}

        {heatmap && (
          <div className="card">
            <h3>Attention Heatmap</h3>
            <img
              src={heatmap}
              alt="heatmap"
              className="medical-image"
            />
          </div>
        )}

      </div>

      {/* RIGHT PANEL - RESULTS */}
      <div className="result-panel">

        {prediction && (
          <div className="card">
            <h2>Prediction Summary</h2>

            <table className="medical-table">
              <thead>
                <tr>
                  <th>Class</th>
                  <th>Probability</th>
                </tr>
              </thead>
              <tbody>
                {prediction.top_k.map((item, index) => (
                  <tr key={index}>
                    <td>{item.class}</td>
                    <td>{item.prob}</td>
                  </tr>
                ))}
              </tbody>
            </table>

            <div className="diagnosis-box">
              <strong>Final Diagnosis:</strong>{" "}
              {prediction.predicted_class}
              <br />
              <strong>Confidence:</strong>{" "}
              {prediction.confidence}
            </div>
          </div>
        )}

        {report && (
          <div className="card">
            <h2>Medical Explanation</h2>
            <pre className="report-text">
              {report}
            </pre>
          </div>
        )}

      </div>
    </div>
  </div>
);
}

export default App;