import { useEffect, useRef, useState } from "react";
import { analyzeImage } from "./api";
import "./App.css";

const MAX_MB = Number(process.env.REACT_APP_MAX_UPLOAD_MB || 8);
const ACCEPTED_TYPES = ["image/jpeg", "image/png", "image/webp", "image/bmp"];

function validate(file) {
  if (!ACCEPTED_TYPES.includes(file.type)) {
    return "Choose a JPEG, PNG, WEBP or BMP photo.";
  }
  if (file.size > MAX_MB * 1024 * 1024) {
    return `That photo is larger than ${MAX_MB} MB. Choose a smaller one.`;
  }
  return null;
}

const percent = (value) => `${Math.round(value * 100)}%`;

function ProbabilityBar({ probability, threshold }) {
  return (
    <div className="meter">
      <div
        className="meter__track"
        role="img"
        aria-label={`Psoriasis probability ${percent(probability)}, decision threshold ${percent(threshold)}`}
      >
        <div className="meter__fill" style={{ width: percent(probability) }} />
        <div className="meter__threshold" style={{ left: percent(threshold) }} />
      </div>
      <div className="meter__scale" aria-hidden="true">
        <span>0%</span>
        <span style={{ left: percent(threshold) }}>threshold</span>
        <span>100%</span>
      </div>
    </div>
  );
}

export default function App() {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [status, setStatus] = useState("idle"); // idle | loading | done
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [strength, setStrength] = useState(70);
  const abortRef = useRef(null);
  const inputRef = useRef(null);

  useEffect(() => () => abortRef.current?.abort(), []);
  useEffect(() => {
    if (!previewUrl) return undefined;
    return () => URL.revokeObjectURL(previewUrl);
  }, [previewUrl]);

  function reset() {
    abortRef.current?.abort();
    setFile(null);
    setPreviewUrl(null);
    setResult(null);
    setError(null);
    setStatus("idle");
    if (inputRef.current) inputRef.current.value = "";
  }

  function handleFileChange(event) {
    const chosen = event.target.files?.[0];
    if (!chosen) return;
    abortRef.current?.abort();
    setResult(null);
    setStatus("idle");
    const problem = validate(chosen);
    if (problem) {
      setFile(null);
      setPreviewUrl(null);
      setError(problem);
      event.target.value = "";
      return;
    }
    setError(null);
    setFile(chosen);
    setPreviewUrl(URL.createObjectURL(chosen));
  }

  async function handleAnalyze() {
    if (!file) return;
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    setStatus("loading");
    setError(null);
    try {
      setResult(await analyzeImage(file, controller.signal));
      setStatus("done");
    } catch (err) {
      if (err.name === "AbortError") return;
      setError(err.message);
      setStatus("idle");
    }
  }

  const loading = status === "loading";

  return (
    <div className="page">
      <header className="masthead">
        <h1>PsoriScan</h1>
        <p>
          Upload a close-up photo of a patch of skin. A neural network estimates how likely it is to show
          psoriasis and highlights the areas that drove its answer.
        </p>
      </header>

      <p className="notice">
        This is a machine learning project, not a medical device. It cannot diagnose psoriasis. If you are
        worried about your skin, see a dermatologist.
      </p>

      <main className="workspace">
        <section className="panel" aria-labelledby="upload-title">
          <h2 id="upload-title">Your photo</h2>
          <label className="picker">
            <input
              ref={inputRef}
              type="file"
              accept={ACCEPTED_TYPES.join(",")}
              onChange={handleFileChange}
              aria-label="Choose a photo"
            />
            <span className="picker__title">{file ? file.name : "Choose a photo"}</span>
            <span className="picker__hint">
              JPEG, PNG, WEBP or BMP, up to {MAX_MB} MB. Photos are analyzed in memory and are not stored.
            </span>
          </label>

          <div className="actions">
            <button type="button" className="button button--primary" onClick={handleAnalyze} disabled={!file || loading}>
              {loading ? "Analyzing…" : "Analyze photo"}
            </button>
            <button type="button" className="button" onClick={reset} disabled={!file && !error}>
              Clear
            </button>
          </div>

          {error && (
            <p className="error" role="alert">
              {error}
            </p>
          )}
        </section>

        <section className="panel stage" aria-labelledby="result-title" aria-busy={loading}>
          <h2 id="result-title">Result</h2>

          {!previewUrl && <p className="empty">Your photo and the result appear here. Choose a photo to begin.</p>}

          {previewUrl && (
            <div className="frame">
              <img className="frame__photo" src={previewUrl} alt="Close-up of the skin you chose" />
              {result && (
                <img
                  className="frame__heat"
                  src={result.heatmap}
                  alt="Heatmap of the regions that raised the psoriasis score"
                  style={{ opacity: strength / 100 }}
                />
              )}
              {loading && <div className="frame__busy"><span className="spinner" aria-hidden="true" /></div>}
            </div>
          )}

          {result && (
            <div className="verdict" aria-live="polite">
              <p className="verdict__label">{result.label}</p>
              <p className="verdict__detail">
                Psoriasis probability {percent(result.probability_psoriasis)}. The model is{" "}
                {percent(result.confidence)} confident in this answer.
              </p>
              <ProbabilityBar probability={result.probability_psoriasis} threshold={result.threshold} />

              <div className="slider">
                <label htmlFor="strength">Heatmap strength</label>
                <input
                  id="strength"
                  type="range"
                  min="0"
                  max="100"
                  value={strength}
                  onChange={(e) => setStrength(Number(e.target.value))}
                />
              </div>
              <p className="footnote">
                Warmer colors mark areas that pushed the score toward psoriasis. The heatmap shows how the
                model reached its answer, not where a condition is. Model: {result.model.name}.
              </p>
            </div>
          )}
        </section>
      </main>
    </div>
  );
}
