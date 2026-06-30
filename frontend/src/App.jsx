import { useState } from "react";
import Header from "./components/Header.jsx";
import UploadPanel from "./components/UploadPanel.jsx";
import ResultsPanel from "./components/ResultsPanel.jsx";

export default function App() {
  const [result, setResult] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  async function handleAnalyze(file) {
    setLoading(true);
    setError(null);
    setResult(null);
    setPreviewUrl(URL.createObjectURL(file));

    try {
      const formData = new FormData();
      formData.append("image", file);

      const response = await fetch("http://127.0.0.1:5001/api/predict", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        throw new Error(payload.detail || "No se pudo analizar la imagen.");
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err.message || "Ocurrió un error inesperado.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="min-h-screen pb-24">
      <Header />
      <UploadPanel onAnalyze={handleAnalyze} loading={loading} />

      {error && (
        <p className="text-center text-rose-400 font-body mt-6 max-w-md mx-auto">
          {error}
        </p>
      )}

      <ResultsPanel result={result} previewUrl={previewUrl} />

      <footer className="text-center text-ink-500 text-xs font-mono mt-20">
        Aquatic Plants Classifier — EfficientNetB0 + Grad-CAM ·{" "}
        <a
          href="https://github.com/TogerAndres"
          className="underline hover:text-teal-400"
          target="_blank"
          rel="noreferrer"
        >
          TogerAndres
        </a>
      </footer>
    </main>
  );
}
