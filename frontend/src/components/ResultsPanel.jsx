import GradCamViewer from "./GradCamViewer.jsx";

function confidenceColor(value) {
  if (value >= 0.75) return "from-teal-400 to-teal-300";
  if (value >= 0.5) return "from-violet-500 to-violet-400";
  return "from-rose-500 to-rose-400";
}

export default function ResultsPanel({ result, previewUrl }) {
  if (!result) return null;

  const sortedProbs = Object.entries(result.all_probabilities).sort(
    (a, b) => b[1] - a[1]
  );

  return (
    <section className="w-full max-w-5xl mx-auto mt-12 grid md:grid-cols-2 gap-8 px-4">
      <div>
        <GradCamViewer
          originalUrl={previewUrl}
          overlayBase64={result.gradcam_overlay_base64}
          dominantRegion={result.dominant_region}
        />
      </div>

      <div className="flex flex-col gap-6">
        <div className="bg-surface/70 backdrop-blur-sm rounded-2xl p-6 border border-violet-500/20">
          <p className="font-mono text-xs uppercase tracking-wider text-ink-500 mb-1">
            Especie predicha
          </p>
          <h2 className="font-display text-2xl font-600 text-ink-100">
            {result.predicted_class}
          </h2>
          <p className="font-mono text-3xl mt-2 bg-gradient-to-r from-violet-400 to-teal-400 bg-clip-text text-transparent">
            {(result.confidence * 100).toFixed(1)}%
          </p>
          <div className="leaf-vein-divider my-4" />
          <p className="text-sm text-ink-300 font-body leading-relaxed">
            {result.explanation_text}
          </p>
        </div>

        <div className="bg-surface/70 backdrop-blur-sm rounded-2xl p-6 border border-violet-500/20">
          <p className="font-mono text-xs uppercase tracking-wider text-ink-500 mb-4">
            Probabilidad por especie
          </p>
          <div className="flex flex-col gap-3">
            {sortedProbs.map(([className, prob]) => (
              <div key={className}>
                <div className="flex justify-between text-xs font-body text-ink-300 mb-1">
                  <span>{className}</span>
                  <span className="font-mono">{(prob * 100).toFixed(1)}%</span>
                </div>
                <div className="w-full h-2 rounded-full bg-surface-2 overflow-hidden">
                  <div
                    className={`confidence-bar h-full rounded-full bg-gradient-to-r ${confidenceColor(prob)}`}
                    style={{ width: `${prob * 100}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}
