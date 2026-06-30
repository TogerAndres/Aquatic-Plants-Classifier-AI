import { useCallback, useRef, useState } from "react";

const ACCEPTED_TYPES = ["image/jpeg", "image/png", "image/webp"];

export default function UploadPanel({ onAnalyze, loading }) {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const [validationError, setValidationError] = useState(null);
  const inputRef = useRef(null);

  const handleFile = useCallback((selected) => {
    if (!selected) return;
    if (!ACCEPTED_TYPES.includes(selected.type)) {
      setValidationError("Formato no soportado. Usa JPG, PNG o WEBP.");
      return;
    }
    setValidationError(null);
    setFile(selected);
    setPreviewUrl(URL.createObjectURL(selected));
  }, []);

  const onDrop = useCallback(
    (event) => {
      event.preventDefault();
      setIsDragging(false);
      handleFile(event.dataTransfer.files?.[0]);
    },
    [handleFile]
  );

  return (
    <div className="w-full max-w-xl mx-auto">
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setIsDragging(true);
        }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={onDrop}
        onClick={() => inputRef.current?.click()}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => e.key === "Enter" && inputRef.current?.click()}
        className={`relative rounded-2xl border-2 border-dashed transition-colors cursor-pointer
          ${isDragging ? "border-teal-400 bg-teal-400/5" : "border-violet-500/30 hover:border-violet-400/60"}
          bg-surface/60 backdrop-blur-sm p-8 text-center`}
      >
        <input
          ref={inputRef}
          type="file"
          accept="image/jpeg,image/png,image/webp"
          className="hidden"
          onChange={(e) => handleFile(e.target.files?.[0])}
        />

        {previewUrl ? (
          <img
            src={previewUrl}
            alt="Vista previa de la imagen seleccionada"
            className="mx-auto max-h-56 rounded-lg object-contain shadow-glow"
          />
        ) : (
          <div className="flex flex-col items-center gap-2 py-6">
            <div className="w-12 h-12 rounded-full bg-gradient-to-br from-violet-500 to-teal-400 opacity-80" />
            <p className="font-body text-ink-100 font-medium">
              Arrastra una imagen aquí o haz clic para elegirla
            </p>
            <p className="font-mono text-xs text-ink-500">JPG · PNG · WEBP — máx. 8 MB</p>
          </div>
        )}
      </div>

      {validationError && (
        <p className="text-sm text-rose-400 mt-2 text-center">{validationError}</p>
      )}

      <button
        type="button"
        disabled={!file || loading}
        onClick={() => onAnalyze(file)}
        className="w-full mt-5 font-display font-600 rounded-xl py-3 transition-all
          bg-gradient-to-r from-violet-500 to-teal-400 text-abyss
          disabled:opacity-40 disabled:cursor-not-allowed
          hover:shadow-glow-teal active:scale-[0.99]"
      >
        {loading ? "Analizando…" : "Analizar imagen"}
      </button>
    </div>
  );
}
