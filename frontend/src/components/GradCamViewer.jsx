import { useRef, useState } from "react";

/**
 * Slider comparador estilo "antes/después" entre la foto original y el
 * mapa de calor Grad-CAM. Es el elemento distintivo de la interfaz: en
 * vez de mostrar el heatmap como una imagen aparte, se revela "dentro"
 * de la misma foto, como si se deslizara una lente sobre la planta.
 */
export default function GradCamViewer({ originalUrl, overlayBase64, dominantRegion }) {
  const [sliderPos, setSliderPos] = useState(50);
  const containerRef = useRef(null);

  return (
    <div className="w-full">
      <div
        ref={containerRef}
        className="relative w-full aspect-square rounded-2xl overflow-hidden shadow-glow select-none"
      >
        <img
          src={originalUrl}
          alt="Imagen original"
          className="absolute inset-0 w-full h-full object-cover"
          draggable={false}
        />
        <div
          className="absolute inset-0 w-full h-full overflow-hidden"
          style={{ clipPath: `inset(0 ${100 - sliderPos}% 0 0)` }}
        >
          <img
            src={`data:image/png;base64,${overlayBase64}`}
            alt="Mapa de calor Grad-CAM"
            className="absolute inset-0 w-full h-full object-cover"
            draggable={false}
          />
        </div>

        {/* Línea divisora de la lente */}
        <div
          className="absolute top-0 bottom-0 w-[2px] bg-teal-300/90"
          style={{ left: `${sliderPos}%` }}
        />

        <span className="absolute top-3 left-3 font-mono text-[10px] uppercase tracking-wider bg-abyss/70 px-2 py-1 rounded-full text-ink-300">
          Original
        </span>
        <span className="absolute top-3 right-3 font-mono text-[10px] uppercase tracking-wider bg-abyss/70 px-2 py-1 rounded-full text-teal-300">
          Grad-CAM
        </span>
      </div>

      <input
        type="range"
        min={0}
        max={100}
        value={sliderPos}
        onChange={(e) => setSliderPos(Number(e.target.value))}
        aria-label="Deslizar entre imagen original y mapa de calor Grad-CAM"
        className="w-full mt-3 accent-teal-400 cursor-pointer"
      />

      <p className="text-xs text-ink-500 font-body mt-2 text-center">
        Desliza la lente para ver dónde se enfocó el modelo —{" "}
        <span className="text-teal-400 font-medium">{dominantRegion?.replace("_", " ")}</span> de la imagen.
      </p>
    </div>
  );
}
