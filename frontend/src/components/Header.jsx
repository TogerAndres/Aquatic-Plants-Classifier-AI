export default function Header() {
  return (
    <header className="flex flex-col items-center gap-3 pt-14 pb-10 px-4 text-center">
      <span className="font-mono text-xs tracking-[0.2em] uppercase text-teal-400 border border-teal-400/30 rounded-full px-3 py-1">
        EfficientNetB0 · Validación cruzada · Grad-CAM
      </span>
      <h1 className="font-display text-3xl sm:text-5xl font-700 tracking-tight text-ink-100 max-w-2xl">
        Clasificador de{" "}
        <span className="bg-gradient-to-r from-violet-400 to-teal-400 bg-clip-text text-transparent">
          Plantas Acuáticas
        </span>
      </h1>
      <p className="font-body text-ink-300 max-w-xl text-sm sm:text-base">
        Sube una foto de Lemna minor, Eichornia crassipes, Monochoria korsakowii
        o Pistia stratiotes. El modelo no solo predice la especie: muestra
        exactamente en qué parte de la imagen se fijó para decidir.
      </p>
    </header>
  );
}
