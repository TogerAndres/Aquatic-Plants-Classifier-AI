/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        abyss: "#0A0E1A",      // fondo base, navy casi negro
        surface: "#121A2E",    // tarjetas
        "surface-2": "#1B2640",
        violet: {
          400: "#9B8CFF",
          500: "#7C5CFC",
          600: "#6240E0",
        },
        teal: {
          300: "#6EE7D8",
          400: "#2DD4BF",
          500: "#14B8A6",
        },
        ink: {
          100: "#EDEFF7",
          300: "#B6BBD0",
          500: "#7E859C",
        },
      },
      fontFamily: {
        display: ["Space Grotesk", "sans-serif"],
        body: ["Inter", "sans-serif"],
        mono: ["JetBrains Mono", "monospace"],
      },
      boxShadow: {
        glow: "0 0 40px -10px rgba(124, 92, 252, 0.45)",
        "glow-teal": "0 0 40px -10px rgba(45, 212, 191, 0.4)",
      },
      backgroundImage: {
        "aurora": "radial-gradient(60% 50% at 20% 0%, rgba(124,92,252,0.25) 0%, transparent 60%), radial-gradient(50% 40% at 90% 10%, rgba(45,212,191,0.18) 0%, transparent 60%)",
      },
    },
  },
  plugins: [],
};
