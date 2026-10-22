/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        // "Sectional": the palette of an aeronautical chart, not a dashboard.
        chart: {
          paper: "#EDF0EE",
          panel: "#F7F9F8",
          line: "#CBD4D2",
          rule: "#9FB0AE",
          ink: "#13242E",
          muted: "#5A6E74",
        },
        magenta: { DEFAULT: "#C4347A", soft: "#F3DCE8", deep: "#8E2256" },
        classb: { DEFAULT: "#2E6DA4", soft: "#DCE7F1", deep: "#1C4668" },
        terrain: { DEFAULT: "#D8C9A8", deep: "#A8966D" },
        caution: { DEFAULT: "#E0902F", soft: "#F7E6CE" },
        alert: { DEFAULT: "#B4442E" },
      },
      fontFamily: {
        display: ["Archivo", "Helvetica Neue", "Arial Narrow", "sans-serif"],
        sans: ["IBM Plex Sans", "Helvetica Neue", "system-ui", "sans-serif"],
        mono: ["IBM Plex Mono", "SFMono-Regular", "Menlo", "monospace"],
      },
      letterSpacing: { chart: "0.08em" },
      boxShadow: {
        panel: "0 1px 0 0 #CBD4D2, 0 1px 3px -1px rgba(19,36,46,0.08)",
      },
    },
  },
  plugins: [],
};
