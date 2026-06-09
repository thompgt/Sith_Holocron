/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        sith: {
          red: '#E50914',
          glow: 'rgba(229, 9, 20, 0.6)',
          obsidian: '#0B0C10',
          charcoal: '#1F2833',
          steel: '#C5C6C7',
        }
      }
    },
  },
  plugins: [],
}
