/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        fraud: { 50: '#fff1f2', 500: '#f43f5e', 900: '#881337' },
        safe: { 50: '#f0fdf4', 500: '#22c55e', 900: '#14532d' },
        warn: { 50: '#fffbeb', 500: '#f59e0b', 900: '#78350f' },
      },
    },
  },
  plugins: [],
}
