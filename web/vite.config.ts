import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// The data files live in the Python package (../src/amagi_inference_calc/data)
// so the CLI and the web page read the same numbers.
export default defineConfig({
  plugins: [react()],
  base: './',
  server: { fs: { allow: ['..'] } },
});
