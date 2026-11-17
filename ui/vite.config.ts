import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The bundle is compiled into the binary and served from memory, so file names need no
// content hash, and stable names keep the committed dist/ diff readable.
export default defineConfig({
  plugins: [react()],
  build: {
    outDir: "dist",
    emptyOutDir: true,
    assetsDir: ".",
    rollupOptions: {
      output: {
        entryFileNames: "app.js",
        chunkFileNames: "[name].js",
        assetFileNames: "[name][extname]",
      },
    },
  },
  server: {
    // `npm run dev` against `ailocal-ui --port 7777 --no-open`, opened with that run's
    // `?t=` token.
    proxy: {
      "/api": {
        target: "http://127.0.0.1:7777",
        changeOrigin: true,
      },
    },
  },
});
