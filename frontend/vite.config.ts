import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The production build is served by FastAPI from `frontend/dist` (single origin, no proxy).
// `npm run dev` proxies the API so the developer experience matches production.
export default defineConfig({
  plugins: [react()],
  envPrefix: ["VITE_", "NEXT_PUBLIC_"],
  server: {
    host: "0.0.0.0",
    port: 5173,
    strictPort: true,
    allowedHosts: true,
    proxy: {
      "/api": { target: "http://127.0.0.1:8000", changeOrigin: true },
      "/documentation": { target: "http://127.0.0.1:8000", changeOrigin: true },
      "/openapi.json": { target: "http://127.0.0.1:8000", changeOrigin: true },
    },
  },
  build: {
    outDir: "dist",
    chunkSizeWarningLimit: 6000,
    rollupOptions: {
      output: {
        manualChunks: { plotly: ["plotly.js-dist-min"] },
      },
    },
  },
});
