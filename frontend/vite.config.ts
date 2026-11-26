import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    // Fail instead of moving to 5174: the API's default CORS_ALLOW_ORIGINS lists only :5173.
    strictPort: true,
  },
  build: {
    rollupOptions: {
      output: {
        manualChunks: {
          "react-vendor": ["react", "react-dom"],
          "markdown-vendor": ["react-markdown", "remark-gfm", "remark-math", "rehype-katex"],
          "icons-vendor": ["lucide-react"],
        },
      },
    },
  },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: ["./src/test/setup.ts"],
  },
});
