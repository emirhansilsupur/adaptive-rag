import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import { openThreadStore } from "./lib/threadStore";
import "./index.css";

// Saved threads load before the first render; opening the store never fails.
void openThreadStore().then((threadStore) => {
  createRoot(document.getElementById("root")!).render(
    <StrictMode>
      <App threadStore={threadStore} />
    </StrictMode>
  );
});
