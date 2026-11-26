import "@testing-library/jest-dom";

// Node 25+ has its own localStorage and sessionStorage globals, which are undefined without
// --localstorage-file and hide jsdom's. Restore jsdom's so tests behave the same on every Node.
const { jsdom } = globalThis as typeof globalThis & { jsdom?: { window: Window } };
if (jsdom) {
  for (const name of ["localStorage", "sessionStorage"] as const) {
    Object.defineProperty(globalThis, name, { configurable: true, get: () => jsdom.window[name] });
  }
}
