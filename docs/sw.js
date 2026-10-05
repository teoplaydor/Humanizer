// App-shell cache: the app opens offline. Model weights are cached separately by transformers.js
// (Cache Storage "transformers-cache") after the first download.
const VERSION = "humanizer-v1";
const SHELL = ["./", "index.html", "app.js", "core.js", "manifest.webmanifest", "icon.svg", "icon-192.png", "icon-512.png",
  "vendor/transformers.min.js", "vendor/ort-wasm-simd-threaded.asyncify.mjs", "vendor/ort-wasm-simd-threaded.asyncify.wasm"];
self.addEventListener("install", e => { e.waitUntil(caches.open(VERSION).then(c => c.addAll(SHELL)).then(() => self.skipWaiting())); });
self.addEventListener("activate", e => {
  e.waitUntil(caches.keys().then(keys => Promise.all(keys.filter(k => k.startsWith("humanizer-") && k !== VERSION).map(k => caches.delete(k)))).then(() => self.clients.claim()));
});
self.addEventListener("fetch", e => {
  const url = new URL(e.request.url);
  if (e.request.method !== "GET" || url.origin !== location.origin) return;   // model files: handled by transformers.js cache
  e.respondWith(caches.match(e.request, { ignoreSearch: true }).then(hit => hit || fetch(e.request)));
});
