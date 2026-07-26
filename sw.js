// Service Worker — GasWipeLab v3.0 オフラインキャッシュ
const CACHE = 'gaswipelab-v3.0';
const STATIC = [
  './',
  './index.html',
  './manifest.json',
  './python/gaswipelab/__init__.py',
  './python/gaswipelab/web_api.py',
  './python/gaswipelab/models/__init__.py',
  './python/gaswipelab/models/calibration_model.py',
  './python/gaswipelab/models/coating_weight.py',
  './python/gaswipelab/models/film_model.py',
  './python/gaswipelab/models/gas_properties.py',
  './python/gaswipelab/models/jet_impingement.py',
  './python/gaswipelab/models/nozzle_model.py',
  './python/gaswipelab/models/splash_risk.py',
  './python/gaswipelab/models/units.py',
  './python/gaswipelab/models/zinc_properties.py',
  './python/gaswipelab/services/__init__.py',
  './python/gaswipelab/services/analysis_service.py',
  './python/gaswipelab/services/calibration_service.py',
  './python/gaswipelab/services/csv_service.py',
  './python/gaswipelab/services/design_service.py',
  './python/gaswipelab/services/settings_service.py',
  './python/gaswipelab/utils/__init__.py',
  './python/gaswipelab/utils/paths.py',
  './python/gaswipelab/utils/validation.py',
];

self.addEventListener('install', event => {
  self.skipWaiting();
  event.waitUntil(
    caches.open(CACHE).then(cache => cache.addAll(STATIC))
  );
});

self.addEventListener('activate', event => {
  event.waitUntil(
    caches.keys()
      .then(keys => Promise.all(keys.filter(k => k !== CACHE).map(k => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

// 自前ファイルはネットワーク優先（更新を確実に取り込み、オフライン時はキャッシュ）。
// CDN（Pyodide・Plotly）はネットワーク優先でブラウザキャッシュに任せる。
self.addEventListener('fetch', event => {
  if (event.request.method !== 'GET') return;
  const url = event.request.url;
  if (url.includes('cdn.') || url.includes('jsdelivr') || url.includes('pyodide')) {
    event.respondWith(fetch(event.request).catch(() => caches.match(event.request)));
    return;
  }
  event.respondWith(
    fetch(event.request)
      .then(response => {
        const copy = response.clone();
        caches.open(CACHE).then(cache => cache.put(event.request, copy)).catch(() => {});
        return response;
      })
      .catch(() => caches.match(event.request))
  );
});
