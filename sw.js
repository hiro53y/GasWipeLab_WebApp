// Service Worker — GasWipeLab v2.2 オフラインキャッシュ
const CACHE = 'gaswipelab-v2.2';
const STATIC = [
  './',
  './index.html',
  './python/gaswipelab/__init__.py',
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
  './python/gaswipelab/services/settings_service.py',
  './python/gaswipelab/utils/__init__.py',
  './python/gaswipelab/utils/paths.py',
  './python/gaswipelab/utils/validation.py',
];

self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE).then(cache => cache.addAll(STATIC))
  );
});

self.addEventListener('activate', event => {
  event.waitUntil(
    caches.keys().then(keys =>
      Promise.all(keys.filter(k => k !== CACHE).map(k => caches.delete(k)))
    )
  );
});

// キャッシュファースト（Pyodide本体はCDNキャッシュに任せる）
self.addEventListener('fetch', event => {
  const url = event.request.url;
  // CDNリソース（Pyodide・Plotly・Tailwind）はネットワーク優先
  if (url.includes('cdn.') || url.includes('jsdelivr') || url.includes('pyodide')) {
    event.respondWith(
      fetch(event.request).catch(() => caches.match(event.request))
    );
    return;
  }
  // 自前ファイルはキャッシュファースト
  event.respondWith(
    caches.match(event.request).then(cached => cached || fetch(event.request))
  );
});
