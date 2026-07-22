// Fully client-side sound-organizing pipeline: audio file -> CLAP embedding
// (in-browser, via parallel transformers.js/WASM workers) -> nearest-prototype classification
// -> per-section 2D layout. No server-side processing, no Python at runtime.
//
// Mirrors the shape of the old python-generated data.json ({nodes, sections}),
// so it plugs into the existing visualization in app.js unchanged.



import { UMAP } from 'https://cdn.jsdelivr.net/npm/umap-js@1/+esm';

const MODEL_ID = 'Xenova/clap-htsat-unfused';
const SR = 48000;              // CLAP's expected sample rate
const MAX_ONESHOT_SEC = 3.0;   // loops go down a different path, not handled here
const AUDIO_EXT = /\.(wav|aif|aiff|flac|mp3|ogg|m4a)$/i;

// Same palette as the old build_map.py, so section colors stay stable across
// the python-trained and browser-classified pipelines.
const COLORS = {
    Kick: '#ff5e62', '808': '#ff3f7f', Snare: '#a55eea', Clap: '#4b7bec',
    Rimshot: '#20bf6b', Snap: '#079992', 'Closed Hat': '#fd9644',
    'Open Hat': '#fed330', Ride: '#26de81', Crash: '#eb3b5a', Cymbal: '#fa8231',
    Tom: '#3867d6', Conga: '#8854d0', Cowbell: '#ffd23f',
    Shaker: '#45aaf2', Tambourine: '#2bcbba', Woodblock: '#a5b1c2',
    Vocal: '#d1ccc0', Unsorted: '#5a6472',
};

let _prototypes = null;

async function loadPrototypes(onProgress) {
    if (_prototypes) return _prototypes;
    onProgress?.('Loading classifier…');
    const res = await fetch('prototypes.json');
    _prototypes = await res.json();
    return _prototypes;
}

// ---- Hybrid local caching helper functions (IndexedDB with localStorage fallback) --

const DB_NAME = 'PulseMapCache';
const STORE_NAME = 'embeddings';
let _useLocalStorage = false;

let _dbPromise = null;
function getDB() {
    if (_dbPromise) return _dbPromise;
    _dbPromise = new Promise((resolve, reject) => {
        if (!window.indexedDB) {
            reject(new Error('IndexedDB not supported'));
            return;
        }
        const req = indexedDB.open(DB_NAME, 1);
        req.onupgradeneeded = (e) => {
            const db = e.target.result;
            if (!db.objectStoreNames.contains(STORE_NAME)) {
                db.createObjectStore(STORE_NAME);
            }
        };
        req.onsuccess = (e) => resolve(e.target.result);
        req.onerror = (e) => {
            _dbPromise = null; // Reset so next calls can retry
            reject(e.target.error);
        };
    });
    return _dbPromise;
}

async function getCachedEmbedding(fingerprint) {
    if (_useLocalStorage) {
        try {
            const val = localStorage.getItem('pm_' + fingerprint);
            return val ? JSON.parse(val) : null;
        } catch (e) {
            return null;
        }
    }
    try {
        const db = await getDB();
        return new Promise((resolve, reject) => {
            const tx = db.transaction(STORE_NAME, 'readonly');
            const store = tx.objectStore(STORE_NAME);
            const req = store.get(fingerprint);
            req.onsuccess = () => resolve(req.result || null);
            req.onerror = () => reject(req.error);
        });
    } catch (e) {
        console.warn('IndexedDB read failed, falling back to localStorage:', e);
        _useLocalStorage = true;
        try {
            const val = localStorage.getItem('pm_' + fingerprint);
            return val ? JSON.parse(val) : null;
        } catch (err) {
            return null;
        }
    }
}

async function cacheEmbedding(fingerprint, data) {
    if (_useLocalStorage) {
        try {
            localStorage.setItem('pm_' + fingerprint, JSON.stringify(data));
        } catch (e) {
            // QuotaExceededError or security blocked
        }
        return;
    }
    try {
        const db = await getDB();
        return new Promise((resolve, reject) => {
            const tx = db.transaction(STORE_NAME, 'readwrite');
            const store = tx.objectStore(STORE_NAME);
            const req = store.put(data, fingerprint);
            req.onsuccess = () => resolve();
            req.onerror = () => reject(req.error);
        });
    } catch (e) {
        console.warn('IndexedDB write failed, falling back to localStorage:', e);
        _useLocalStorage = true;
        try {
            localStorage.setItem('pm_' + fingerprint, JSON.stringify(data));
        } catch (err) {
            // ignore
        }
    }
}

// ---- audio decode + preprocessing (Web Audio API replacement for librosa) --

let _decodeCtx = null;
function decodeCtx() {
    if (!_decodeCtx) _decodeCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: SR });
    return _decodeCtx;
}

async function decodeToFloat32Mono(file) {
    const buf = await file.arrayBuffer();
    const decoded = await decodeCtx().decodeAudioData(buf); // already at 48kHz (context rate)
    if (decoded.numberOfChannels === 1) return decoded.getChannelData(0);
    // Downmix to mono by averaging channels.
    const n = decoded.length, out = new Float32Array(n);
    for (let ch = 0; ch < decoded.numberOfChannels; ch++) {
        const data = decoded.getChannelData(ch);
        for (let i = 0; i < n; i++) out[i] += data[i] / decoded.numberOfChannels;
    }
    return out;
}

/** Trim leading/trailing near-silence (mirrors librosa.effects.trim(top_db=30)). */
function trimSilence(samples, topDb = 30, frameSize = 1024) {
    const n = samples.length;
    if (n === 0) return samples;
    let peak = 0;
    for (let i = 0; i < n; i++) peak = Math.max(peak, Math.abs(samples[i]));
    if (peak === 0) return samples;
    const threshold = peak * Math.pow(10, -topDb / 20);

    let start = 0, end = n;
    for (let i = 0; i < n; i += frameSize) {
        let rms = 0, cnt = 0;
        for (let j = i; j < Math.min(i + frameSize, n); j++, cnt++) rms += samples[j] * samples[j];
        rms = Math.sqrt(rms / Math.max(cnt, 1));
        if (rms > threshold) { start = i; break; }
    }
    for (let i = n; i > 0; i -= frameSize) {
        const s = Math.max(0, i - frameSize);
        let rms = 0, cnt = 0;
        for (let j = s; j < i; j++, cnt++) rms += samples[j] * samples[j];
        rms = Math.sqrt(rms / Math.max(cnt, 1));
        if (rms > threshold) { end = i; break; }
    }
    return start < end ? samples.slice(start, end) : samples;
}

// ---- layout (mirrors build_map.py) -----------------------------------------

function makeBiquad(type, freq, q, sr) {
    const w0 = 2 * Math.PI * freq / sr;
    const alpha = Math.sin(w0) / (2 * q);
    const cosw0 = Math.cos(w0);
    
    let b0, b1, b2, a0, a1, a2;
    
    if (type === 'bandpass') {
        b0 = Math.sin(w0) / 2;
        b1 = 0;
        b2 = -b0;
        a0 = 1 + alpha;
        a1 = -2 * cosw0;
        a2 = 1 - alpha;
    } else if (type === 'lowpass') {
        b0 = (1 - cosw0) / 2;
        b1 = 1 - cosw0;
        b2 = b0;
        a0 = 1 + alpha;
        a1 = -2 * cosw0;
        a2 = 1 - alpha;
    } else if (type === 'highpass') {
        b0 = (1 + cosw0) / 2;
        b1 = -(1 + cosw0);
        b2 = b0;
        a0 = 1 + alpha;
        a1 = -2 * cosw0;
        a2 = 1 - alpha;
    }
    
    const a0_inv = 1 / a0;
    return {
        b0: b0 * a0_inv, b1: b1 * a0_inv, b2: b2 * a0_inv,
        a1: a1 * a0_inv, a2: a2 * a0_inv,
        x1: 0, x2: 0, y1: 0, y2: 0,
        process(x) {
            const y = this.b0 * x + this.b1 * this.x1 + this.b2 * this.x2 - this.a1 * this.y1 - this.a2 * this.y2;
            this.x2 = this.x1; this.x1 = x;
            this.y2 = this.y1; this.y1 = y;
            return y;
        }
    };
}

function extractAcousticFeatures(samples) {
    const n = samples.length;
    if (n === 0) return Array(13).fill(0);
    
    // Create biquad filters
    const lpSub = makeBiquad('lowpass', 60, 0.7, SR);
    const bpBass = makeBiquad('bandpass', 150, 0.7, SR);
    const bpMid = makeBiquad('bandpass', 1000, 0.7, SR);
    const hpHigh = makeBiquad('highpass', 2500, 0.7, SR);
    const lpPitch = makeBiquad('lowpass', 150, 0.7, SR);
    
    let peak = 0;
    let sumSq = 0;
    let timeSumSq = 0;
    
    let subSumSq = 0;
    let bassSumSq = 0;
    let midSumSq = 0;
    let highSumSq = 0;
    
    let crossings = 0;
    let highCrossings = 0;
    let pitchCrossings = 0;
    
    let lastVal = 0;
    let lastHigh = 0;
    let lastPitch = 0;
    
    for (let i = 0; i < n; i++) {
        const x = samples[i];
        const absX = Math.abs(x);
        if (absX > peak) peak = absX;
        
        const xSq = x * x;
        sumSq += xSq;
        timeSumSq += i * xSq;
        
        const ySub = lpSub.process(x);
        subSumSq += ySub * ySub;
        
        const yBass = bpBass.process(x);
        bassSumSq += yBass * yBass;
        
        const yMid = bpMid.process(x);
        midSumSq += yMid * yMid;
        
        const yHigh = hpHigh.process(x);
        highSumSq += yHigh * yHigh;
        
        const yPitch = lpPitch.process(x);
        
        if (i > 0 && ((x >= 0 && lastVal < 0) || (x < 0 && lastVal >= 0))) {
            crossings++;
        }
        lastVal = x;
        
        if (i > 0 && ((yHigh >= 0 && lastHigh < 0) || (yHigh < 0 && lastHigh >= 0))) {
            highCrossings++;
        }
        lastHigh = yHigh;
        
        if (i > 0 && ((yPitch >= 0 && lastPitch < 0) || (yPitch < 0 && lastPitch >= 0))) {
            pitchCrossings++;
        }
        lastPitch = yPitch;
    }
    
    const rms = Math.sqrt(sumSq / n || 1);
    const crestFactor = peak / (rms || 1e-4);
    const temporalCentroid = (timeSumSq / (sumSq || 1)) / n;
    
    const getRms = (start, end) => {
        if (start >= n) return 0;
        const stop = Math.min(end, n);
        let sum = 0;
        for (let i = start; i < stop; i++) sum += samples[i] * samples[i];
        return Math.sqrt(sum / (stop - start || 1));
    };
    
    // temporal decay windows
    const rms1 = getRms(0, 960);
    const rms2 = getRms(960, 3840);
    const rms3 = getRms(3840, 14400);
    const rms4 = getRms(14400, 48000);
    
    const subBassRms = Math.sqrt(subSumSq / n || 1);
    const bassRms = Math.sqrt(bassSumSq / n || 1);
    const midRms = Math.sqrt(midSumSq / n || 1);
    const highRms = Math.sqrt(highSumSq / n || 1);
    
    const grit = crossings / n;
    const brightness = highCrossings / n;
    
    const pitchRate = pitchCrossings / n;
    const fundamentalFreq = Math.min(200, (pitchRate * SR) / 2);
    
    return [
        temporalCentroid,
        crestFactor,
        rms1, rms2, rms3, rms4,
        subBassRms, bassRms, midRms, highRms,
        grit, brightness,
        fundamentalFreq
    ];
}

// Deterministic PRNG so a section's UMAP layout is stable across reloads
// (mirrors build_map.py's random_state=42). umap-js takes a [0,1) source.
function mulberry32(seed) {
    return function () {
        seed |= 0; seed = (seed + 0x6d2b79f5) | 0;
        let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
        t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}

function localLayout(embeddings) {
    const n = embeddings.length;
    if (n === 1) return [[0.5, 0.5]];
    if (n <= 6) {  // too few for UMAP -> circle
        return embeddings.map((_, i) => {
            const ang = (2 * Math.PI * i) / n;
            return [0.5 + 0.35 * Math.cos(ang), 0.5 + 0.35 * Math.sin(ang)];
        });
    }

    // Neighbor-preserving 2D embedding via real UMAP -- a faithful port of
    // build_map.py's local_layout. Unlike linear PCA (which only captures the
    // two highest-variance directions and leaves near-neighbors scattered),
    // UMAP explicitly pulls acoustically-similar sounds together and pushes
    // dissimilar ones apart, so what sits next to a sound actually sounds like it.
    // Lower nEpochs (default ~200) for speed on slower machines; 100 epochs is
    // still enough to resolve local neighborhoods on typical section sizes.
    const umap = new UMAP({
        nComponents: 2,
        nNeighbors: Math.min(15, n - 1),
        minDist: 0.12,
        nEpochs: 100,
        random: mulberry32(42),
    });
    const xy = umap.fit(embeddings);

    // Normalise into the unit box (same as build_map.py).
    const xs = xy.map(p => p[0]), ys = xy.map(p => p[1]);
    const minX = Math.min(...xs), maxX = Math.max(...xs);
    const minY = Math.min(...ys), maxY = Math.max(...ys);
    const spanX = (maxX - minX) || 1, spanY = (maxY - minY) || 1;

    return xy.map(([x, y]) => [
        (x - minX) / spanX,
        (y - minY) / spanY
    ]);
}

function pca1D(vectors) {
    // 1-component PCA via power iteration -- enough to order a handful of
    // section centroids along their dominant axis (mirrors np.linalg.svd usage
    // in build_map.py, without needing a full linear-algebra library).
    const n = vectors.length, dim = vectors[0].length;
    const mean = new Array(dim).fill(0);
    for (const v of vectors) for (let i = 0; i < dim; i++) mean[i] += v[i] / n;
    const centered = vectors.map(v => v.map((x, i) => x - mean[i]));
    let vec = new Array(dim).fill(0).map(() => Math.random());
    for (let iter = 0; iter < 50; iter++) {
        const next = new Array(dim).fill(0);
        for (const c of centered) {
            const dot = c.reduce((s, x, i) => s + x * vec[i], 0);
            for (let i = 0; i < dim; i++) next[i] += dot * c[i];
        }
        const norm = Math.sqrt(next.reduce((s, x) => s + x * x, 0)) || 1;
        vec = next.map(x => x / norm);
    }
    return centered.map(c => c.reduce((s, x, i) => s + x * vec[i], 0));
}

// Worker code content template
const WORKER_CODE = `
/* worker import */ import { AutoProcessor, ClapAudioModelWithProjection, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3';

let _processor = null;
let _model = null;

function scaleVec(vec, mean, scale) {
    const out = new Float64Array(vec.length);
    for (let i = 0; i < vec.length; i++) out[i] = (vec[i] - mean[i]) / scale[i];
    return out;
}

function euclidean(a, b) {
    let s = 0;
    for (let i = 0; i < a.length; i++) { const d = a[i] - b[i]; s += d * d; }
    return Math.sqrt(s);
}

const MARGIN_UNSORTED = 0.20;

function classify(embedding, proto) {
    const scaled = scaleVec(embedding, proto.scaler_mean, proto.scaler_scale);
    const bestPerClass = {};
    for (const p of proto.prototypes) {
        const d = euclidean(scaled, p.vec);
        if (!(p.label in bestPerClass) || d < bestPerClass[p.label]) bestPerClass[p.label] = d;
    }
    const ranked = Object.entries(bestPerClass).sort((a, b) => a[1] - b[1]);
    const [bestLabel, bestDist] = ranked[0];
    const secondDist = ranked[1] ? ranked[1][1] : bestDist + 999;
    const margin = secondDist - bestDist;
    const confidence = 1 / (1 + Math.exp(-(margin - 1)));
    const section = margin >= MARGIN_UNSORTED ? bestLabel : 'Unsorted';
    const tags = ranked.slice(0, 3).map(([label, dist]) => [label, +(1 / (1 + dist)).toFixed(3)]);
    return { section, confidence, tags };
}

self.onmessage = async (e) => {
    const { type, id, samples, proto, modelId } = e.data;
    try {
        if (type === 'init') {
            const onnxEnv = env.backends?.onnx || env.onnx;
            if (onnxEnv && onnxEnv.wasm) {
                onnxEnv.wasm.numThreads = 1;
            }
            
            _processor = await AutoProcessor.from_pretrained(modelId);
            _model = await ClapAudioModelWithProjection.from_pretrained(modelId, { dtype: 'q8' });
            self.postMessage({ type: 'ready' });
        } else if (type === 'process') {
            const inputs = await _processor(samples);
            const { audio_embeds } = await _model(inputs);
            const emb = Array.from(audio_embeds.data);
            const norm = Math.sqrt(emb.reduce((s, x) => s + x * x, 0)) || 1;
            const normalizedEmb = emb.map(x => x / norm);
            const classification = classify(normalizedEmb, proto);
            self.postMessage({ type: 'result', id, embedding: normalizedEmb, ...classification });
        }
    } catch (err) {
        self.postMessage({ type: 'error', id, error: err.message });
    }
};
`;

// ---- main entry point -------------------------------------------------------

/**
 * Build the sectioned map from a FileList/array of File objects.
 * onProgress(msg, done, total) is called throughout for UI feedback.
 */
export async function buildMap(files, onProgress) {
    const audioFiles = Array.from(files).filter(f => AUDIO_EXT.test(f.name));
    onProgress?.(`Found ${audioFiles.length} sounds`, 0, audioFiles.length);

    const proto = await loadPrototypes(onProgress);

    // 1. Check local IndexedDB cache for all files in parallel
    onProgress?.('Checking cache…');
    const cachedResults = [];
    const todoFiles = [];
    
    await Promise.all(audioFiles.map(async (file) => {
        const fingerprint = `${file.name}|${file.size}|${file.lastModified}`;
        const cached = await getCachedEmbedding(fingerprint);
        if (cached) {
            cachedResults.push({
                file,
                path: file.webkitRelativePath || file.name,
                filename: file.name,
                ...cached
            });
        } else {
            todoFiles.push(file);
        }
    }));

    let results = [...cachedResults];

    // 2. Only spawn threads if we have new sounds that aren't cached
    if (todoFiles.length > 0) {
        // Dynamically scale threads to maximize CPU utilization based on available cores
        const cores = navigator.hardwareConcurrency || 2;
        const NUM_WORKERS = Math.max(2, cores > 4 ? cores - 2 : cores - 1);
        onProgress?.(`Initializing ${NUM_WORKERS} threads for ${todoFiles.length} new sounds…`);
        
        const blob = new Blob([WORKER_CODE], { type: 'application/javascript' });
        const workerUrl = URL.createObjectURL(blob);
        const workers = [];
        
        let nextFileIdx = 0;
        let doneCount = cachedResults.length;
        onProgress?.('Analyzing sounds…', doneCount, audioFiles.length);

        // The worker processing loop: dynamically pulls the next file from the queue
        async function runTask(worker) {
            const fileIdx = nextFileIdx++;
            if (fileIdx >= todoFiles.length) return;
            
            const file = todoFiles[fileIdx];
            let prepared = null;
            let acousticFeatures = null;
            try {
                let samples = await decodeToFloat32Mono(file);
                samples = trimSilence(samples);
                const durationSec = samples.length / SR;
                if (durationSec > 0 && durationSec <= MAX_ONESHOT_SEC) {
                    acousticFeatures = extractAcousticFeatures(samples);
                    prepared = samples;
                }
            } catch (e) {
                console.warn(`Skipping ${file.name}:`, e);
            }
            
            if (prepared) {
                await new Promise((resolve) => {
                    worker.onmessage = async (e) => {
                        if (e.data.type === 'result') {
                            const { embedding, section, confidence, tags } = e.data;
                            const resultItem = { file, path: file.webkitRelativePath || file.name, filename: file.name,
                                embedding, section, confidence, tags, acousticFeatures };
                            results.push(resultItem);
                            
                            // Write to local cache asynchronously in the background
                            const fingerprint = `${file.name}|${file.size}|${file.lastModified}`;
                            cacheEmbedding(fingerprint, { embedding, section, confidence, tags, acousticFeatures });
                            
                            doneCount++;
                            onProgress?.('Analyzing sounds…', doneCount, audioFiles.length);
                            resolve();
                        } else if (e.data.type === 'error') {
                            console.warn(`Worker error for ${file.name}:`, e.data.error);
                            doneCount++;
                            onProgress?.('Analyzing sounds…', doneCount, audioFiles.length);
                            resolve();
                        }
                    };
                    worker.postMessage({ type: 'process', id: fileIdx, samples: prepared, proto }, [prepared.buffer]);
                });
            } else {
                doneCount++;
                onProgress?.('Analyzing sounds…', doneCount, audioFiles.length);
            }
            
            // Loop back to fetch next file in line
            await runTask(worker);
        }

        const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));
        const staggerDelay = Math.max(250, Math.round(2000 / NUM_WORKERS));

        // Stagger worker creation and initialization.
        // Workers begin processing IMMEDIATELY as they boot, no blocking barrier.
        const workerPromises = [];
        for (let wIdx = 0; wIdx < NUM_WORKERS; wIdx++) {
            if (wIdx > 0) {
                await sleep(staggerDelay); // Space out worker instantiation to prevent IndexedDB/WASM compiler lockups
            }
            
            const worker = new Worker(workerUrl, { type: 'module' });
            workers.push(worker);
            
            const p = new Promise((resolve, reject) => {
                worker.onmessage = (e) => {
                    if (e.data.type === 'ready') {
                        // Thread is loaded, begin processing files right away
                        runTask(worker).then(resolve);
                    } else if (e.data.type === 'error') {
                        reject(new Error(e.data.error));
                    }
                };
                worker.postMessage({ type: 'init', modelId: MODEL_ID });
            });
            workerPromises.push(p);
        }

        await Promise.all(workerPromises);

        // Clean up worker instances and URLs
        workers.forEach(w => w.terminate());
        URL.revokeObjectURL(workerUrl);
    } else {
        // Fast path: everything loaded from cache, update progress to 100%
        onProgress?.('Analyzing sounds…', audioFiles.length, audioFiles.length);
    }

    if (results.length === 0) return { nodes: [], sections: [] };

    // Group by section, local layout within each.
    const bySection = {};
    for (const r of results) (bySection[r.section] ||= []).push(r);

    const order = Object.keys(bySection).filter(s => s !== 'Unsorted');
    if (bySection.Unsorted) order.push('Unsorted');

    // Order sections along their dominant embedding axis, like build_map.py.
    const centroids = order.map(sec => {
        const items = bySection[sec];
        const dim = items[0].embedding.length;
        const c = new Array(dim).fill(0);
        for (const it of items) for (let i = 0; i < dim; i++) c[i] += it.embedding[i] / items.length;
        return c;
    });
    let sortedOrder = order;
    if (order.length > 2) {
        const proj = pca1D(centroids);
        sortedOrder = order.map((s, i) => [proj[i], s]).sort((a, b) => a[0] - b[0]).map(p => p[1]);
        if (bySection.Unsorted) {
            sortedOrder = sortedOrder.filter(s => s !== 'Unsorted');
            sortedOrder.push('Unsorted');
        }
    }

    const maxCount = Math.max(...sortedOrder.map(s => bySection[s].length));
    const maxPad = Math.max(22, Math.round(7 * Math.sqrt(maxCount)));
    const GAP = 56;
    const CELL = maxPad + GAP;
    const cols = Math.ceil(Math.sqrt(sortedOrder.length));
    const nodes = [];
    const sectionMeta = [];

    sortedOrder.forEach((sec, idx) => {
        const items = bySection[sec];
        const ox = (idx % cols) * CELL, oy = Math.floor(idx / cols) * CELL;
        // Neighbor-preserving 2D layout (UMAP) over the CLAP embeddings -- the
        // perceptual-similarity space -- so acoustically-similar sounds land next
        // to each other (matches build_map.py). Acoustic features are far too
        // coarse (13 dims) to capture "sounds alike".
        const xy = localLayout(items.map(it => it.embedding));

        // Scale this section's pad size based on its own count
        const sectionPad = Math.max(22, Math.round(7 * Math.sqrt(items.length)));
        // Center the section within its CELL box
        const shiftX = (maxPad - sectionPad) / 2;
        const shiftY = (maxPad - sectionPad) / 2;

        items.forEach((it, j) => {
            const gx = ox + shiftX + xy[j][0] * sectionPad, gy = oy + shiftY + xy[j][1] * sectionPad;
            nodes.push({
                path: it.path, filename: it.filename, section: sec,
                confidence: it.confidence, tags: it.tags,
                x: gx, y: gy, color: COLORS[sec] || '#888888',
                _file: it.file, // kept for local audio playback (object URL)
                relX: xy[j][0], // save local relative X for gradient coloring
                relY: xy[j][1], // save local relative Y
            });
        });
        const cx = nodes.filter(n => n.section === sec).reduce((s, n) => s + n.x, 0) / items.length;
        const cy = nodes.filter(n => n.section === sec).reduce((s, n) => s + n.y, 0) / items.length;
        sectionMeta.push({ name: sec, count: items.length, color: COLORS[sec] || '#888888', cx, cy });
    });
    sectionMeta.sort((a, b) => b.count - a.count);

    return { nodes, sections: sectionMeta };
}

export async function clearAllCaches() {
    // Clear localStorage pm_ keys, corrections, last_result
    for (let i = localStorage.length - 1; i >= 0; i--) {
        const key = localStorage.key(i);
        if (key && (key.startsWith('pm_') || key === 'pulsemap_corrections' || key === 'pulsemap_last_result')) {
            localStorage.removeItem(key);
        }
    }
    // Delete IndexedDB database
    if (window.indexedDB) {
        return new Promise((resolve) => {
            const req = indexedDB.deleteDatabase(DB_NAME);
            req.onsuccess = () => resolve();
            req.onerror = () => resolve();
            req.onblocked = () => resolve();
        });
    }
}


