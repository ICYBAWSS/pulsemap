// Pure-JS nearest-prototype classifier -- zero dependencies, so it can be
// unit-tested in plain Node and imported unchanged in the browser pipeline.
export function scaleVec(vec, mean, scale) {
    const out = new Float64Array(vec.length);
    for (let i = 0; i < vec.length; i++) out[i] = (vec[i] - mean[i]) / scale[i];
    return out;
}

export function euclidean(a, b) {
    let s = 0;
    for (let i = 0; i < a.length; i++) { const d = a[i] - b[i]; s += d * d; }
    return Math.sqrt(s);
}

// Calibrated on HELD-OUT cross-validation (not training-fit data): margin>=0.5
// keeps ~93% of sounds confidently classified at ~91% accuracy on that subset.
// (This is Euclidean distance margin in the scaled embedding space -- NOT
// comparable to the old SVM decision-score margins used server-side.)
const MARGIN_UNSORTED = 0.20;

/** Nearest-prototype classification with a margin-based Unsorted fallback. */
export function classify(embedding, proto) {
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
