import { buildMap, clearAllCaches } from './pipeline.js';

const container = document.getElementById('graph-container');
const loadingOverlay = document.getElementById('loading');
const hoverBanner = document.getElementById('hoverBanner');

// --- audition (the core interaction) ----------------------------------------
// Sounds are local File objects (from a folder drop/pick), never uploaded
// anywhere -- played via an in-memory object URL, cached per node so hovering
// the same sound repeatedly doesn't re-allocate a URL every time.
let currentAudio = null;

// --- showNow toast notification ----------------------------------------------
function showNow(message) {
    let el = document.getElementById('toast');
    if (!el) {
        el = document.createElement('div');
        el.id = 'toast';
        el.style.cssText = 'position:fixed;bottom:20px;right:20px;background:#3d6bff;color:#fff;padding:8px 16px;border-radius:8px;font-size:13px;font-weight:600;z-index:9999;box-shadow:0 4px 12px rgba(0,0,0,0.3);transition:opacity 0.3s;pointer-events:none;';
        document.body.appendChild(el);
    }
    el.textContent = message;
    el.style.opacity = '1';
    clearTimeout(window.__toastTimeout);
    window.__toastTimeout = setTimeout(() => { el.style.opacity = '0'; }, 3000);
}

function playAudio(node) {
    if (currentAudio) { currentAudio.pause(); currentAudio.currentTime = 0; }
    if (!node) return;
    if (!node._file && !node.__objUrl) {
        showNow("⚠️ Drop folder again to preview sound");
        return;
    }
    if (!node.__objUrl) node.__objUrl = URL.createObjectURL(node._file);
    currentAudio = new Audio(node.__objUrl);
    currentAudio.play().catch(() => {});
}

function stopAudio() {
    if (currentAudio) { currentAudio.pause(); currentAudio.currentTime = 0; }
}

// Big top banner: always-on DOM text (never fights with canvas rendering the
// way in-canvas labels did — those blocked/overlapped the nodes). Just mirrors
// whatever's currently hovered; idle text otherwise.
function showHoverBanner(node) {
    if (node) {
        hoverBanner.classList.remove('idle');
        hoverBanner.innerHTML =
            `${node.filename} · <span class="sec" style="color:${node.color}">${node.section}</span> ${Math.round(node.confidence * 100)}%`;
    } else {
        hoverBanner.classList.add('idle');
        hoverBanner.textContent = 'Hover a sound to preview';
    }
}

// --- interaction state ------------------------------------------------------
let query = '';
let hovered = null;
let activeSectionHover = null;

// --- intro reveal: grey -> fly to home -> colorize on arrival ---------------
const GREY = '#5b6472';
function hexToRgb(h) {
    h = h.replace('#', '');
    return [parseInt(h.slice(0, 2), 16), parseInt(h.slice(2, 4), 16), parseInt(h.slice(4, 6), 16)];
}
function lerpColor(a, b, t) {
    const pa = hexToRgb(a), pb = hexToRgb(b);
    const m = i => Math.round(pa[i] + (pb[i] - pa[i]) * t);
    return `rgb(${m(0)},${m(1)},${m(2)})`;
}

function hexToHsl(hex) {
    hex = hex.replace('#', '');
    let r = parseInt(hex.slice(0, 2), 16) / 255;
    let g = parseInt(hex.slice(2, 4), 16) / 255;
    let b = parseInt(hex.slice(4, 6), 16) / 255;
    let max = Math.max(r, g, b), min = Math.min(r, g, b);
    let h = 0, s = 0, l = (max + min) / 2;
    if (max !== min) {
        let d = max - min;
        s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
        switch (max) {
            case r: h = (g - b) / d + (g < b ? 6 : 0); break;
            case g: h = (b - r) / d + 2; break;
            case b: h = (r - g) / d + 4; break;
        }
        h /= 6;
    }
    return [Math.round(h * 360), Math.round(s * 100), Math.round(l * 100)];
}
function matches(node) {
    if (!query) return true;
    const q = query.toLowerCase();
    if (node.filename.toLowerCase().includes(q)) return true;
    if (node.section.toLowerCase().includes(q)) return true;
    return node.tags.some(t => t[0].toLowerCase().includes(q));
}

function saveMapToCache(data) {
    try {
        const cleanNodes = data.nodes.map(n => ({
            path: n.path,
            filename: n.filename,
            section: n.section,
            confidence: n.confidence,
            tags: n.tags,
            x: n.x,
            y: n.y,
            color: n.color,
            relX: n.relX,
            relY: n.relY
        }));
        const cacheData = {
            nodes: cleanNodes,
            sections: data.sections
        };
        localStorage.setItem('pulsemap_last_result', JSON.stringify(cacheData));
    } catch (e) {
        console.warn('Failed to cache map layout:', e);
    }
}

// Renders the sectioned canvas from an in-memory {nodes, sections} result
// (built entirely client-side by pipeline.js -- no server round-trip).
function renderMap(data) {
    window.__introEndedTriggered = false;
    document.getElementById('drop').classList.add('hidden');
    saveMapToCache(data);
    loadingOverlay.classList.remove('hidden');
    // Intro: all sounds begin grey, gathered in the center; the sim then flings
    // them out to their section homes and each colorizes on arrival.
    const gcx = data.nodes.reduce((s, n) => s + n.x, 0) / data.nodes.length;
    const gcy = data.nodes.reduce((s, n) => s + n.y, 0) / data.nodes.length;
    const nodes = data.nodes.map(n => {
        const sx = gcx + (Math.random() - 0.5) * 24, sy = gcy + (Math.random() - 0.5) * 24;
        return { ...n, id: n.path, x: sx, y: sy, homeX: n.x, homeY: n.y, __colorT: 0,
            __introStartX: sx, __introStartY: sy };
    });
    const sections = data.sections;

    // Section anchors (fixed) + colors, for drag-drop targeting and recoloring.
    const secCentroid = {}, secColor = {};
    sections.forEach(s => { secCentroid[s.name] = [s.cx, s.cy]; secColor[s.name] = s.color; });

    // User corrections persist across reloads (and can be exported for training).
    const CORR_KEY = 'pulsemap_corrections';
    let corrections = {};
    try { corrections = JSON.parse(localStorage.getItem(CORR_KEY) || '{}'); } catch (e) {}

    // Move a node into a section: recolor, and (on load) reposition into its cluster.
    function placeInSection(node, sec, reposition) {
        node.section = sec;
        if (secColor[sec]) node.color = secColor[sec];
        if (reposition && secCentroid[sec]) {
            const [ccx, ccy] = secCentroid[sec];
            const ang = Math.random() * 2 * Math.PI, rad = 3 + Math.random() * 7;
            node.homeX = ccx + Math.cos(ang) * rad;
            node.homeY = ccy + Math.sin(ang) * rad;
            node.x = node.homeX; node.y = node.homeY;
        }
    }
    // Apply saved corrections on load.
    nodes.forEach(n => {
        if (corrections[n.path] && corrections[n.path] !== n.section && secCentroid[corrections[n.path]])
            placeInSection(n, corrections[n.path], true);
    });

    document.getElementById('subtitle').textContent =
        `${nodes.length.toLocaleString()} sounds · ${sections.length} sections`;

    const vw = () => container.clientWidth || 1280;
    const vh = () => container.clientHeight || 720;

    // Intro fly-home tween: drives each node from its gathered start position to
    // its home DIRECTLY (independent of the physics engine's alpha/energy
    // budget), over a fixed duration regardless of node count or travel
    // distance. Pure physics for this was unreliable -- a handful of sounds
    // spread across a wide section grid have no collision-crowd pressure to
    // help propel them (unlike a big dense cluster), so the spring force alone
    // could run out of simulated "energy" before covering the distance and
    // freeze mid-flight. A manual tween can't run out of energy. Nodes are
    // pinned (fx/fy) during the tween so the physics engine doesn't fight it,
    // then released to physics for the organic settle/breathe phase.
    const INTRO_T0 = performance.now();
    const INTRO_DURATION = 1100;
    const easeOutCubic = t => 1 - Math.pow(1 - t, 3);

    const Graph = ForceGraph()(container)
        .width(vw())
        .height(vh())
        .graphData({ nodes, links: [] })
        .backgroundColor('#0e0f13')
        .nodeRelSize(4)
        .onNodeHover(n => {
            container.style.cursor = n ? 'pointer' : null;
            hovered = n || null;
            // NOTE: deliberately no reheat() here. With ~dozens of packed nodes,
            // hovering constantly swaps the target and full-reheating the sim on
            // every swap kept it at high energy forever -> perpetual jiggle. The
            // hover-grow effect below already gives tactile feedback without physics.
            showHoverBanner(n);
            if (n) playAudio(n); else stopAudio();
        })
        .onNodeClick(n => playAudio(n))
        .nodeCanvasObject((node, ctx, scale) => {
            if (!node.__landed) {
                const t = Math.min(1, (performance.now() - INTRO_T0) / INTRO_DURATION);
                const e = easeOutCubic(t);
                node.fx = node.__introStartX + (node.homeX - node.__introStartX) * e;
                node.fy = node.__introStartY + (node.homeY - node.__introStartY) * e;
                node.x = node.fx; node.y = node.fy;
                if (t >= 1) { 
                    node.fx = null; node.fy = null; node.__landed = true; 
                    if (!window.__introEndedTriggered) {
                        window.__introEndedTriggered = true;
                        setTimeout(() => {
                            Graph.refresh();
                            Graph.d3ReheatSimulation();
                        }, 100);
                    }
                }
            }
            const hit = matches(node);
            // smooth hover grow
            const target = node === hovered ? 1 : 0;
            node.__s = (node.__s || 0) + (target - (node.__s || 0)) * 0.25;
            const r = (hit ? 4 : 2.4) / Math.sqrt(scale) * (1 + 0.8 * node.__s);
            // colorize as the sound arrives at its home (grey -> section color, one-way)
            if (Math.hypot(node.x - node.homeX, node.y - node.homeY) < 8)
                node.__colorT = Math.min(1, node.__colorT + 0.16);
            // Highlight active group (dim all other groups if one is hovered/legend-hovered)
            const activeGroup = hovered ? hovered.section : activeSectionHover;
            let col = lerpColor(GREY, node.color, node.__colorT);
            
            // If this node is in the active highlighted group, display it as a beautiful 2D map of colors:
            // - X-Axis (length/sustain): Varies the Hue (rotates by up to +-30 degrees).
            // - Y-Axis (brightness/pitch): Varies the Lightness (adjusts by up to +-15% so higher frequencies glow lighter).
            if (activeGroup && node.section === activeGroup && node.relX !== undefined && node.relY !== undefined) {
                const [h, s, l] = hexToHsl(node.color);
                const shiftedHue = (h + (node.relX - 0.5) * 60 + 360) % 360;
                const shiftedL = Math.max(25, Math.min(85, l + (node.relY - 0.5) * 30));
                col = `hsl(${shiftedHue}, ${s}%, ${shiftedL}%)`;
            }

            // confidence -> opacity; search miss -> heavily dimmed
            let a = 0.35 + 0.65 * (node.confidence || 1);
            if (!hit) a *= 0.12;

            if (activeGroup && node.section !== activeGroup) {
                a *= 0.15;
            }

            // glow (kept modest relative to the collision radius below, or halos overlap even when cores don't)
            ctx.globalAlpha = a * 0.3;
            ctx.fillStyle = col;
            ctx.beginPath(); ctx.arc(node.x, node.y, r * 1.8, 0, 2 * Math.PI); ctx.fill();
            // core
            ctx.globalAlpha = a;
            ctx.fillStyle = col;
            ctx.beginPath(); ctx.arc(node.x, node.y, r, 0, 2 * Math.PI); ctx.fill();
            ctx.globalAlpha = 1;
            // No in-canvas text (filenames/section names) — it occluded nodes
            // and reflowed distractingly. Hover info lives in the top banner;
            // section identity comes from the sidebar legend + node color.
        })
        .nodePointerAreaPaint((node, color, ctx, scale) => {
            // Keep the interactive hover/click radius at a minimum size on screen
            // so that when you zoom out, the target does not shrink to an un-hittable sub-pixel.
            const r = Math.max(5, 5 / (scale || 1));
            ctx.fillStyle = color;
            ctx.beginPath(); ctx.arc(node.x, node.y, r, 0, 2 * Math.PI); ctx.fill();
        })
        // Drag-target ring only (section signage removed — see note above).
        .onRenderFramePost((ctx, scale) => {
            // While dragging a sound near a section, ring where it will drop.
            if (dragTarget && secCentroid[dragTarget]) {
                const [tx, ty] = secCentroid[dragTarget];
                ctx.globalAlpha = 0.9;
                ctx.strokeStyle = secColor[dragTarget] || '#fff';
                ctx.lineWidth = 2 / scale;
                ctx.setLineDash([4 / scale, 3 / scale]);
                ctx.beginPath(); ctx.arc(tx, ty, 15, 0, 2 * Math.PI); ctx.stroke();
                ctx.setLineDash([]);
                ctx.globalAlpha = 1;
            }
        });

    // Fluid physics: each sound is spring-pulled toward its meaningful home
    // position, with collision + gentle repulsion so clusters look organic and
    // react when disturbed. Sections stay put (homes are the computed layout);
    // only the sounds within them breathe and give way.
    // d3.forceX/Y(fn) only evaluates `fn` ONCE, when the force is (re-)registered
    // on the simulation — not every tick. So when a node's homeX/homeY changes
    // (e.g. after a drag), the spring keeps pulling toward the OLD target unless
    // we re-register the force. refreshHomeForces() does that.
    const refreshHomeForces = () => {
        // Soft spring strength (0.20-0.25) allows nodes to slide smoothly to their PCA homes,
        // letting collision forces push them apart to prevent overlaps without creating physical resonance (vibrations).
        const springStrength = nodes.length > 800 ? 0.25 : 0.20;
        Graph.d3Force('x', d3.forceX(d => d.homeX).strength(springStrength));
        Graph.d3Force('y', d3.forceY(d => d.homeY).strength(springStrength));
    };

    const nCount = nodes.length;
    const velocityDecay = nCount > 800 ? 0.30 : 0.36; // Dynamic friction damping to prevent bouncing and overshoots
    const alphaDecay = nCount > 800 ? 0.006 : 0.018;   // Dynamic alpha decay: runs for exactly 1146 ticks on large maps
    const cooldown = nCount > 800 ? 35000 : 15000;       // Dynamic maximum run time in milliseconds
    const collideStrength = nCount > 800 ? 0.45 : 0.7;  // Softened collision for large counts to prevent lag
    const collideIter = nCount > 800 ? 1 : 2;

    Graph
        .d3Force('link', null)
        .d3Force('center', null)
        .d3Force('charge', null)
        .d3Force('collide', d3.forceCollide(3.8).strength(collideStrength).iterations(collideIter))
        .d3VelocityDecay(velocityDecay)
        .d3AlphaDecay(alphaDecay)
        .cooldownTime(cooldown);

    refreshHomeForces();
    const reheat = () => {
        Graph.d3ReheatSimulation();
    };

    // ---- drag a sound into another section to reclassify it ----------------
    const DROP_RADIUS = 16;  // how close to a section's center counts as "dropped in it"
    let dragTarget = null;   // section name currently hovered while dragging (for highlight)

    function nearestSection(x, y, exclude) {
        let best = null, bestD = Infinity;
        for (const s of sections) {
            if (s.name === exclude) continue;
            const d = Math.hypot(x - s.cx, y - s.cy);
            if (d < bestD) { bestD = d; best = s.name; }
        }
        return { name: best, dist: bestD };
    }

    Graph.enableNodeDrag(true)
        .onNodeDrag(node => {
            node.__landed = true; // user interaction always wins over the intro tween
            const t = nearestSection(node.x, node.y, node.section);
            dragTarget = (t.name && t.dist < DROP_RADIUS) ? t.name : null;
        })
        .onNodeDragEnd(node => {
            const t = nearestSection(node.x, node.y, node.section);
            if (t.name && t.dist < DROP_RADIUS) {
                placeInSection(node, t.name, false);
                node.homeX = node.x; node.homeY = node.y;  // its new anchor = where you dropped it
                refreshHomeForces();  // re-register so the spring targets the NEW home
                corrections[node.path] = t.name;
                localStorage.setItem(CORR_KEY, JSON.stringify(corrections));
                updateCounts();
                updateExport();
                showNow(`✓ moved to ${t.name}`);
            }
            // Release to the sim: a valid drop settles into its new section, an
            // invalid one springs fluidly back to its home. Physics handles both.
            node.fx = null; node.fy = null;
            dragTarget = null;
            reheat();
        });

    // Legend / navigator: click a section to fly to it.
    const rows = document.getElementById('legend-rows');
    rows.innerHTML = ''; // Clear previous entries to prevent duplication on multiple folder loads
    const legendCount = {};
    sections.forEach(s => {
        const el = document.createElement('div');
        el.className = 'legrow';
        const count = document.createElement('div');
        count.className = 'legcount';
        count.textContent = s.count;
        legendCount[s.name] = count;
        el.innerHTML = `<div class="swatch" style="background:${s.color}"></div>` +
            `<div class="legname">${s.name}</div>`;
        el.appendChild(count);
        el.onclick = () => {
            const fitZoom = window.__fitZoom || Graph.zoom() || 1.2;
            Graph.centerAt(s.cx, s.cy, 600);
            Graph.zoom(fitZoom * 2.6, 600);
        };
        
        // Hovering a legend row highlights its group on the map
        el.onmouseenter = () => {
            activeSectionHover = s.name;
            Graph.nodeRelSize(4); // Force redraw
        };
        el.onmouseleave = () => {
            activeSectionHover = null;
            Graph.nodeRelSize(4); // Force redraw
        };
        
        rows.appendChild(el);
    });

    // Recount sections live after a reclassification (updates legend + signage).
    function updateCounts() {
        const c = {};
        nodes.forEach(n => { c[n.section] = (c[n.section] || 0) + 1; });
        sections.forEach(s => {
            s.count = c[s.name] || 0;
            if (legendCount[s.name]) legendCount[s.name].textContent = s.count;
        });
    }

    // Export the user's corrections (human-verified labels) for retraining.
    const exportBtn = document.getElementById('export');
    function updateExport() {
        const n = Object.keys(corrections).length;
        exportBtn.style.display = n ? 'block' : 'none';
        exportBtn.textContent = `⤓ Export ${n} fix${n === 1 ? '' : 'es'}`;
    }
    exportBtn.onclick = () => {
        const blob = new Blob([JSON.stringify(corrections, null, 2)], { type: 'application/json' });
        const a = document.createElement('a');
        a.href = URL.createObjectURL(blob);
        a.download = 'corrections.json';
        a.click();
    };
    updateCounts();
    updateExport();

    // Search
    const searchEl = document.getElementById('search');
    searchEl.addEventListener('input', e => {
        query = e.target.value.trim();
        Graph.nodeRelSize(4); // force a redraw
    });

    window.Graph = Graph;  // expose for debugging

    window.addEventListener('resize', () => { Graph.width(vw()).height(vh()); });

    // Explicit, deterministic framing from stable home positions (live positions
    // drift with physics, so bounds must come from the anchors).
    const xs = nodes.map(n => n.homeX), ys = nodes.map(n => n.homeY);
    const minX = Math.min(...xs), maxX = Math.max(...xs);
    const minY = Math.min(...ys), maxY = Math.max(...ys);
    const cx = (minX + maxX) / 2, cy = (minY + maxY) / 2;
    const frame = () => {
        Graph.width(vw()).height(vh());
        const zoom = Math.min(
            (vw() - 260) / (maxX - minX + 20),
            vh() / (maxY - minY + 20)
        );
        window.__fitZoom = zoom;
        Graph.centerAt(cx, cy, 0).zoom(zoom, 0);
    };
    loadingOverlay.classList.add('hidden');
    setTimeout(frame, 300);
    setTimeout(frame, 900);
}

// ---- drop-zone: the entire "no terminal, no install" front door ------------
// User drags a folder in (or picks one), everything downstream (CLAP embedding,
// classification, layout) runs right here in the tab via pipeline.js.

const dropEl = document.getElementById('drop');
const dropZone = document.getElementById('dropZone');
const dropInput = document.getElementById('dropInput');
const progressEl = document.getElementById('progress');
const progressLabel = document.getElementById('progressLabel');
const progressBar = document.getElementById('progressBarInner');
const progressCount = document.getElementById('progressCount');

/** Recursively collect every File under a list of dropped DataTransferItems
 * (drag-and-drop of a folder gives directory *entries*, not File objects
 * directly -- this walks them). Falls back to flat files for a plain
 * multi-file drop. */
async function collectFilesFromDrop(items) {
    function readEntry(entry) {
        return new Promise((resolve) => {
            if (entry.isFile) {
                entry.file((file) => resolve([file]), () => resolve([]));
            } else if (entry.isDirectory) {
                const reader = entry.createReader();
                const all = [];
                const readBatch = () => {
                    reader.readEntries(async (entries) => {
                        if (!entries.length) {
                            resolve((await Promise.all(all)).flat());
                            return;
                        }
                        all.push(...entries.map(readEntry));
                        readBatch();
                    }, () => resolve([]));
                };
                readBatch();
            } else {
                resolve([]);
            }
        });
    }
    const entries = Array.from(items)
        .map((item) => item.webkitGetAsEntry && item.webkitGetAsEntry())
        .filter(Boolean);
    if (entries.length) {
        const nested = await Promise.all(entries.map(readEntry));
        return nested.flat();
    }
    // Fallback: no directory-entry support -- use whatever plain files were dropped.
    return Array.from(items).map((item) => item.getAsFile && item.getAsFile()).filter(Boolean);
}

function formatEta(seconds) {
    if (seconds < 60) return `${seconds}s`;
    const m = Math.floor(seconds / 60);
    const s = seconds % 60;
    if (m < 60) return `${m}m ${s}s`;
    const h = Math.floor(m / 60);
    const rm = m % 60;
    return `${h}h ${rm}m ${s}s`;
}

async function handleFiles(files) {
    if (!files.length) return;
    dropEl.classList.add('hidden');
    progressEl.classList.add('show');
    progressLabel.textContent = 'Getting ready…';
    progressBar.style.width = '0%';
    progressCount.textContent = '';

    try {
        let startTime = null;
        let initialDone = 0;
        let lastDone = 0;
        let lastTime = 0;
        let smoothedRate = null;

        const result = await buildMap(files, (msg, done, total) => {
            progressLabel.textContent = msg;
            if (total) {
                progressBar.style.width = `${Math.round((100 * done) / total)}%`;
                
                // Calculate ETA based on worker processing speed
                if (msg.startsWith('Analyzing sounds')) {
                    const now = performance.now();
                    if (done > initialDone) {
                        if (startTime === null) {
                            // First file has finished! Start the timer now.
                            startTime = now;
                            lastDone = initialDone;
                            lastTime = now;
                        }
                        
                        const elapsedMs = now - lastTime;
                        if (elapsedMs > 0) {
                            const instantRate = (done - lastDone) / elapsedMs;
                            if (smoothedRate === null) {
                                smoothedRate = instantRate;
                            } else {
                                const alpha = 0.2; // 20% weight on recent rate
                                smoothedRate = alpha * instantRate + (1 - alpha) * smoothedRate;
                            }
                        }
                        
                        lastDone = done;
                        lastTime = now;
                        
                        if (smoothedRate) {
                            const remaining = total - done;
                            const etaSec = Math.max(0, Math.round((remaining / smoothedRate) / 1000));
                            progressCount.textContent = `${done} / ${total} (ETA: ${formatEta(etaSec)})`;
                        } else {
                            progressCount.textContent = `${done} / ${total}`;
                        }
                    } else {
                        initialDone = done; // Keep syncing initial done until first file lands
                        progressCount.textContent = `${done} / ${total}`;
                    }
                } else {
                    // Reset timer if we are in another stage (e.g. Starting threads...)
                    startTime = null;
                    progressCount.textContent = `${done} / ${total}`;
                }
            } else {
                progressBar.style.width = '15%';
                progressCount.textContent = '';
            }
        });
        progressEl.classList.remove('show');
        if (!result.nodes.length) {
            alert('No usable one-shot sounds found in that folder (loops and non-audio files are skipped).');
            dropEl.classList.remove('hidden');
            return;
        }
        renderMap(result);
    } catch (e) {
        console.error(e);
        progressEl.classList.remove('show');
        dropEl.classList.remove('hidden');
        alert('Something went wrong analyzing your sounds. Check the browser console for details.');
    }
}

window.handleFiles = handleFiles; // exposed for debugging/testing

dropZone.addEventListener('click', () => dropInput.click());
dropInput.addEventListener('change', (e) => handleFiles(Array.from(e.target.files)));

window.addEventListener('dragover', (e) => {
    e.preventDefault();
    document.getElementById('drop').classList.remove('hidden');
    dropZone.classList.add('drag');
});
window.addEventListener('dragleave', (e) => {
    e.preventDefault();
    // Only hide if dragging completely out of the browser window
    if (!e.relatedTarget || e.relatedTarget.nodeName === "HTML") {
        const hasMap = document.getElementById('graph-container').children.length > 0;
        if (hasMap) {
            document.getElementById('drop').classList.add('hidden');
        }
        dropZone.classList.remove('drag');
    }
});
window.addEventListener('drop', async (e) => {
    e.preventDefault();
    dropZone.classList.remove('drag');
    const files = await collectFilesFromDrop(e.dataTransfer.items);
    if (files && files.length > 0) {
        handleFiles(files);
    } else {
        const hasMap = document.getElementById('graph-container').children.length > 0;
        if (hasMap) {
            document.getElementById('drop').classList.add('hidden');
        }
    }
});

// Clear Cache Button
const clearCacheBtn = document.getElementById('clear-cache');
if (clearCacheBtn) {
    clearCacheBtn.addEventListener('click', async () => {
        if (confirm('Are you sure you want to clear the sound database, corrections, and last mapped kit? This will reset the app.')) {
            await clearAllCaches();
            window.location.reload();
        }
    });
}

// On startup, load last map from cache if it exists
try {
    const cached = localStorage.getItem('pulsemap_last_result');
    if (cached) {
        const data = JSON.parse(cached);
        if (data && data.nodes && data.nodes.length > 0) {
            // Render the cached map immediately
            setTimeout(() => {
                renderMap(data);
                showNow("📂 Last map loaded. Drop folder to preview sounds.");
            }, 100);
        }
    }
} catch (e) {
    console.warn('Failed to load cached map on startup:', e);
}
