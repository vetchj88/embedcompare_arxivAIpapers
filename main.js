import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { FontLoader } from 'three/addons/loaders/FontLoader.js';
import { TextGeometry } from 'three/addons/geometries/TextGeometry.js';
import { GUI } from 'three/addons/libs/lil-gui.module.min.js';

// --- Basic Scene Setup ---
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 2000);
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);
const loadingIndicator = document.getElementById('loading-indicator');

// --- Lighting ---
const ambientLight = new THREE.AmbientLight(0xffffff, 0.7);
scene.add(ambientLight);
const directionalLight = new THREE.DirectionalLight(0xffffff, 0.5);
directionalLight.position.set(5, 10, 7.5);
scene.add(directionalLight);

// --- Controls ---
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
camera.position.set(0, 15, 60);
controls.update();

// --- Data, Groups, and Object Storage ---
const tooltip = document.getElementById('tooltip');
const explanationPanel = document.createElement('div');
explanationPanel.id = 'explanation-panel';
explanationPanel.innerHTML = '<h3>Hover over a paper</h3><p class="empty-state">Explanations will appear here.</p>';
document.body.appendChild(explanationPanel);

const explanationStyle = document.createElement('style');
explanationStyle.textContent = `
    #explanation-panel {
        position: absolute;
        top: 110px;
        right: 480px;
        width: 340px;
        max-height: 65vh;
        overflow-y: auto;
        background: rgba(15, 15, 15, 0.88);
        border: 1px solid #333;
        border-radius: 12px;
        padding: 16px 18px;
        font-size: 14px;
        line-height: 1.45;
        color: #e5e5e5;
        z-index: 60;
        pointer-events: auto;
        box-shadow: 0 12px 30px rgba(0, 0, 0, 0.45);
    }
    #explanation-panel.hidden { display: none; }
    #explanation-panel h3 {
        margin: 0 0 6px 0;
        font-size: 16px;
    }
    #explanation-panel .meta {
        color: #bbbbbb;
        font-size: 12px;
        margin-bottom: 8px;
    }
    #explanation-panel .edge {
        border-top: 1px solid #2a2a2a;
        padding-top: 10px;
        margin-top: 10px;
    }
    #explanation-panel .edge:first-of-type {
        border-top: none;
        padding-top: 0;
        margin-top: 0;
    }
    #explanation-panel .signals {
        color: #9cdcfe;
        font-size: 12px;
        margin-top: 4px;
    }
    #explanation-panel .scores {
        color: #dcdcaa;
        font-size: 12px;
        margin-top: 4px;
    }
    #explanation-panel .empty-state {
        color: #888888;
        font-style: italic;
        margin: 12px 0 0 0;
    }
`;
document.head.appendChild(explanationStyle);

const explanationSettings = {
    showPanel: true,
    showOverlay: false,
    maxEdges: 5,
    filterSharedDatasets: true,
    filterMutualCitations: true,
    filterTaxonomyMatches: true,
};

let explanationGraph = null;
let explanationIndex = null;
const explanationCache = new Map();
let lastHoverKey = null;
let lastOverlayModel = null;
let currentHoverEdges = [];

const models = [
    { name: 'BGE-Large', group: new THREE.Group(), shape: 'square', file: 'papers_bge_large.json', metadata: [], points: null, idToIndex: null, overlayGroup: null },
    { name: 'GTE-Large', group: new THREE.Group(), shape: 'circle', file: 'papers_gte_large.json', metadata: [], points: null, idToIndex: null, overlayGroup: null },
    { name: 'MiniLM', group: new THREE.Group(), shape: 'triangle', file: 'papers_minilm.json', metadata: [], points: null, idToIndex: null, overlayGroup: null }
];

const textureLoader = new THREE.TextureLoader();
const textures = {
    square: textureLoader.load('./square.png'),
    circle: textureLoader.load('./circle.png'),
    triangle: textureLoader.load('./triangle.png'),
};

// --- Group Positioning ---
const separationDistance = 25;
models[0].group.position.x = -separationDistance;
models[1].group.position.x = 0;
models[2].group.position.x = separationDistance;
models.forEach(model => scene.add(model.group));

// --- Raycasting & Overlay Setup ---
const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2(1, 1);
const mousePosition = new THREE.Vector2();
raycaster.params.Points.threshold = 0.5;
const explanationOverlayMaterial = new THREE.LineBasicMaterial({
    color: 0xffd700,
    transparent: true,
    opacity: 0.45,
});

// --- Utility Helpers ---
function updatePanelVisibility() {
    explanationPanel.style.display = explanationSettings.showPanel ? 'block' : 'none';
}

function resetExplanationPanel() {
    if (!explanationSettings.showPanel) {
        explanationPanel.classList.add('hidden');
        return;
    }
    explanationPanel.classList.remove('hidden');
    explanationPanel.innerHTML = '<h3>Hover over a paper</h3><p class="empty-state">Explanations will appear here.</p>';
}

function buildExplanationIndex() {
    if (explanationIndex || !explanationGraph || !Array.isArray(explanationGraph.edges)) {
        return;
    }
    explanationIndex = new Map();
    explanationGraph.edges.forEach(edge => {
        const normalized = {
            source_id: edge.source_id,
            target_id: edge.target_id,
            scores: edge.scores ?? {},
            signals: edge.signals ?? {},
            metadata: edge.metadata ?? {},
        };
        if (!explanationIndex.has(normalized.source_id)) {
            explanationIndex.set(normalized.source_id, []);
        }
        explanationIndex.get(normalized.source_id).push(normalized);

        const mirrored = {
            source_id: edge.target_id,
            target_id: edge.source_id,
            scores: edge.scores ?? {},
            signals: edge.signals ?? {},
            metadata: {
                source: edge.metadata?.target ?? null,
                target: edge.metadata?.source ?? null,
            },
        };
        if (!explanationIndex.has(mirrored.source_id)) {
            explanationIndex.set(mirrored.source_id, []);
        }
        explanationIndex.get(mirrored.source_id).push(mirrored);
    });

    explanationIndex.forEach(list => {
        list.sort((a, b) => (b.scores?.hybrid_similarity ?? 0) - (a.scores?.hybrid_similarity ?? 0));
    });
}

function getFilterKey() {
    return [
        explanationSettings.filterSharedDatasets ? 1 : 0,
        explanationSettings.filterMutualCitations ? 1 : 0,
        explanationSettings.filterTaxonomyMatches ? 1 : 0,
        explanationSettings.maxEdges,
    ].join('');
}

function invalidateExplanationCache() {
    explanationCache.clear();
    lastHoverKey = null;
    currentHoverEdges = [];
}

function edgeMatchesFilters(edge) {
    const activeTypes = [];
    if (explanationSettings.filterSharedDatasets) activeTypes.push('shared_datasets');
    if (explanationSettings.filterMutualCitations) activeTypes.push('mutual_citations');
    if (explanationSettings.filterTaxonomyMatches) activeTypes.push('taxonomy_matches');

    if (activeTypes.length === 0) {
        return true;
    }

    return activeTypes.some(type => {
        if (type === 'shared_datasets') {
            return Array.isArray(edge.signals?.shared_datasets) && edge.signals.shared_datasets.length > 0;
        }
        if (type === 'mutual_citations') {
            return edge.signals?.mutual_citations === true;
        }
        if (type === 'taxonomy_matches') {
            return Array.isArray(edge.signals?.taxonomy_matches) && edge.signals.taxonomy_matches.length > 0;
        }
        return false;
    });
}

function getFilteredEdges(paperId) {
    if (!explanationGraph || !Array.isArray(explanationGraph.edges)) {
        return [];
    }
    buildExplanationIndex();
    const cacheKey = `${paperId}|${getFilterKey()}`;
    if (explanationCache.has(cacheKey)) {
        return explanationCache.get(cacheKey);
    }

    const candidates = explanationIndex?.get(paperId) ?? [];
    const filtered = candidates.filter(edgeMatchesFilters);
    const limited = filtered.slice(0, explanationSettings.maxEdges);
    explanationCache.set(cacheKey, limited);
    return limited;
}

function formatScore(value) {
    if (typeof value !== 'number' || Number.isNaN(value)) {
        return 'N/A';
    }
    return value.toFixed(3);
}

function buildTooltipExplanation() {
    if (!currentHoverEdges.length) {
        return '';
    }
    const edge = currentHoverEdges[0];
    const parts = [];
    if (edge.signals?.shared_datasets?.length) {
        parts.push(`Shared datasets: ${edge.signals.shared_datasets.join(', ')}`);
    }
    if (edge.signals?.taxonomy_matches?.length) {
        parts.push(`Taxonomy: ${edge.signals.taxonomy_matches.join(', ')}`);
    }
    if (edge.signals?.mutual_citations) {
        parts.push('Mutual citations detected');
    }
    if (parts.length === 0) {
        return '';
    }
    const scoreSummary = `Hybrid ${formatScore(edge.scores?.hybrid_similarity)} | Temporal ${formatScore(edge.scores?.temporal_alignment)} | Graph ${formatScore(edge.scores?.graph_affinity)}`;
    return `<hr><strong>Top Explanation</strong><p>${scoreSummary}</p>${parts.map(text => `<p>${text}</p>`).join('')}`;
}

function clearOverlay(model) {
    if (!model?.overlayGroup) {
        return;
    }
    while (model.overlayGroup.children.length) {
        const child = model.overlayGroup.children[0];
        if (child.geometry) {
            child.geometry.dispose();
        }
        model.overlayGroup.remove(child);
    }
}

function updateOverlay(model, sourceIndex, edges) {
    if (!model || !model.points) {
        return;
    }
    clearOverlay(model);
    if (!explanationSettings.showOverlay || !edges.length) {
        return;
    }
    const positionAttr = model.points.geometry.getAttribute('position');
    if (!positionAttr) {
        return;
    }
    const positions = [];
    const source = new THREE.Vector3(
        positionAttr.getX(sourceIndex),
        positionAttr.getY(sourceIndex),
        positionAttr.getZ(sourceIndex),
    );

    edges.forEach(edge => {
        const targetIndex = model.idToIndex?.get(edge.target_id);
        if (targetIndex === undefined) {
            return;
        }
        const target = new THREE.Vector3(
            positionAttr.getX(targetIndex),
            positionAttr.getY(targetIndex),
            positionAttr.getZ(targetIndex),
        );
        positions.push(source.x, source.y, source.z, target.x, target.y, target.z);
    });

    if (!positions.length) {
        return;
    }

    const overlayGeometry = new THREE.BufferGeometry();
    overlayGeometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    const lines = new THREE.LineSegments(overlayGeometry, explanationOverlayMaterial);
    model.overlayGroup.add(lines);
}

function updateExplanationPanel(paper, edges) {
    if (!explanationSettings.showPanel) {
        return;
    }
    if (!paper) {
        resetExplanationPanel();
        return;
    }
    const header = `<h3>${paper.title}</h3><p class="meta">${paper.authors} • ${paper.date}</p>`;
    if (!edges.length) {
        explanationPanel.innerHTML = `${header}<p class="empty-state">No explanation edges match the current filters.</p>`;
        return;
    }

    const body = edges
        .map(edge => {
            const targetMeta = edge.metadata?.target ?? {};
            const signalParts = [];
            if (edge.signals?.shared_datasets?.length) {
                signalParts.push(`Datasets: ${edge.signals.shared_datasets.join(', ')}`);
            }
            if (edge.signals?.taxonomy_matches?.length) {
                signalParts.push(`Taxonomy: ${edge.signals.taxonomy_matches.join(', ')}`);
            }
            if (edge.signals?.mutual_citations) {
                signalParts.push('Mutual Citations');
            }
            if (edge.signals?.shared_authors?.length) {
                signalParts.push(`Shared Authors: ${edge.signals.shared_authors.join(', ')}`);
            }
            const scoreSummary = `Hybrid ${formatScore(edge.scores?.hybrid_similarity)} | Temporal ${formatScore(edge.scores?.temporal_alignment)} | Graph ${formatScore(edge.scores?.graph_affinity)}`;
            const metaLine = [targetMeta.authors ?? '', targetMeta.date ?? ''].filter(Boolean).join(' • ');
            return `
                <div class="edge">
                    <div class="target"><strong>${targetMeta.title ?? 'Related Paper'}</strong></div>
                    <div class="meta">${metaLine}</div>
                    <div class="scores">${scoreSummary}</div>
                    ${signalParts.length ? `<div class="signals">${signalParts.join(' • ')}</div>` : ''}
                </div>
            `;
        })
        .join('');

    explanationPanel.innerHTML = `${header}${body}`;
}

function handlePaperHover(model, paper, index) {
    if (!paper) {
        currentHoverEdges = [];
        return;
    }
    const hoverKey = `${model.name}:${paper.id}:${getFilterKey()}`;
    if (hoverKey !== lastHoverKey) {
        const edges = getFilteredEdges(paper.id);
        currentHoverEdges = edges;
        updateExplanationPanel(paper, edges);
        if (explanationSettings.showOverlay) {
            if (lastOverlayModel && lastOverlayModel !== model) {
                clearOverlay(lastOverlayModel);
            }
            updateOverlay(model, index, edges);
            lastOverlayModel = model;
        } else if (lastOverlayModel) {
            clearOverlay(lastOverlayModel);
            lastOverlayModel = null;
        }
        lastHoverKey = hoverKey;
    }
}

// --- Main Data Loading Function ---
async function loadAllData() {
    const fontPromise = new FontLoader().loadAsync('https://unpkg.com/three@0.165.0/examples/fonts/helvetiker_regular.typeface.json');
    const dataPromises = models.map(model => fetch(`./${model.file}`).then(res => res.json()).then(data => ({ ...model, data })));
    const analysisPromise = fetch('analysis_summary.json').then(res => res.json());

    const [font, analysisData, ...loadedModels] = await Promise.all([fontPromise, analysisPromise, ...dataPromises]);
    explanationGraph = analysisData.explanation_graph ?? null;

    loadedModels.forEach(loadedModel => {
        const model = models.find(m => m.name === loadedModel.name);
        if (!model) {
            return;
        }
        model.metadata = loadedModel.data;
        model.idToIndex = new Map();
        model.metadata.forEach((paper, idx) => {
            model.idToIndex.set(paper.id, idx);
        });
        createPointCloud(model, font);
        if (!model.overlayGroup) {
            model.overlayGroup = new THREE.Group();
            model.overlayGroup.visible = explanationSettings.showOverlay;
            model.group.add(model.overlayGroup);
        }
    });

    setupGUI(analysisData);
    updatePanelVisibility();
    resetExplanationPanel();
    loadingIndicator.style.display = 'none';
}

// --- Point Cloud Creation Function ---
function createPointCloud(model, font) {
    const data = model.metadata;
    const positions = data.flatMap(p => [p.x, p.y, p.z]);

    const clusterColors = {};
    const uniqueClusters = [...new Set(data.map(p => p.cluster_id))];
    uniqueClusters.forEach(id => {
        if (id === -1) {
            clusterColors[id] = new THREE.Color(0x666666);
        } else {
            const color = new THREE.Color();
            color.setHSL((id * 0.13) % 1.0, 0.7, 0.6);
            clusterColors[id] = color;
        }
    });
    const colors = data.flatMap(p => clusterColors[p.cluster_id].toArray());

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));

    geometry.computeBoundingBox();
    const center = new THREE.Vector3();
    geometry.boundingBox.getCenter(center);
    geometry.translate(-center.x, -center.y, -center.z);

    geometry.computeBoundingBox();
    const size = new THREE.Vector3();
    geometry.boundingBox.getSize(size);
    const maxDim = Math.max(size.x, size.y, size.z);

    const material = new THREE.PointsMaterial({
        map: textures[model.shape],
        alphaTest: 0.5,
        vertexColors: true,
        size: 0.4,
        sizeAttenuation: true,
    });

    const points = new THREE.Points(geometry, material);
    points.name = model.name;
    model.points = points;
    model.group.add(points);

    const axesHelper = new THREE.AxesHelper(maxDim * 0.75);
    model.group.add(axesHelper);

    const textGeom = new TextGeometry(model.name, { font: font, size: maxDim * 0.15, depth: maxDim * 0.02 });
    const textMat = new THREE.MeshPhongMaterial({ color: 0xffffff });
    const textMesh = new THREE.Mesh(textGeom, textMat);
    textMesh.position.set(0, size.y / 2 + maxDim * 0.2, 0);
    model.group.add(textMesh);
}

// --- GUI Controller (Fully Merged) ---
function setupGUI(analysisData) {
    const gui = new GUI();
    const params = {
        'BGE-Large': true,
        'GTE-Large': true,
        'MiniLM': true,
    };

    const visibilityFolder = gui.addFolder('Cloud Visibility');
    visibilityFolder.add(params, 'BGE-Large').name('BGE-Large (Square)').onChange(val => models.find(m => m.name === 'BGE-Large').group.visible = val);
    visibilityFolder.add(params, 'GTE-Large').name('GTE-Large (Circle)').onChange(val => models.find(m => m.name === 'GTE-Large').group.visible = val);
    visibilityFolder.add(params, 'MiniLM').name('MiniLM (Triangle)').onChange(val => models.find(m => m.name === 'MiniLM').group.visible = val);

    const metricsFolder = gui.addFolder('Performance Metrics');
    for (const modelName in analysisData.model_metrics) {
        const modelFolder = metricsFolder.addFolder(modelName);
        const metrics = analysisData.model_metrics[modelName];
        modelFolder.add(metrics, 'Num Clusters').name('Clusters').disable();
        modelFolder.add(metrics, 'Num Outliers').name('Outliers').disable();
        modelFolder.add(metrics, 'Calinski-Harabasz Score').name('C-H Score (Higher is Better)').disable();
        modelFolder.add(metrics, 'Davies-Bouldin Score').name('D-B Score (Lower is Better)').disable();
    }

    if (analysisData.comparative_metrics && analysisData.comparative_metrics['Adjusted Rand Index']) {
        const comparisonFolder = metricsFolder.addFolder('Model Agreement (ARI)');
        const ari = analysisData.comparative_metrics['Adjusted Rand Index'];
        for (const comp in ari) {
            comparisonFolder.add(ari, comp).name(comp.replace(/_/g, ' vs ')).disable();
        }
    }
    metricsFolder.open();

    const explanationFolder = gui.addFolder('Explanation Controls');
    explanationFolder.add(explanationSettings, 'showPanel').name('Show Side Panel').onChange(() => {
        updatePanelVisibility();
        if (!explanationSettings.showPanel) {
            explanationPanel.classList.add('hidden');
        } else {
            explanationPanel.classList.remove('hidden');
            lastHoverKey = null;
        }
    });
    explanationFolder.add(explanationSettings, 'showOverlay').name('Show Edge Overlay').onChange(value => {
        models.forEach(model => {
            if (model.overlayGroup) {
                model.overlayGroup.visible = value;
                if (!value) {
                    clearOverlay(model);
                }
            }
        });
        if (!value) {
            lastOverlayModel = null;
        }
    });
    explanationFolder.add(explanationSettings, 'maxEdges', 1, 10, 1).name('Max Edges per Paper').onChange(() => {
        invalidateExplanationCache();
    });
    explanationFolder.add(explanationSettings, 'filterSharedDatasets').name('Filter: Shared Datasets').onChange(() => {
        invalidateExplanationCache();
    });
    explanationFolder.add(explanationSettings, 'filterMutualCitations').name('Filter: Mutual Citations').onChange(() => {
        invalidateExplanationCache();
    });
    explanationFolder.add(explanationSettings, 'filterTaxonomyMatches').name('Filter: Taxonomy Matches').onChange(() => {
        invalidateExplanationCache();
    });
}

// --- Event Listener & Animation Loop ---
window.addEventListener('pointermove', event => {
    pointer.x = (event.clientX / window.innerWidth) * 2 - 1;
    pointer.y = -(event.clientY / window.innerHeight) * 2 + 1;
    mousePosition.x = event.clientX;
    mousePosition.y = event.clientY;
});

function animate() {
    requestAnimationFrame(animate);
    controls.update();

    raycaster.setFromCamera(pointer, camera);
    const visibleClouds = models.filter(m => m.points && m.group.visible).map(m => m.points);
    const intersects = raycaster.intersectObjects(visibleClouds);

    if (intersects.length > 0) {
        const intersection = intersects[0];
        const index = intersection.index;
        const cloudName = intersection.object.name;
        const model = models.find(m => m.name === cloudName);

        if (model && model.metadata[index]) {
            const paper = model.metadata[index];
            tooltip.style.display = 'block';
            tooltip.style.left = `${mousePosition.x + 15}px`;
            tooltip.style.top = `${mousePosition.y + 15}px`;

            handlePaperHover(model, paper, index);

            const explanationHtml = buildTooltipExplanation();
            tooltip.innerHTML = `
                <strong>${paper.title}</strong>
                <p>Authors: ${paper.authors}</p>
                <p>Published: ${paper.date}</p>
                <p>Cluster ID: ${paper.cluster_id}</p>
                <a href="https://arxiv.org/abs/${paper.id}" target="_blank">View on arXiv</a>
                ${explanationHtml}
            `;
        }
    } else {
        tooltip.style.display = 'none';
        resetExplanationPanel();
        if (lastOverlayModel) {
            clearOverlay(lastOverlayModel);
            lastOverlayModel = null;
        }
        lastHoverKey = null;
        currentHoverEdges = [];
    }

    renderer.render(scene, camera);
}

// --- Initial Load and Start ---
loadAllData();
animate();

window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
});
