import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { FontLoader } from 'three/addons/loaders/FontLoader.js';
import { TextGeometry } from 'three/addons/geometries/TextGeometry.js';
import { GUI } from 'three/addons/libs/lil-gui.module.min.js';

// --- DOM References ---
const loadingIndicator = document.getElementById('loading-indicator');
const tooltip = document.getElementById('tooltip');
const timeControls = document.getElementById('time-controls');
const timeSlider = document.getElementById('time-slider');
const timeLabel = document.getElementById('time-label');

// --- Scene Setup ---
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 2000);
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);

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

// --- Data Structures ---
const models = [
    { name: 'BGE-Large', group: new THREE.Group(), shape: 'square', file: 'papers_bge_large.json', metadata: [], points: null, axesHelper: null, labelMesh: null },
    { name: 'GTE-Large', group: new THREE.Group(), shape: 'circle', file: 'papers_gte_large.json', metadata: [], points: null, axesHelper: null, labelMesh: null },
    { name: 'MiniLM', group: new THREE.Group(), shape: 'triangle', file: 'papers_minilm.json', metadata: [], points: null, axesHelper: null, labelMesh: null }
];
const textureLoader = new THREE.TextureLoader();
const textures = {
    square: textureLoader.load('./square.png'),
    circle: textureLoader.load('./circle.png'),
    triangle: textureLoader.load('./triangle.png'),
};

const separationDistance = 25;
models[0].group.position.x = -separationDistance;
models[1].group.position.x = 0;
models[2].group.position.x = separationDistance;
models.forEach(model => scene.add(model.group));

// --- Raycasting ---
const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2(1, 1);
const mousePosition = new THREE.Vector2();
raycaster.params.Points.threshold = 0.5;

// --- Temporal State ---
let temporalManifest = null;
let currentFont = null;
const snapshotCache = new Map();
let snapshotRequestToken = 0;

// --- Utility Functions ---
function disposeMesh(mesh) {
    if (!mesh) return;
    if (mesh.geometry) mesh.geometry.dispose();
    if (mesh.material) {
        if (Array.isArray(mesh.material)) {
            mesh.material.forEach(mat => mat.dispose());
        } else {
            mesh.material.dispose();
        }
    }
}

function formatSnapshotLabel(snapshot) {
    if (!snapshot) return '--';
    if (snapshot.label) return snapshot.label;
    try {
        return new Date(snapshot.timestamp).toLocaleString(undefined, {
            year: 'numeric',
            month: 'short',
            day: 'numeric',
        });
    } catch (error) {
        return snapshot.timestamp || '--';
    }
}

async function loadSnapshot(path) {
    if (snapshotCache.has(path)) {
        return snapshotCache.get(path);
    }
    const response = await fetch(path);
    if (!response.ok) {
        throw new Error(`Failed to load snapshot: ${path}`);
    }
    const payload = await response.json();
    snapshotCache.set(path, payload);
    return payload;
}

function createPointCloud(model, font, data) {
    model.metadata = data;

    if (model.points) {
        model.group.remove(model.points);
        disposeMesh(model.points);
        model.points = null;
    }
    if (model.axesHelper) {
        model.group.remove(model.axesHelper);
        model.axesHelper = null;
    }
    if (model.labelMesh) {
        model.group.remove(model.labelMesh);
        disposeMesh(model.labelMesh);
        model.labelMesh = null;
    }

    if (!data || data.length === 0) {
        return;
    }

    const positions = data.flatMap(p => [p.x, p.y, p.z]);

    const clusterColors = {};
    const uniqueClusters = [...new Set(data.map(p => p.cluster_id))];
    uniqueClusters.forEach(id => {
        if (id === -1) clusterColors[id] = new THREE.Color(0x666666);
        else {
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

    const axesHelper = new THREE.AxesHelper(Math.max(maxDim * 0.75, 1));
    model.axesHelper = axesHelper;
    model.group.add(axesHelper);

    if (font) {
        const textGeom = new TextGeometry(model.name, { font, size: Math.max(maxDim * 0.15, 1.5), depth: Math.max(maxDim * 0.02, 0.25) });
        const textMat = new THREE.MeshPhongMaterial({ color: 0xffffff });
        const textMesh = new THREE.Mesh(textGeom, textMat);
        textMesh.position.set(0, size.y / 2 + Math.max(maxDim * 0.2, 2), 0);
        model.labelMesh = textMesh;
        model.group.add(textMesh);
    }
}

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
}

async function applySnapshot(index) {
    if (!temporalManifest || !currentFont) return;
    const descriptor = temporalManifest.snapshots[index];
    if (!descriptor) return;
    const requestId = ++snapshotRequestToken;
    const label = formatSnapshotLabel(descriptor);
    timeLabel.textContent = `${label} (loading...)`;
    try {
        const data = await loadSnapshot(descriptor.path);
        if (requestId !== snapshotRequestToken) {
            return;
        }
        for (const model of models) {
            const snapshotData = data.models?.[model.name] ?? [];
            createPointCloud(model, currentFont, snapshotData);
        }
        timeLabel.textContent = label;
    } catch (error) {
        console.error(error);
        timeLabel.textContent = 'Failed to load timeline';
    }
}

function setupTimeControls(manifest) {
    if (!manifest || !manifest.snapshots || manifest.snapshots.length === 0) {
        timeControls.classList.add('hidden');
        return;
    }
    temporalManifest = manifest;
    timeControls.classList.remove('hidden');
    timeSlider.min = 0;
    timeSlider.max = manifest.snapshots.length - 1;
    timeSlider.value = manifest.snapshots.length - 1;
    const latestIndex = manifest.snapshots.length - 1;
    timeLabel.textContent = formatSnapshotLabel(manifest.snapshots[latestIndex]);
    timeSlider.oninput = event => {
        const index = parseInt(event.target.value, 10);
        applySnapshot(index);
    };
    applySnapshot(latestIndex);
}

async function loadStaticData(font) {
    const dataPromises = models.map(model =>
        fetch(`./${model.file}`).then(res => res.json()).then(data => ({ ...model, data }))
    );
    const loadedModels = await Promise.all(dataPromises);
    loadedModels.forEach(loadedModel => {
        const model = models.find(m => m.name === loadedModel.name);
        if (model) {
            createPointCloud(model, font, loadedModel.data);
        }
    });
}

async function loadAllData() {
    const fontPromise = new FontLoader().loadAsync('https://unpkg.com/three@0.165.0/examples/fonts/helvetiker_regular.typeface.json');
    const analysisPromise = fetch('analysis_summary.json').then(res => res.json());
    const manifestPromise = fetch('temporal_snapshots/manifest.json')
        .then(res => res.ok ? res.json() : null)
        .catch(() => null);

    const [font, analysisData, manifest] = await Promise.all([fontPromise, analysisPromise, manifestPromise]);
    currentFont = font;

    if (manifest) {
        setupTimeControls(manifest);
    } else {
        await loadStaticData(font);
        timeControls.classList.add('hidden');
    }

    setupGUI(analysisData);
    loadingIndicator.style.display = 'none';
}

// --- Interaction & Rendering ---
window.addEventListener('pointermove', event => {
    pointer.x = (event.clientX / window.innerWidth) * 2 - 1;
    pointer.y = - (event.clientY / window.innerHeight) * 2 + 1;
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
            const lastUpdated = paper.last_updated ? `<p>Last Updated: ${new Date(paper.last_updated).toLocaleDateString()}</p>` : '';
            tooltip.innerHTML = `
                <strong>${paper.title}</strong>
                <p>Authors: ${paper.authors}</p>
                <p>Published: ${paper.date}</p>
                <p>Cluster ID: ${paper.cluster_id}</p>
                ${lastUpdated}
                <a href="https://arxiv.org/abs/${paper.id}" target="_blank">View on arXiv</a>
            `;
        }
    } else {
        tooltip.style.display = 'none';
    }

    renderer.render(scene, camera);
}

window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
});

loadAllData();
animate();
