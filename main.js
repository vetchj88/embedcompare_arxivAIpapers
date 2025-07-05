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
const models = [
    { name: 'BGE-Large', group: new THREE.Group(), shape: 'square', file: 'papers_bge_large.json', metadata: [], points: null },
    { name: 'GTE-Large', group: new THREE.Group(), shape: 'circle', file: 'papers_gte_large.json', metadata: [], points: null },
    { name: 'MiniLM', group: new THREE.Group(), shape: 'triangle', file: 'papers_minilm.json', metadata: [], points: null }
];
const textureLoader = new THREE.TextureLoader();
const textures = {
    'square': textureLoader.load('./square.png'),
    'circle': textureLoader.load('./circle.png'),
    'triangle': textureLoader.load('./triangle.png')
};

// --- Group Positioning ---
const separationDistance = 25;
models[0].group.position.x = -separationDistance;
models[1].group.position.x = 0;
models[2].group.position.x = separationDistance;
models.forEach(model => scene.add(model.group));

// --- Raycasting Setup ---
const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2(1, 1);
const mousePosition = new THREE.Vector2();
raycaster.params.Points.threshold = 0.5;

// --- Main Data Loading Function ---
async function loadAllData() {
    const fontPromise = new FontLoader().loadAsync('https://unpkg.com/three@0.165.0/examples/fonts/helvetiker_regular.typeface.json');
    const dataPromises = models.map(model =>
        fetch(`./${model.file}`).then(res => res.json()).then(data => ({ ...model, data }))
    );
    // analysis_summary.json is also loaded here
    const analysisPromise = fetch('analysis_summary.json').then(res => res.json());

    const [font, analysisData, ...loadedModels] = await Promise.all([fontPromise, analysisPromise, ...dataPromises]);
    
    loadedModels.forEach(loadedModel => {
        const model = models.find(m => m.name === loadedModel.name);
        if (model) {
            model.metadata = loadedModel.data;
            createPointCloud(model, font);
        }
    });

    setupGUI(analysisData);
    loadingIndicator.style.display = 'none';
}

// --- Point Cloud Creation Function ---
function createPointCloud(model, font) {
    const data = model.metadata;
    const positions = data.flatMap(p => [p.x, p.y, p.z]);
    
    const clusterColors = {};
    const uniqueClusters = [...new Set(data.map(p => p.cluster_id))];
    uniqueClusters.forEach(id => {
        if (id === -1) clusterColors[id] = new THREE.Color(0x666666); // Outlier color
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
        sizeAttenuation: true
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
    // ✅ RESTORED: Loop through model_metrics from the JSON file
    for (const modelName in analysisData.model_metrics) {
        const modelFolder = metricsFolder.addFolder(modelName);
        const metrics = analysisData.model_metrics[modelName];
        
        // Add all metrics from the old GUI
        modelFolder.add(metrics, 'Num Clusters').name('Clusters').disable();
        modelFolder.add(metrics, 'Num Outliers').name('Outliers').disable();
        modelFolder.add(metrics, 'Calinski-Harabasz Score').name('C-H Score (Higher is Better)').disable();
        modelFolder.add(metrics, 'Davies-Bouldin Score').name('D-B Score (Lower is Better)').disable();
    }

    // ✅ RESTORED: Add the comparative ARI scores
    if (analysisData.comparative_metrics && analysisData.comparative_metrics['Adjusted Rand Index']) {
        const comparisonFolder = metricsFolder.addFolder('Model Agreement (ARI)');
        const ari = analysisData.comparative_metrics['Adjusted Rand Index'];
        for (const comp in ari) {
            comparisonFolder.add(ari, comp).name(comp.replace(/_/g, ' vs ')).disable();
        }
    }
    metricsFolder.open();
}


// --- Event Listener & Animation Loop ---
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
            tooltip.innerHTML = `
                <strong>${paper.title}</strong>
                <p>Authors: ${paper.authors}</p>
                <p>Published: ${paper.date}</p>
                <p>Cluster ID: ${paper.cluster_id}</p>
                <a href="https://arxiv.org/abs/${paper.id}" target="_blank">View on arXiv</a>
            `;
        }
    } else {
        tooltip.style.display = 'none';
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