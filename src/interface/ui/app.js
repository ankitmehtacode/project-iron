import { createGaussianProxy } from './components/Visualizer/GaussianSplat.js';
import { updateDashboard } from './components/Dashboard/AnalyticsPanel.js';

const container = document.getElementById('canvas-3d');
const video = document.getElementById('video-player');
const frameDisplay = document.getElementById('frame-display');

if (!container || !video) {
    console.error("CRITICAL ERROR: DOM elements not found. Check your HTML structure.");
}

const scene = new THREE.Scene();
const gridHelper = new THREE.GridHelper(500, 50);
scene.add(gridHelper);

const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth / 2, window.innerHeight);
container.appendChild(renderer.domElement);

// CRITICAL FIX: Moved camera closer to observe the origin-based dummy data
camera.position.set(0, 0, 30);
camera.lookAt(0, 0, 0);

const trajectoryLines = [];
let gaussianSplatProxy = null;
let globalTrajectoryData = null;

function buildTrajectories(data) {
    const initialPoints = [];
    const embeddings = [];

    data.tracks.forEach(track => {
        const points = [];
        const colors = [];
        track.path.forEach(pt => {
            points.push(new THREE.Vector3(pt.x, pt.y, pt.z));
            colors.push(
                pt.r !== undefined ? pt.r : 1,
                pt.g !== undefined ? pt.g : 1,
                pt.b !== undefined ? pt.b : 1
            );
        });

        const geometry = new THREE.BufferGeometry().setFromPoints(points);
        geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
        const material = new THREE.LineBasicMaterial({ vertexColors: true, linewidth: 2 });
        const line = new THREE.Line(geometry, material);

        line.geometry.setDrawRange(0, 0);
        scene.add(line);
        trajectoryLines.push({ line: line, data: track.path });

        if (track.path.length > 0) {
            initialPoints.push(track.path[0]);
            embeddings.push([
                track.path[0].r !== undefined ? track.path[0].r : 1,
                track.path[0].g !== undefined ? track.path[0].g : 1,
                track.path[0].b !== undefined ? track.path[0].b : 1
            ]);
        }
    });

    if (initialPoints.length > 0) {
        gaussianSplatProxy = createGaussianProxy(initialPoints, embeddings);
        scene.add(gaussianSplatProxy.mesh);
    }
}

fetch('trajectory_data.json')
    .then(response => {
        if (!response.ok) throw new Error("trajectory_data.json not found.");
        return response.json();
    })
    .then(data => {
        globalTrajectoryData = data;
        buildTrajectories(data);
    })
    .catch(err => console.error("Data Loading Error:", err));

function sync3DWithVideo() {
    if (!globalTrajectoryData) return;

    const currentFrame = Math.floor(video.currentTime * globalTrajectoryData.fps);
    frameDisplay.innerText = currentFrame;

    for (let i = 0; i < trajectoryLines.length; i++) {
        const lineObj = trajectoryLines[i];
        const path = lineObj.data;
        const targetIdx = Math.min(currentFrame, path.length - 1);

        if (targetIdx >= 0) {
            lineObj.line.geometry.setDrawRange(0, targetIdx);
            const currentPos = path[targetIdx];

            // CRITICAL FIX: Manipulating the exact BufferAttribute array for points
            if (gaussianSplatProxy && gaussianSplatProxy.mesh.geometry.attributes.position) {
                const positions = gaussianSplatProxy.mesh.geometry.attributes.position.array;
                positions[i * 3 + 0] = currentPos.x;
                positions[i * 3 + 1] = currentPos.y;
                positions[i * 3 + 2] = currentPos.z;
            }
        }
    }

    // Tell the GPU the vertices have moved
    if (gaussianSplatProxy && gaussianSplatProxy.mesh.geometry.attributes.position) {
        gaussianSplatProxy.mesh.geometry.attributes.position.needsUpdate = true;
    }
}

// video.addEventListener('timeupdate', sync3DWithVideo);
// video.addEventListener('seeked', sync3DWithVideo);

function animate() {
    requestAnimationFrame(animate);
    if (globalTrajectoryData) {
        updateDashboard(globalTrajectoryData.tracks.length);
        sync3DWithVideo();
    }
    renderer.render(scene, camera);
}
animate();
