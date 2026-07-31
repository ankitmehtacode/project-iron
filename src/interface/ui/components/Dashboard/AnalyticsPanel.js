let lastTime = performance.now();
let frames = 0;
let currentFps = 0;

const calculateFPS = () => {
    const now = performance.now();
    frames++;
    if (now >= lastTime + 1000) {
        currentFps = Math.round((frames * 1000) / (now - lastTime));
        frames = 0;
        lastTime = now;
    }
    return currentFps;
};

export const updateDashboard = (activeTracks) => {
    const fps = calculateFPS();

    // Failsafe: performance.memory is a Chrome-specific API.
    // If testing in Firefox/Safari, it will return 'N/A' instead of crashing.
    const mem = performance.memory ? (performance.memory.usedJSHeapSize / 1048576).toFixed(2) : 'N/A';

    const panel = document.getElementById('analytics-panel');
    if (panel) {
        panel.innerHTML = `
            <div>Active Tracks: ${activeTracks}</div>
            <div>RAM Usage: ${mem} MB</div>
            <div>FPS: ${fps}</div>
        `;
    }
};
