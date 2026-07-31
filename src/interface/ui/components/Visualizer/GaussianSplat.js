export const createGaussianProxy = (points, embeddings) => {
    const worker = new Worker(new URL('../../workers/GaussianSorter.worker.js', import.meta.url));
    const count = points.length;

    const sab = new SharedArrayBuffer(count * 4 * 7);
    const dummyViewMatrix = new Float32Array([1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]);
    worker.postMessage({ buffer: sab, pointCount: count, viewMatrix: dummyViewMatrix });

    const geometry = new THREE.BufferGeometry();
    const positions = new Float32Array(count * 3);
    const colors = new Float32Array(count * 3);

    points.forEach((p, i) => {
        positions[i * 3 + 0] = p.x;
        positions[i * 3 + 1] = p.y;
        positions[i * 3 + 2] = p.z;
        colors[i * 3 + 0] = embeddings[i][0] || 1;
        colors[i * 3 + 1] = embeddings[i][1] || 1;
        colors[i * 3 + 2] = embeddings[i][2] || 1;
    });

    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

    const material = new THREE.ShaderMaterial({
        vertexColors: true, // CRITICAL FIX: Injects 'color' into the shader
        vertexShader: `
            varying vec3 vColor;
            void main() {
                vColor = color;
                vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
                gl_Position = projectionMatrix * mvPosition;
                gl_PointSize = 40.0 * (1.0 / -mvPosition.z);
            }
        `,
        fragmentShader: `
            varying vec3 vColor;
            void main() {
                float r2 = dot(gl_PointCoord - 0.5, gl_PointCoord - 0.5);
                if (r2 > 0.25) discard;
                gl_FragColor = vec4(vColor, exp(-r2 * 6.0));
            }
        `,
        transparent: true,
        depthWrite: false,
        blending: THREE.AdditiveBlending
    });

    const mesh = new THREE.Points(geometry, material);
    return { mesh, buffer: sab };
};
