self.onmessage = function(e) {
    const { buffer, pointCount, viewMatrix } = e.data;

    const vMatrix = viewMatrix || [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
    const positions = new Float32Array(buffer);

    const depths = new Int32Array(pointCount);
    const indices = new Uint32Array(pointCount);

    let minDepth = Infinity;
    let maxDepth = -Infinity;

    // Calculate depth in view space
    for (let i = 0; i < pointCount; i++) {
        const offset = i * 7;
        const px = positions[offset + 0];
        const py = positions[offset + 1];
        const pz = positions[offset + 2];

        const depth = Math.floor(
            (vMatrix[2] * px +
             vMatrix[6] * py +
             vMatrix[10] * pz +
             vMatrix[14]) * 1000
        );

        depths[i] = depth;
        indices[i] = i;

        if (depth < minDepth) minDepth = depth;
        if (depth > maxDepth) maxDepth = depth;
    }

    // Normalize depths to positive integers for Radix Sort
    for (let i = 0; i < pointCount; i++) {
        depths[i] -= minDepth;
    }
    maxDepth -= minDepth;

    // Integer-based Radix Sort
    let exp = 1;
    const tempIndices = new Uint32Array(pointCount);

    while (Math.floor(maxDepth / exp) > 0) {
        const count = new Int32Array(10);

        for (let i = 0; i < pointCount; i++) {
            const digit = Math.floor(depths[indices[i]] / exp) % 10;
            count[digit]++;
        }
        for (let i = 1; i < 10; i++) {
            count[i] += count[i - 1];
        }
        for (let i = pointCount - 1; i >= 0; i--) {
            const digit = Math.floor(depths[indices[i]] / exp) % 10;
            tempIndices[count[digit] - 1] = indices[i];
            count[digit]--;
        }
        for (let i = 0; i < pointCount; i++) {
            indices[i] = tempIndices[i];
        }
        exp *= 10;
    }

    // Sort Descending (Back-to-Front for transparency blending)
    for (let i = 0, j = pointCount - 1; i < j; i++, j--) {
        const tmp = tempIndices[i];
        tempIndices[i] = tempIndices[j];
        tempIndices[j] = tmp;
    }

    // Apply sorted order back into the SharedArrayBuffer safely
    const tempBuffer = new Float32Array(pointCount * 7);
    for (let i = 0; i < pointCount; i++) {
        const sortedIdx = tempIndices[i];
        for (let j = 0; j < 7; j++) {
            tempBuffer[i * 7 + j] = positions[sortedIdx * 7 + j];
        }
    }
    positions.set(tempBuffer);

    self.postMessage({ sorted: true });
};
