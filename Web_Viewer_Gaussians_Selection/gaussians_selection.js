let defaultViewMatrix = [0.47, 0.04, 0.88, 0, -0.11, 0.99, 0.02, 0, -0.88, -0.11, 0.47, 0, 0.07, 0.03, 6.55, 1,];
let viewMatrix = defaultViewMatrix;

// Selection state management
let selectionMode = false;  // Controls if selection mode is active
const NO_SELECTION = -999999;  // A value that won't match any real label
let selectedLabel = NO_SELECTION;  // Currently selected label
let displacementManager;

const Z_NEAR = 0.2;
const Z_FAR = 200;

let colorManager; // Will be initialized in main()

function setupColorUniforms(gl, program) {
    // Getting uniform locations
    const uniforms = {
        u_customColors: gl.getUniformLocation(program, "u_customColors"),
        u_customColorLabels: gl.getUniformLocation(program, "u_customColorLabels"),
        u_numCustomColors: gl.getUniformLocation(program, "u_numCustomColors")
    };

    const colorMap = new Map();

    // Function to update shader uniforms with current colors
    function updateColorUniforms() {
        const colorArray = new Float32Array(300); // 100 vec3s
        const labelArray = new Int32Array(100);
        let index = 0;

        for (const [label, color] of colorMap.entries()) {
            labelArray[index] = label;
            colorArray[index * 3] = color[0];
            colorArray[index * 3 + 1] = color[1];
            colorArray[index * 3 + 2] = color[2];
            index++;
        }

        gl.uniform3fv(uniforms.u_customColors, colorArray);
        gl.uniform1iv(uniforms.u_customColorLabels, labelArray);
        gl.uniform1i(uniforms.u_numCustomColors, colorMap.size);
    }

    return {
        updateColor: (label, color) => {
            if (label === NO_SELECTION) return;
            colorMap.set(label, color);
            updateColorUniforms();
        },

        resetColor: (label) => {
            if (label === NO_SELECTION) return;
            colorMap.delete(label);
            updateColorUniforms();
        },

        resetAllColors: () => {
            colorMap.clear();
            updateColorUniforms();
        },

        updateColorUniforms
    };
}

function calculateProjectionMatrix(focalX, focalY, width, height) {
    const zRange = Z_FAR - Z_NEAR;
    const zRangeFactor = Z_FAR / zRange;
    const zOffset = -(Z_FAR * Z_NEAR) / zRange;

    const projectionMatrix = [
        [(2 * focalX) / width, 0, 0, 0],
        [0, -(2 * focalY) / height, 0, 0],
        [0, 0, zRangeFactor, 1],
        [0, 0, zOffset, 0],
    ];

    return projectionMatrix.flat();
}

function getViewMatrix(camera) {
    const rotationMatrix = camera.rotation.flat(); // Extracting the rotation as a constant
    const cameraPosition = camera.position;

    // Helper function to calculate translation part of the matrix
    function calculateTranslationPart(index) {
        return (
            -cameraPosition[0] * rotationMatrix[index] -
            cameraPosition[1] * rotationMatrix[index + 3] -
            cameraPosition[2] * rotationMatrix[index + 6]
        );
    }

    const camToWorld = [
        [rotationMatrix[0], rotationMatrix[1], rotationMatrix[2], 0],
        [rotationMatrix[3], rotationMatrix[4], rotationMatrix[5], 0],
        [rotationMatrix[6], rotationMatrix[7], rotationMatrix[8], 0],
        [
            calculateTranslationPart(0),
            calculateTranslationPart(1),
            calculateTranslationPart(2),
            1,
        ],
    ].flat();

    return camToWorld;
}


function multiply4(matrixA, matrixB) {
    const multiplyRowByCol = (row, col) =>
        matrixB[row] * matrixA[col] +
        matrixB[row + 1] * matrixA[col + 4] +
        matrixB[row + 2] * matrixA[col + 8] +
        matrixB[row + 3] * matrixA[col + 12];

    return [
        multiplyRowByCol(0, 0), multiplyRowByCol(0, 1), multiplyRowByCol(0, 2), multiplyRowByCol(0, 3),
        multiplyRowByCol(4, 0), multiplyRowByCol(4, 1), multiplyRowByCol(4, 2), multiplyRowByCol(4, 3),
        multiplyRowByCol(8, 0), multiplyRowByCol(8, 1), multiplyRowByCol(8, 2), multiplyRowByCol(8, 3),
        multiplyRowByCol(12, 0), multiplyRowByCol(12, 1), multiplyRowByCol(12, 2), multiplyRowByCol(12, 3),
    ];
}

function invert4(matrix) {
    const computeCofactor = (i, j, k, l, m, n, o, p) =>
        matrix[i] * matrix[j] - matrix[k] * matrix[l];

    const cofactors = [
        computeCofactor(0, 5, 1, 4),
        computeCofactor(0, 6, 2, 4),
        computeCofactor(0, 7, 3, 4),
        computeCofactor(1, 6, 2, 5),
        computeCofactor(1, 7, 3, 5),
        computeCofactor(2, 7, 3, 6),
        computeCofactor(8, 13, 9, 12),
        computeCofactor(8, 14, 10, 12),
        computeCofactor(8, 15, 11, 12),
        computeCofactor(9, 14, 10, 13),
        computeCofactor(9, 15, 11, 13),
        computeCofactor(10, 15, 11, 14)
    ];

    const calculateDeterminant = () =>
        cofactors[0] * cofactors[11] -
        cofactors[1] * cofactors[10] +
        cofactors[2] * cofactors[9] +
        cofactors[3] * cofactors[8] -
        cofactors[4] * cofactors[7] +
        cofactors[5] * cofactors[6];

    const det = calculateDeterminant();
    if (!det) return null;

    return [
        (matrix[5] * cofactors[11] - matrix[6] * cofactors[10] + matrix[7] * cofactors[9]) / det,
        (matrix[2] * cofactors[10] - matrix[1] * cofactors[11] - matrix[3] * cofactors[9]) / det,
        (matrix[13] * cofactors[5] - matrix[14] * cofactors[4] + matrix[15] * cofactors[3]) / det,
        (matrix[10] * cofactors[4] - matrix[9] * cofactors[5] - matrix[11] * cofactors[3]) / det,
        (matrix[6] * cofactors[8] - matrix[4] * cofactors[11] - matrix[7] * cofactors[7]) / det,
        (matrix[0] * cofactors[11] - matrix[2] * cofactors[8] + matrix[3] * cofactors[7]) / det,
        (matrix[14] * cofactors[2] - matrix[12] * cofactors[5] - matrix[15] * cofactors[1]) / det,
        (matrix[8] * cofactors[5] - matrix[10] * cofactors[2] + matrix[11] * cofactors[1]) / det,
        (matrix[4] * cofactors[10] - matrix[5] * cofactors[8] + matrix[7] * cofactors[6]) / det,
        (matrix[1] * cofactors[8] - matrix[0] * cofactors[10] - matrix[3] * cofactors[6]) / det,
        (matrix[12] * cofactors[4] - matrix[13] * cofactors[2] + matrix[15] * cofactors[0]) / det,
        (matrix[9] * cofactors[2] - matrix[8] * cofactors[4] - matrix[11] * cofactors[0]) / det,
        (matrix[5] * cofactors[7] - matrix[4] * cofactors[9] - matrix[6] * cofactors[6]) / det,
        (matrix[0] * cofactors[9] - matrix[1] * cofactors[7] + matrix[2] * cofactors[6]) / det,
        (matrix[13] * cofactors[1] - matrix[12] * cofactors[3] - matrix[14] * cofactors[0]) / det,
        (matrix[8] * cofactors[3] - matrix[9] * cofactors[1] + matrix[10] * cofactors[0]) / det,
    ];
}

function rotateMatrix4(matrix, angleRad, axisX, axisY, axisZ) {
    const len = Math.hypot(axisX, axisY, axisZ);
    const [x, y, z] = [axisX / len, axisY / len, axisZ / len]; // Normalized rotation axis

    const sin = Math.sin(angleRad);
    const cos = Math.cos(angleRad);
    const oneMinusCos = 1 - cos;

    const rotationMatrix = calculateRotationMatrix(x, y, z, sin, cos, oneMinusCos);

    const rotatedMatrix = [
        matrix[0] * rotationMatrix[0] + matrix[4] * rotationMatrix[1] + matrix[8] * rotationMatrix[2],
        matrix[1] * rotationMatrix[0] + matrix[5] * rotationMatrix[1] + matrix[9] * rotationMatrix[2],
        matrix[2] * rotationMatrix[0] + matrix[6] * rotationMatrix[1] + matrix[10] * rotationMatrix[2],
        matrix[3] * rotationMatrix[0] + matrix[7] * rotationMatrix[1] + matrix[11] * rotationMatrix[2],

        matrix[0] * rotationMatrix[3] + matrix[4] * rotationMatrix[4] + matrix[8] * rotationMatrix[5],
        matrix[1] * rotationMatrix[3] + matrix[5] * rotationMatrix[4] + matrix[9] * rotationMatrix[5],
        matrix[2] * rotationMatrix[3] + matrix[6] * rotationMatrix[4] + matrix[10] * rotationMatrix[5],
        matrix[3] * rotationMatrix[3] + matrix[7] * rotationMatrix[4] + matrix[11] * rotationMatrix[5],

        matrix[0] * rotationMatrix[6] + matrix[4] * rotationMatrix[7] + matrix[8] * rotationMatrix[8],
        matrix[1] * rotationMatrix[6] + matrix[5] * rotationMatrix[7] + matrix[9] * rotationMatrix[8],
        matrix[2] * rotationMatrix[6] + matrix[6] * rotationMatrix[7] + matrix[10] * rotationMatrix[8],
        matrix[3] * rotationMatrix[6] + matrix[7] * rotationMatrix[7] + matrix[11] * rotationMatrix[8],

        ...matrix.slice(12, 16), // Copying remaining elements unchanged
    ];

    return rotatedMatrix;
}

function calculateRotationMatrix(x, y, z, sin, cos, oneMinusCos) {
    return [
        x * x * oneMinusCos + cos, y * x * oneMinusCos + z * sin, z * x * oneMinusCos - y * sin,
        x * y * oneMinusCos - z * sin, y * y * oneMinusCos + cos, z * y * oneMinusCos + x * sin,
        x * z * oneMinusCos + y * sin, y * z * oneMinusCos - x * sin, z * z * oneMinusCos + cos
    ];
}

function translateMatrix4(matrix, x, y, z) {
    const computeTranslation = (rowIndex) =>
        matrix[rowIndex] * x + matrix[rowIndex + 4] * y + matrix[rowIndex + 8] * z + matrix[rowIndex + 12];

    const translatedX = computeTranslation(0);
    const translatedY = computeTranslation(1);
    const translatedZ = computeTranslation(2);
    const translatedW = computeTranslation(3);

    return [
        ...matrix.slice(0, 12),
        translatedX,
        translatedY,
        translatedZ,
        translatedW
    ];
}

function createWorker(self) {
    let buffer;
    let vertexCount = 0;
    let viewProj;
    const rowLength = 3 * 4 + 3 * 4 + 4 + 4
    let lastProj = [];
    let depthIndex = new Uint32Array();
    let lastVertexCount = 0;
    let labelData;  // New array to store labels
    let displacementMap = new Map(); // label -> (x,y,z) offset
    let visibilityMap = new Map();

    var _floatView = new Float32Array(1);
    var _int32View = new Int32Array(_floatView.buffer);

    function floatToHalf(float) {
        _floatView[0] = float;
        var f = _int32View[0];

        var sign = (f >> 31) & 0x0001;
        var exp = (f >> 23) & 0x00ff;
        var frac = f & 0x007fffff;

        var newExp;
        if (exp == 0) {
            newExp = 0;
        } else if (exp < 113) {
            newExp = 0;
            frac |= 0x00800000;
            frac = frac >> (113 - exp);
            if (frac & 0x01000000) {
                newExp = 1;
                frac = 0;
            }
        } else if (exp < 142) {
            newExp = exp - 112;
        } else {
            newExp = 31;
            frac = 0;
        }

        return (sign << 15) | (newExp << 10) | (frac >> 13);
    }

    function packHalf2x16(x, y) {
        return (floatToHalf(x) | (floatToHalf(y) << 16)) >>> 0;
    }

    function resetAllVisibility() {
        visibilityMap.clear();
        generateTexture(); // Regenerating texture with all objects visible
    }

    function generateTexture() {
        if (!buffer) return;
        const f_buffer = new Float32Array(buffer);
        const u_buffer = new Uint8Array(buffer);

        var texwidth = 1024 * 2;
        var texheight = Math.ceil((2 * vertexCount) / texwidth);
        var texdata = new Uint32Array(texwidth * texheight * 4);
        var texdata_c = new Uint8Array(texdata.buffer);
        var texdata_f = new Float32Array(texdata.buffer);

        if (!labelData || labelData.length !== vertexCount) {
            labelData = new Int32Array(vertexCount);
        }

        for (let i = 0; i < vertexCount; i++) {
            const label = labelData[i];
            // Checking if this splat should be visible
            const isVisible = !visibilityMap.has(label) || visibilityMap.get(label);

            // If not visible, set alpha to 0
            const alphaMultiplier = isVisible ? 1 : 0;

            const displacement = displacementMap.get(label) || {x: 0, y: 0, z: 0};

            texdata_f[8 * i + 0] = f_buffer[8 * i + 0] + displacement.x;
            texdata_f[8 * i + 1] = f_buffer[8 * i + 1] + displacement.y;
            texdata_f[8 * i + 2] = f_buffer[8 * i + 2] + displacement.z;
            texdata_f[8 * i + 3] = label;

            // Applying visibility to alpha channel
            texdata_c[4 * (8 * i + 7) + 0] = u_buffer[32 * i + 24 + 0];
            texdata_c[4 * (8 * i + 7) + 1] = u_buffer[32 * i + 24 + 1];
            texdata_c[4 * (8 * i + 7) + 2] = u_buffer[32 * i + 24 + 2];
            texdata_c[4 * (8 * i + 7) + 3] = u_buffer[32 * i + 24 + 3] * alphaMultiplier;

            let scale = [f_buffer[8 * i + 3 + 0], f_buffer[8 * i + 3 + 1], f_buffer[8 * i + 3 + 2]];
            let rot = [
                (u_buffer[32 * i + 28 + 0] - 128) / 128,
                (u_buffer[32 * i + 28 + 1] - 128) / 128,
                (u_buffer[32 * i + 28 + 2] - 128) / 128,
                (u_buffer[32 * i + 28 + 3] - 128) / 128
            ];

            const M = [
                1.0 - 2.0 * (rot[2] * rot[2] + rot[3] * rot[3]),
                2.0 * (rot[1] * rot[2] + rot[0] * rot[3]),
                2.0 * (rot[1] * rot[3] - rot[0] * rot[2]),
                2.0 * (rot[1] * rot[2] - rot[0] * rot[3]),
                1.0 - 2.0 * (rot[1] * rot[1] + rot[3] * rot[3]),
                2.0 * (rot[2] * rot[3] + rot[0] * rot[1]),
                2.0 * (rot[1] * rot[3] + rot[0] * rot[2]),
                2.0 * (rot[2] * rot[3] - rot[0] * rot[1]),
                1.0 - 2.0 * (rot[1] * rot[1] + rot[2] * rot[2])
            ].map((k, i) => k * scale[Math.floor(i / 3)]);

            const sigma = [
                M[0] * M[0] + M[3] * M[3] + M[6] * M[6],
                M[0] * M[1] + M[3] * M[4] + M[6] * M[7],
                M[0] * M[2] + M[3] * M[5] + M[6] * M[8],
                M[1] * M[1] + M[4] * M[4] + M[7] * M[7],
                M[1] * M[2] + M[4] * M[5] + M[7] * M[8],
                M[2] * M[2] + M[5] * M[5] + M[8] * M[8]
            ];

            texdata[8 * i + 4] = packHalf2x16(4 * sigma[0], 4 * sigma[1]);
            texdata[8 * i + 5] = packHalf2x16(4 * sigma[2], 4 * sigma[3]);
            texdata[8 * i + 6] = packHalf2x16(4 * sigma[4], 4 * sigma[5]);
        }

        self.postMessage({texdata, texwidth, texheight}, [texdata.buffer]);
    }


    // Hit testing function for selection with depth testing
    function performHitTesting(x, y, viewMatrix, projectionMatrix, viewport) {
        if (!buffer || !labelData) return NO_SELECTION;

        const combinedMatrix = multiply4(projectionMatrix, viewMatrix);
        let closestDist = Infinity;
        let closestDepth = Infinity; // Variable to track the closest depth
        let selectedLabel = NO_SELECTION;
        const f_buffer = new Float32Array(buffer);

        for (let i = 0; i < vertexCount; i++) {
            // Getting the position from the buffer
            const pos = [f_buffer[8 * i + 0], f_buffer[8 * i + 1], f_buffer[8 * i + 2], 1.0];

            // Projecting point
            const projected = project(pos, combinedMatrix);
            if (!projected) continue;

            // Converting to screen coordinates
            const screenX = (projected[0] / projected[3] + 1) * 0.5 * viewport[0];
            const screenY = (projected[1] / projected[3] + 1) * 0.5 * viewport[1];
            const depth = projected[2] / projected[3]; // Normalized device coordinates (NDC) depth

            // Calculating distance to click point
            const dist = Math.hypot(screenX - x, screenY - y);

            // Updating if this is the closest point within threshold and with nearest depth
            if (dist < 10 && (dist < closestDist || (dist === closestDist && depth < closestDepth))) {
                closestDist = dist;
                closestDepth = depth;
                selectedLabel = labelData[i];
            }
        }

        return selectedLabel;
    }

    function project(pos, matrix) {
        const result = [0, 0, 0, 0];
        for (let i = 0; i < 4; i++) {
            result[i] = pos[0] * matrix[i] + pos[1] * matrix[i + 4] + pos[2] * matrix[i + 8] + pos[3] * matrix[i + 12];
        }
        if (result[3] <= 0) return null;
        return result;
    }


    function project(pos, matrix) {
        const result = [0, 0, 0, 0];
        for (let i = 0; i < 4; i++) {
            result[i] = pos[0] * matrix[i] + pos[1] * matrix[i + 4] + pos[2] * matrix[i + 8] + pos[3] * matrix[i + 12];
        }
        if (result[3] <= 0) return null;
        return result;
    }


    function runSort(viewProj) {
        if (!buffer) return;
        const f_buffer = new Float32Array(buffer);

        if (lastVertexCount == vertexCount) {
            let dot = lastProj[2] * viewProj[2] + lastProj[6] * viewProj[6] + lastProj[10] * viewProj[10];
            if (Math.abs(dot - 1) < 0.01) {
                return;
            }
        } else {
            generateTexture();
            lastVertexCount = vertexCount;
        }

        //console.time("sort");
        let maxDepth = -Infinity;
        let minDepth = Infinity;
        let sizeList = new Int32Array(vertexCount);

        for (let i = 0; i < vertexCount; i++) {
            let depth = ((viewProj[2] * f_buffer[8 * i + 0] + viewProj[6] * f_buffer[8 * i + 1] + viewProj[10] * f_buffer[8 * i + 2]) * 4096) | 0;
            sizeList[i] = depth;
            if (depth > maxDepth) maxDepth = depth;
            if (depth < minDepth) minDepth = depth;
        }

        let depthInv = (256 * 256) / (maxDepth - minDepth);
        let counts0 = new Uint32Array(256 * 256);
        for (let i = 0; i < vertexCount; i++) {
            sizeList[i] = ((sizeList[i] - minDepth) * depthInv) | 0;
            counts0[sizeList[i]]++;
        }

        let starts0 = new Uint32Array(256 * 256);
        for (let i = 1; i < 256 * 256; i++) starts0[i] = starts0[i - 1] + counts0[i - 1];

        depthIndex = new Uint32Array(vertexCount);
        for (let i = 0; i < vertexCount; i++) {
            const sortedIndex = starts0[sizeList[i]]++;
            depthIndex[sortedIndex] = i;
        }
        //console.timeEnd("sort");
        lastProj = viewProj;

        self.postMessage({depthIndex, viewProj, vertexCount}, [depthIndex.buffer]);
    }

    function processPlyBuffer(inputBuffer) {
        const ubuf = new Uint8Array(inputBuffer);
        const header = new TextDecoder().decode(ubuf.slice(0, 1024 * 10));
        const header_end = "end_header\n";
        const header_end_index = header.indexOf(header_end);

        if (header_end_index < 0) {
            console.error("Invalid PLY header");
            return null;
        }

        const vertexCount = parseInt(/element vertex (\d+)\n/.exec(header)[1]);
        console.log("Processing PLY with vertex count:", vertexCount);
        let row_offset = 0;
        const offsets = {};
        const types = {};

        const TYPE_MAP = {
            double: "getFloat64",
            int: "getInt32",
            uint: "getUint32",
            float: "getFloat32",
            short: "getInt16",
            ushort: "getUint16",
            uchar: "getUint8",
        };

        for (let prop of header
            .slice(0, header_end_index)
            .split("\n")
            .filter((k) => k.startsWith("property "))) {
            const [p, type, name] = prop.split(" ");
            const arrayType = TYPE_MAP[type] || "getInt8";
            types[name] = arrayType;
            offsets[name] = row_offset;
            row_offset += parseInt(arrayType.replace(/[^\d]/g, "")) / 8;
        }
        console.log("Bytes per row", row_offset, types, offsets);

        let dataView = new DataView(inputBuffer, header_end_index + header_end.length);
        let row = 0;

        const attrs = new Proxy({}, {
            get(target, prop) {
                if (!types[prop]) throw new Error(prop + " not found");
                return dataView[types[prop]](row * row_offset + offsets[prop], true);
            }
        });

        // Calculating importance and sorting indices
        console.time("calculate importance");
        let sizeList = new Float32Array(vertexCount);
        let sizeIndex = new Uint32Array(vertexCount);
        for (row = 0; row < vertexCount; row++) {
            sizeIndex[row] = row;
            if (!types["scale_0"]) continue;
            const size = Math.exp(attrs.scale_0) * Math.exp(attrs.scale_1) * Math.exp(attrs.scale_2);
            const opacity = 1 / (1 + Math.exp(-attrs.opacity));
            sizeList[row] = size * opacity;
        }
        console.timeEnd("calculate importance");

        console.time("sort");
        sizeIndex.sort((b, a) => sizeList[a] - sizeList[b]);
        console.timeEnd("sort");

        // Creating buffer with space for labels
        const buffer = new ArrayBuffer(rowLength * vertexCount);
        labelData = new Int32Array(vertexCount);  // Create separate label array


        console.time("build buffer");
        for (let j = 0; j < vertexCount; j++) {
            row = sizeIndex[j];  // Use sorted index

            const position = new Float32Array(buffer, j * rowLength, 3);
            const scales = new Float32Array(buffer, j * rowLength + 4 * 3, 3);
            const rgba = new Uint8ClampedArray(buffer, j * rowLength + 4 * 3 + 4 * 3, 4);
            const rot = new Uint8ClampedArray(buffer, j * rowLength + 4 * 3 + 4 * 3 + 4, 4);

            position[0] = attrs.x;
            position[1] = attrs.y;
            position[2] = attrs.z;

            if (types["scale_0"]) {
                const qlen = Math.sqrt(attrs.rot_0 ** 2 + attrs.rot_1 ** 2 + attrs.rot_2 ** 2 + attrs.rot_3 ** 2);

                rot[0] = (attrs.rot_0 / qlen) * 128 + 128;
                rot[1] = (attrs.rot_1 / qlen) * 128 + 128;
                rot[2] = (attrs.rot_2 / qlen) * 128 + 128;
                rot[3] = (attrs.rot_3 / qlen) * 128 + 128;

                scales[0] = Math.exp(attrs.scale_0);
                scales[1] = Math.exp(attrs.scale_1);
                scales[2] = Math.exp(attrs.scale_2);
            } else {
                scales[0] = scales[1] = scales[2] = 0.01;
                rot[0] = 255;
                rot[1] = rot[2] = rot[3] = 0;
            }

            if (types["f_dc_0"]) {
                const SH_C0 = 0.28209479177387814;
                rgba[0] = (0.5 + SH_C0 * attrs.f_dc_0) * 255;
                rgba[1] = (0.5 + SH_C0 * attrs.f_dc_1) * 255;
                rgba[2] = (0.5 + SH_C0 * attrs.f_dc_2) * 255;
            } else {
                rgba[0] = attrs.red;
                rgba[1] = attrs.green;
                rgba[2] = attrs.blue;
            }

            rgba[3] = types["opacity"] ? (1 / (1 + Math.exp(-attrs.opacity))) * 255 : 255;

            // Storing label
            labelData[j] = types["label"] ? attrs.label : NO_SELECTION
        }
        console.timeEnd("build buffer");
        console.log("Processing complete with vertex count:", vertexCount);

        return buffer;
    }

    const throttledSort = () => {
        if (!sortRunning) {
            sortRunning = true;
            let lastView = viewProj;
            runSort(lastView);
            setTimeout(() => {
                sortRunning = false;
                if (lastView !== viewProj) {
                    throttledSort();
                }
            }, 0);
        }
    };

    let sortRunning;
    self.onmessage = (e) => {
        if (e.data.ply) {
            vertexCount = 0;
            runSort(viewProj);
            buffer = processPlyBuffer(e.data.ply);
            vertexCount = Math.floor(buffer.byteLength / rowLength);
            console.log("Processing complete with vertex count:", vertexCount);
            postMessage({buffer: buffer, vertexCount: vertexCount});
        } else if (e.data.buffer) {
            buffer = e.data.buffer;
            vertexCount = e.data.vertexCount;

        } else if (e.data.type === 'resetVisibility') {
            resetAllVisibility();
            self.postMessage({type: 'visibilityReset'}); // Notifying main thread
        } else if (e.data.type === 'toggleVisibility') {
            const label = e.data.label;
            if (label !== NO_SELECTION) {
                visibilityMap.set(label, e.data.visible);
                generateTexture(); // Regenerating texture to apply visibility changes
            }
        } else if (e.data.vertexCount) {
            vertexCount = e.data.vertexCount;
        } else if (e.data.view) {
            viewProj = e.data.view;
            throttledSort();
        } else if (e.data.type === 'select') {
            // Handling selection request
            const selectedLabel = performHitTesting(e.data.x, e.data.y, e.data.viewMatrix, e.data.projectionMatrix, e.data.viewport);
            self.postMessage({type: 'selection', label: selectedLabel});
        } else if (e.data.type === 'updateDisplacement') {
            // Handling displacement updates
            const {label, dx, dy, dz} = e.data;
            let displacement = displacementMap.get(label) || {x: 0, y: 0, z: 0};
            displacement.x += dx;
            displacement.y += dy;
            displacement.z += dz;
            displacementMap.set(label, displacement);

            // Updating positions in the buffer
            if (buffer) {
                const f_buffer = new Float32Array(buffer);
                for (let i = 0; i < vertexCount; i++) {
                    const label_i = labelData[i];
                    if (label_i === label) {
                        const positionOffset = 8 * i;
                        f_buffer[positionOffset] += dx;
                        f_buffer[positionOffset + 1] += dy;
                        f_buffer[positionOffset + 2] += dz;
                    }
                }
                // regenerating the texture after moving things around
                generateTexture();
                self.postMessage({buffer: buffer, vertexCount: vertexCount});
            }
        }
    };
}

const vertexShaderSource = `#version 300 es
precision highp float;
precision highp int;

uniform highp usampler2D u_texture;
uniform mat4 projection;
uniform mat4 view;
uniform vec2 focal;
uniform vec2 viewport;
uniform bool uSelectionMode;
uniform int uSelectedLabel;

// Displacement uniforms
uniform bool u_enableDisplacement;
uniform vec3 u_displacementMap[100]; // Array to store displacements for different labels
uniform int u_displacementLabels[100]; // Array to store which labels are displaced
uniform int u_numDisplacements; // Number of active displacements

in vec2 position;
in int index;

flat out int vLabel;
out vec4 vColor;
out vec2 vPosition;

// Function to get displacement for a label
vec3 getDisplacement(int label) {
    for(int i = 0; i < u_numDisplacements; i++) {
        if(u_displacementLabels[i] == label) {
            return u_displacementMap[i];
        }
    }
    return vec3(0.0);
}

void main() {
    uvec4 cen = texelFetch(u_texture, ivec2((uint(index) & 0x3ffu) << 1, uint(index) >> 10), 0);
    vec3 centerPosition = uintBitsToFloat(cen.xyz);
    vLabel = int(uintBitsToFloat(cen.w));
    
    // Applying displacement if enabled
    if(u_enableDisplacement) {
        centerPosition += getDisplacement(vLabel);
    }
    
    vec4 cam = view * vec4(centerPosition, 1.0);
    vec4 pos2d = projection * cam;
    
    float clip = 1.2 * pos2d.w;
    if (pos2d.z < -clip || pos2d.x < -clip || pos2d.x > clip || pos2d.y < -clip || pos2d.y > clip) {
        gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
        return;
    }

    // Covariance calculation 
    uvec4 cov = texelFetch(u_texture, ivec2(((uint(index) & 0x3ffu) << 1) | 1u, uint(index) >> 10), 0);
    vec2 u1 = unpackHalf2x16(cov.x);
    vec2 u2 = unpackHalf2x16(cov.y);
    vec2 u3 = unpackHalf2x16(cov.z);
    mat3 Vrk = mat3(u1.x, u1.y, u2.x, u1.y, u2.y, u3.x, u2.x, u3.x, u3.y);

    mat3 J = mat3(
        focal.x / cam.z, 0., -(focal.x * cam.x) / (cam.z * cam.z),
        0., -focal.y / cam.z, (focal.y * cam.y) / (cam.z * cam.z),
        0., 0., 0.
    );

    mat3 T = transpose(mat3(view)) * J;
    mat3 cov2d = transpose(T) * Vrk * T;

    float mid = (cov2d[0][0] + cov2d[1][1]) / 2.0;
    float radius = length(vec2((cov2d[0][0] - cov2d[1][1]) / 2.0, cov2d[0][1]));
    float lambda1 = mid + radius;
    float lambda2 = mid - radius;

    if(lambda2 < 0.0) return;
    vec2 diagonalVector = normalize(vec2(cov2d[0][1], lambda1 - cov2d[0][0]));
    vec2 majorAxis = min(sqrt(2.0 * lambda1), 1024.0) * diagonalVector;
    vec2 minorAxis = min(sqrt(2.0 * lambda2), 1024.0) * vec2(diagonalVector.y, -diagonalVector.x);

    vColor = clamp(pos2d.z/pos2d.w+1.0, 0.0, 1.0) * 
            vec4((cov.w) & 0xffu, (cov.w >> 8) & 0xffu, (cov.w >> 16) & 0xffu, (cov.w >> 24) & 0xffu) / 255.0;
    vPosition = position;

    vec2 vCenter = vec2(pos2d) / pos2d.w;
    gl_Position = vec4(
        vCenter
        + position.x * majorAxis / viewport
        + position.y * minorAxis / viewport, 0.0, 1.0);
}
`.trim();

const fragmentShaderSource = `#version 300 es
precision highp float;
precision highp int;

uniform bool uSelectionMode;
uniform int uSelectedLabel;

uniform bool u_enableCustomColor;
uniform vec3 u_customColor;
uniform vec3 u_customColors[100]; // Array to store custom colors
uniform int u_customColorLabels[100]; // Array to store which labels have custom colors
uniform int u_numCustomColors; // Number of custom colors

flat in int vLabel;
in vec4 vColor;
in vec2 vPosition;

out vec4 fragColor;

// Function to get custom color for a label
vec3 getCustomColor(int label) {
    for(int i = 0; i < u_numCustomColors; i++) {
        if(u_customColorLabels[i] == label) {
            return mix(vColor.rgb, u_customColors[i], 0.6);
        }
    }
    return vColor.rgb;
}

void main() {
    float A = -dot(vPosition, vPosition);
    if (A < -4.0) discard;
    float B = exp(A) * vColor.a;
    // Applying selection color if in selection mode and label matches
    vec3 finalColor = getCustomColor(vLabel);

    // Applying custom color if enabled
    if (u_enableCustomColor && vLabel == uSelectedLabel) {
        finalColor = u_customColor;
    }

    // Applying selection highlight if in selection mode
    if (uSelectionMode && vLabel == uSelectedLabel) {
        finalColor = mix(finalColor, vec3(1.0, 0.0, 0.0), 0.5);
    }
    
    fragColor = vec4(B * finalColor, B);
}`.trim();

function setupFramebuffer(gl) {
    const framebuffer = gl.createFramebuffer();

    if (!framebuffer) {
        console.error("Failed to create framebuffer");
        return null;
    }

    // Creating color texture
    const colorTexture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, colorTexture);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, gl.canvas.width, gl.canvas.height, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

    // Creating depth renderbuffer
    const depthBuffer = gl.createRenderbuffer();
    gl.bindRenderbuffer(gl.RENDERBUFFER, depthBuffer);
    gl.renderbufferStorage(gl.RENDERBUFFER, gl.DEPTH_COMPONENT16, gl.canvas.width, gl.canvas.height);

    // Setting up framebuffer
    gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, colorTexture, 0);
    gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.RENDERBUFFER, depthBuffer);

    // Checking framebuffer status
    if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) {
        console.error('Framebuffer is incomplete');
        return null;
    }

    // Resetting state
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.bindRenderbuffer(gl.RENDERBUFFER, null);
    gl.bindTexture(gl.TEXTURE_2D, null);

    return {framebuffer, colorTexture};
}

async function main() {
    let carousel = true;
    const rowLength = 3 * 4 + 3 * 4 + 4 + 4
    let splatData = null; // Will be initialized when a file is dropped

    try {
        viewMatrix = JSON.parse(decodeURIComponent(location.hash.slice(1)));
        carousel = false;
    } catch (err) {
    }

    const downsample = 1 / devicePixelRatio;

    const colorPicker = document.getElementById('color-picker');
    const applyColorBtn = document.getElementById('apply-color');
    const resetColorBtn = document.getElementById('reset-color');


    document.getElementById("spinner").style.display = "none";
    document.getElementById("message").innerText = "Drop a .ply file to view";

    const worker = new Worker(URL.createObjectURL(new Blob([
        `const NO_SELECTION = ${NO_SELECTION};`,
        "const multiply4 = " + multiply4.toString() + ";",
        "(",
        createWorker.toString(),
        ")(self)"], {
        type: "application/javascript",
    }),),);

    const canvas = document.getElementById("canvas");
    const fps = document.getElementById("fps");
    const camid = document.getElementById("camid");

    let projectionMatrix;

    const gl = canvas.getContext("webgl2", {
        antialias: false,
    });


    // Setting up framebuffer
    const result = setupFramebuffer(gl);
    if (!result) {
        console.error("Failed to set up framebuffer");
        return;
    }
    const {framebuffer, colorTexture} = result;

    function setupDisplacementUniforms(gl, program) {
        // Getting uniform locations
        const uniforms = {
            u_enableDisplacement: gl.getUniformLocation(program, "u_enableDisplacement"),
            u_displacementMap: gl.getUniformLocation(program, "u_displacementMap"),
            u_displacementLabels: gl.getUniformLocation(program, "u_displacementLabels"),
            u_numDisplacements: gl.getUniformLocation(program, "u_numDisplacements")
        };

        // Initializing arrays for displacements
        const displacementMap = new Map(); // label -> [x, y, z]
        let activeDisplacements = 0;

        // Function to update shader uniforms with current displacements
        function updateDisplacementUniforms() {
            const displacementArray = new Float32Array(300); // 100 vec3s
            const labelArray = new Int32Array(100);
            let index = 0;

            for (const [label, displacement] of displacementMap.entries()) {
                labelArray[index] = label;
                displacementArray[index * 3] = displacement[0];
                displacementArray[index * 3 + 1] = displacement[1];
                displacementArray[index * 3 + 2] = displacement[2];
                index++;
            }

            gl.uniform1i(uniforms.u_enableDisplacement, displacementMap.size > 0);
            gl.uniform3fv(uniforms.u_displacementMap, displacementArray);
            gl.uniform1iv(uniforms.u_displacementLabels, labelArray);
            gl.uniform1i(uniforms.u_numDisplacements, displacementMap.size);
        }

        return {
            updateDisplacement: (label, dx = 0, dy = 0, dz = 0) => {
                if (label === NO_SELECTION) return;

                const current = displacementMap.get(label) || [0, 0, 0];
                current[0] += dx;
                current[1] += dy;
                current[2] += dz;
                displacementMap.set(label, current);

                updateDisplacementUniforms();
                saveDisplacements(displacementMap);
            },

            resetDisplacements: () => {
                displacementMap.clear();
                updateDisplacementUniforms();
                saveDisplacements(displacementMap);
            },

            loadDisplacements: (savedDisplacements) => {
                displacementMap.clear();
                for (const [label, displacement] of Object.entries(savedDisplacements)) {
                    displacementMap.set(parseInt(label), displacement);
                }
                updateDisplacementUniforms();
            }
        };
    }

    function loadDisplacements() {
        const saved = localStorage.getItem('splatDisplacements');
        if (saved) {
            const displacementsObj = JSON.parse(saved);
            displacementManager.loadDisplacements(displacementsObj);
        }
    }

    function updateSplatDisplacement(label, dx = 0, dy = 0, dz = 0) {
        if (label === NO_SELECTION || !displacementManager) return;
        displacementManager.updateDisplacement(label, dx, dy, dz);
    }

    function saveDisplacements() {
        const displacementsObj = Object.fromEntries(displacements);
        localStorage.setItem('splatDisplacements', JSON.stringify(displacementsObj));
    }

    function resetAllDisplacements() {
        if (!displacementManager) return;

        displacementManager.resetDisplacements();
        console.log("All displacements have been reset");
    }

    // Show/hide color controls based on selection
    function updateSelectionInfo(label) {
        const infoEl = document.getElementById('selection-info');
        const objectEl = document.getElementById('selected-object');
        const colorControls = document.getElementById('color-controls');

        console.log('Selection update:', {
            label,
            selectionMode,
            colorControlsExists: !!colorControls,
            infoElExists: !!infoEl
        }); // Debug log

        if (label >= 0 && selectionMode) {
            infoEl.style.display = 'block';
            objectEl.textContent = label// + ' ' + labels[label];
            colorControls.classList.add('visible');
            console.log('Color controls should be visible now:', colorControls.classList.contains('visible')); // Debug log
        } else {
            if (!selectionMode) {
                infoEl.style.display = 'none';
                colorControls.classList.remove('visible');
            }
        }
    }

    // Creating and compiling main vertex shader
    const vertexShader = gl.createShader(gl.VERTEX_SHADER);
    gl.shaderSource(vertexShader, vertexShaderSource);
    gl.compileShader(vertexShader);
    if (!gl.getShaderParameter(vertexShader, gl.COMPILE_STATUS)) console.error(gl.getShaderInfoLog(vertexShader));

    // Creating and compiling main fragment shader
    const fragmentShader = gl.createShader(gl.FRAGMENT_SHADER);
    gl.shaderSource(fragmentShader, fragmentShaderSource);
    gl.compileShader(fragmentShader);
    if (!gl.getShaderParameter(fragmentShader, gl.COMPILE_STATUS)) console.error(gl.getShaderInfoLog(fragmentShader));

    // Creating and linking main program
    const program = gl.createProgram();
    gl.attachShader(program, vertexShader);
    gl.attachShader(program, fragmentShader);
    gl.linkProgram(program);
    gl.useProgram(program);

    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) console.error(gl.getProgramInfoLog(program));
    displacementManager = setupDisplacementUniforms(gl, program);

    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) console.error(gl.getProgramInfoLog(program));
    displacementManager = setupDisplacementUniforms(gl, program);
    colorManager = setupColorUniforms(gl, program);


    gl.disable(gl.DEPTH_TEST);

    // Enabling blending
    gl.enable(gl.BLEND);
    gl.blendFuncSeparate(gl.ONE_MINUS_DST_ALPHA, gl.ONE, gl.ONE_MINUS_DST_ALPHA, gl.ONE,);
    gl.blendEquationSeparate(gl.FUNC_ADD, gl.FUNC_ADD);

    // Getting uniform locations
    // After the other uniform locations...
    const u_projection = gl.getUniformLocation(program, "projection");
    const u_viewport = gl.getUniformLocation(program, "viewport");
    const u_focal = gl.getUniformLocation(program, "focal");
    const u_view = gl.getUniformLocation(program, "view");
    const u_selectionMode = gl.getUniformLocation(program, 'uSelectionMode');
    const u_selectedLabel = gl.getUniformLocation(program, 'uSelectedLabel');
    const u_enableDisplacement = gl.getUniformLocation(program, "u_enableDisplacement");
    const u_displacement = gl.getUniformLocation(program, "u_displacement");
    // Initializing selection uniforms
    const displacements = new Map(); // Track displacements by label
    gl.uniform1i(u_selectionMode, 0);
    gl.uniform1i(u_selectedLabel, NO_SELECTION);

    // Setting up vertex positions
    const triangleVertices = new Float32Array([-2, -2, 2, -2, 2, 2, -2, 2]);
    const vertexBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, vertexBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, triangleVertices, gl.STATIC_DRAW);
    const a_position = gl.getAttribLocation(program, "position");
    gl.enableVertexAttribArray(a_position);
    gl.bindBuffer(gl.ARRAY_BUFFER, vertexBuffer);
    gl.vertexAttribPointer(a_position, 2, gl.FLOAT, false, 0, 0);

    // Setting up texture
    var texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    var u_textureLocation = gl.getUniformLocation(program, "u_texture");
    gl.uniform1i(u_textureLocation, 0);

    // Setting up index buffer
    const indexBuffer = gl.createBuffer();
    const a_index = gl.getAttribLocation(program, "index");
    gl.enableVertexAttribArray(a_index);
    gl.bindBuffer(gl.ARRAY_BUFFER, indexBuffer);
    gl.vertexAttribIPointer(a_index, 1, gl.INT, false, 0, 0);
    gl.vertexAttribDivisor(a_index, 1);

    const resize = () => {
        // Updating uniform values for camera and viewport
        gl.uniform2fv(u_focal, new Float32Array([camera.fx, camera.fy]));
        gl.uniform2fv(u_viewport, new Float32Array([innerWidth, innerHeight]));

        // Updating canvas dimensions
        gl.canvas.width = Math.round(innerWidth / downsample);
        gl.canvas.height = Math.round(innerHeight / downsample);
        gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);

        // Updating projection matrix
        projectionMatrix = calculateProjectionMatrix(camera.fx, camera.fy, innerWidth, innerHeight,);
        gl.uniformMatrix4fv(u_projection, false, projectionMatrix);
    };

    window.addEventListener("resize", resize);
    resize();

    let vertexCount = 0;

    // Handling worker messages
    worker.onmessage = (e) => {
        if (e.data.type === 'selection') {
            selectedLabel = e.data.label;
            gl.uniform1i(u_selectedLabel, selectedLabel);
            updateSelectionInfo(selectedLabel);
        } else if (e.data.type === 'visibilityReset') {
            console.log("All objects visibility has been reset");
        } else if (e.data.buffer) {
            console.log("Received buffer from worker:", {
                bufferSize: e.data.buffer.byteLength, receivedVertexCount: e.data.vertexCount
            });
            // Setting vertex count first
            vertexCount = e.data.vertexCount;
            splatData = new Uint8Array(e.data.buffer);
        } else if (e.data.texdata) {
            const {texdata, texwidth, texheight} = e.data;

            gl.activeTexture(gl.TEXTURE0);
            gl.bindTexture(gl.TEXTURE_2D, texture);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);

            gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32UI, texwidth, texheight, 0, gl.RGBA_INTEGER, gl.UNSIGNED_INT, texdata);
        } else if (e.data.depthIndex) {
            const {depthIndex, viewProj} = e.data;
            gl.bindBuffer(gl.ARRAY_BUFFER, indexBuffer);
            gl.bufferData(gl.ARRAY_BUFFER, depthIndex, gl.DYNAMIC_DRAW);
            vertexCount = e.data.vertexCount;
        }
    };

    let activeKeys = [];
    let currentCameraIndex = 0;

    // Handling color application
    applyColorBtn.addEventListener('click', () => {
        if (selectedLabel === NO_SELECTION) return;

        const color = colorPicker.value;
        // Converting hex to RGB (0-1 range)
        const r = parseInt(color.substr(1, 2), 16) / 255;
        const g = parseInt(color.substr(3, 2), 16) / 255;
        const b = parseInt(color.substr(5, 2), 16) / 255;

        colorManager.updateColor(selectedLabel, [r, g, b]);
    });

    // Handling color reset
    resetColorBtn.addEventListener('click', () => {
        if (selectedLabel === NO_SELECTION) return;
        colorManager.resetColor(selectedLabel);
    });


    window.addEventListener("keydown", (e) => {
        carousel = false;
        if (!activeKeys.includes(e.code)) activeKeys.push(e.code);
        if (/\d/.test(e.key)) {
            currentCameraIndex = parseInt(e.key);
            camera = cameras[currentCameraIndex];
            viewMatrix = getViewMatrix(camera);
        }
        if (["-", "_"].includes(e.key)) {
            currentCameraIndex = (currentCameraIndex + cameras.length - 1) % cameras.length;
            viewMatrix = getViewMatrix(cameras[currentCameraIndex]);
        }
        if (["+", "="].includes(e.key)) {
            currentCameraIndex = (currentCameraIndex + 1) % cameras.length;
            viewMatrix = getViewMatrix(cameras[currentCameraIndex]);
        }
        camid.innerText = "cam  " + currentCameraIndex;
        if (e.code == "KeyV") {
            location.hash = "#" + JSON.stringify(viewMatrix.map((k) => Math.round(k * 100) / 100),);
            camid.innerText = "";
        } else if (e.code === "KeyP") {
            carousel = true;
            camid.innerText = "";
        }
    });
    window.addEventListener("keyup", (e) => {
        activeKeys = activeKeys.filter((k) => k !== e.code);
    });
    window.addEventListener("blur", () => {
        activeKeys = [];
    });

    window.addEventListener("wheel", (e) => {
        carousel = false;
        e.preventDefault();
        const lineHeight = 10;
        const scale = e.deltaMode == 1 ? lineHeight : e.deltaMode == 2 ? innerHeight : 1;
        let inv = invert4(viewMatrix);
        if (e.shiftKey) {
            inv = translateMatrix4(inv, (e.deltaX * scale) / innerWidth, (e.deltaY * scale) / innerHeight, 0,);
        } else if (e.ctrlKey || e.metaKey) {
            inv = translateMatrix4(inv, 0, 0, (-10 * (e.deltaY * scale)) / innerHeight,);
        } else {
            let d = 4;
            inv = translateMatrix4(inv, 0, 0, d);
            inv = rotateMatrix4(inv, -(e.deltaX * scale) / innerWidth, 0, 1, 0);
            inv = rotateMatrix4(inv, (e.deltaY * scale) / innerHeight, 1, 0, 0);
            inv = translateMatrix4(inv, 0, 0, -d);
        }

        viewMatrix = invert4(inv);
    }, {passive: false},);

    let startX, startY, down;
    canvas.addEventListener("mousedown", (e) => {
        carousel = false;
        e.preventDefault();
        startX = e.clientX;
        startY = e.clientY;
        down = e.ctrlKey || e.metaKey ? 2 : 1;
    });
    canvas.addEventListener("contextmenu", (e) => {
        carousel = false;
        e.preventDefault();
        startX = e.clientX;
        startY = e.clientY;
        down = 2;
    });

    canvas.addEventListener("mousemove", (e) => {
        e.preventDefault();
        if (down == 1) {
            let inv = invert4(viewMatrix);
            let dx = (5 * (e.clientX - startX)) / innerWidth;
            let dy = (5 * (e.clientY - startY)) / innerHeight;
            let d = 4;

            inv = translateMatrix4(inv, 0, 0, d);
            inv = rotateMatrix4(inv, dx, 0, 1, 0);
            inv = rotateMatrix4(inv, -dy, 1, 0, 0);
            inv = translateMatrix4(inv, 0, 0, -d);
            viewMatrix = invert4(inv);

            startX = e.clientX;
            startY = e.clientY;
        } else if (down == 2) {
            let inv = invert4(viewMatrix);
            inv = translateMatrix4(inv, (-10 * (e.clientX - startX)) / innerWidth, 0, (10 * (e.clientY - startY)) / innerHeight,);
            viewMatrix = invert4(inv);

            startX = e.clientX;
            startY = e.clientY;
        }
    });
    canvas.addEventListener("mouseup", (e) => {
        e.preventDefault();
        down = false;
        startX = 0;
        startY = 0;
    });

    let altX = 0, altY = 0;
    canvas.addEventListener("touchstart", (e) => {
        e.preventDefault();
        if (e.touches.length === 1) {
            carousel = false;
            startX = e.touches[0].clientX;
            startY = e.touches[0].clientY;
            down = 1;
        } else if (e.touches.length === 2) {
            carousel = false;
            startX = e.touches[0].clientX;
            altX = e.touches[1].clientX;
            startY = e.touches[0].clientY;
            altY = e.touches[1].clientY;
            down = 1;
        }
    }, {passive: false},);
    canvas.addEventListener("touchmove", (e) => {
        e.preventDefault();
        if (e.touches.length === 1 && down) {
            let inv = invert4(viewMatrix);
            let dx = (4 * (e.touches[0].clientX - startX)) / innerWidth;
            let dy = (4 * (e.touches[0].clientY - startY)) / innerHeight;

            let d = 4;
            inv = translateMatrix4(inv, 0, 0, d);
            inv = rotateMatrix4(inv, dx, 0, 1, 0);
            inv = rotateMatrix4(inv, -dy, 1, 0, 0);
            inv = translateMatrix4(inv, 0, 0, -d);

            viewMatrix = invert4(inv);

            startX = e.touches[0].clientX;
            startY = e.touches[0].clientY;
        } else if (e.touches.length === 2) {
            const dtheta = Math.atan2(startY - altY, startX - altX) - Math.atan2(e.touches[0].clientY - e.touches[1].clientY, e.touches[0].clientX - e.touches[1].clientX,);
            const dscale = Math.hypot(startX - altX, startY - altY) / Math.hypot(e.touches[0].clientX - e.touches[1].clientX, e.touches[0].clientY - e.touches[1].clientY,);
            const dx = (e.touches[0].clientX + e.touches[1].clientX - (startX + altX)) / 2;
            const dy = (e.touches[0].clientY + e.touches[1].clientY - (startY + altY)) / 2;
            let inv = invert4(viewMatrix);
            inv = rotateMatrix4(inv, dtheta, 0, 0, 1);

            inv = translateMatrix4(inv, -dx / innerWidth, -dy / innerHeight, 0);

            inv = translateMatrix4(inv, 0, 0, 3 * (1 - dscale));

            viewMatrix = invert4(inv);

            startX = e.touches[0].clientX;
            altX = e.touches[1].clientX;
            startY = e.touches[0].clientY;
            altY = e.touches[1].clientY;
        }
    }, {passive: false},);
    canvas.addEventListener("touchend", (e) => {
        e.preventDefault();
        down = false;
        startX = 0;
        startY = 0;
    }, {passive: false},);


    /*     function updateSplatDisplacement(label, dx = 0, dy = 0, dz = 0) {
            if (label === NO_SELECTION) return;

            // Update displacement map for the specific label
            const current = displacements.get(label) || [0, 0, 0];
            current[0] += dx;
            current[1] += dy;
            current[2] += dz;
            displacements.set(label, current);

            // Save to localStorage
            saveDisplacements();

            // Set the uniforms for the currently selected label
            gl.uniform1i(u_enableDisplacement, 1);
            gl.uniform3fv(u_displacement, new Float32Array(current));
            gl.uniform1i(u_selectedLabel, label);
        } */

    // Selection mode toggle
    document.addEventListener('keydown', (e) => {
        carousel = false;
        if (!activeKeys.includes(e.code)) activeKeys.push(e.code);
        if (e.key === 'Escape') {
            selectionMode = !selectionMode;
            console.log("Selection mode:", selectionMode ? "enabled" : "disabled");
            gl.uniform1i(u_selectionMode, selectionMode ? 1 : 0);
            if (!selectionMode) {
                selectedLabel = NO_SELECTION;
                gl.uniform1i(u_selectedLabel, selectedLabel); // Telling fragment shader to disable selection
                updateSelectionInfo(selectedLabel);
            }
        }
        if (e.key === 'Delete' && selectedLabel !== NO_SELECTION) {
            worker.postMessage({
                type: 'toggleVisibility',
                label: selectedLabel,
                visible: false
            });
            console.log(`Hidden object with label: ${selectedLabel}`);
        }
        if (e.code === "KeyR") {
            resetAllDisplacements();
        }
        if (e.shiftKey && e.code === 'KeyR') {
            worker.postMessage({type: 'resetVisibility'});
            console.log("Resetting visibility of all objects");
        }
        if (e.code === "KeyC") {
            colorManager.resetAllColors();
        }
    });

// Click handler for selection
    canvas.addEventListener('click', (e) => {
        if (!selectionMode) return;

        const rect = canvas.getBoundingClientRect();
        const x = (e.clientX - rect.left) * gl.canvas.width / rect.width;
        const y = (rect.height - (e.clientY - rect.top)) * gl.canvas.height / rect.height;

        worker.postMessage({
            type: 'select',
            x: x,
            y: y,
            viewMatrix: actualViewMatrix,
            projectionMatrix: projectionMatrix,
            viewport: [gl.canvas.width, gl.canvas.height]
        });
    });

    let jumpDelta = 0;

    let lastFrame = 0;
    let avgFps = 0;
    let start = 0;

    window.addEventListener("gamepadconnected", (e) => {
        const gp = navigator.getGamepads()[e.gamepad.index];
        console.log(`Gamepad connected at index ${gp.index}: ${gp.id}. It has ${gp.buttons.length} buttons and ${gp.axes.length} axes.`,);
    });
    window.addEventListener("gamepaddisconnected", (e) => {
        console.log("Gamepad disconnected");
    });

    // State monitoring for debugging
    /*const debugState = () => {
        console.log("Render State:", {
            vertexCount: vertexCount,
            hasValidContext: !!gl,
            programLinked: gl?.getProgramParameter(program, gl.LINK_STATUS),
            activeTexture: gl?.getParameter(gl.ACTIVE_TEXTURE),
            viewport: gl?.getParameter(gl.VIEWPORT),
            error: gl?.getError(),
            bufferStatus: {
                hasIndexBuffer: gl?.isBuffer(indexBuffer),
                hasVertexBuffer: gl?.isBuffer(vertexBuffer),
                hasFramebuffer: gl?.isFramebuffer(framebuffer),
            },
            textureStatus: {
                isTextureBound: gl?.getParameter(gl.TEXTURE_BINDING_2D) !== null,
                activeTextureUnit: gl?.getParameter(gl.ACTIVE_TEXTURE),
                hasTexture: gl?.isTexture(texture),
            },
            shaderStatus: {
                program: program,
                isProgram: gl?.isProgram(program),
                programLinked: gl?.getProgramParameter(program, gl.LINK_STATUS),
                activeProgram: gl?.getParameter(gl.CURRENT_PROGRAM),
            }
        });
    };*/

    // debugState();

    let leftGamepadTrigger, rightGamepadTrigger, actualViewMatrix

    const frame = (now) => {
        let inv = invert4(viewMatrix);
        let shiftKey = activeKeys.includes("Shift") || activeKeys.includes("ShiftLeft") || activeKeys.includes("ShiftRight");

        if (activeKeys.includes("ArrowUp")) {
            if (shiftKey) {
                inv = translateMatrix4(inv, 0, -0.03, 0);
            } else {
                inv = translateMatrix4(inv, 0, 0, 0.1);
            }
        }
        if (activeKeys.includes("ArrowDown")) {
            if (shiftKey) {
                inv = translateMatrix4(inv, 0, 0.03, 0);
            } else {
                inv = translateMatrix4(inv, 0, 0, -0.1);
            }
        }
        if (activeKeys.includes("ArrowLeft")) inv = translateMatrix4(inv, -0.03, 0, 0);
        //
        if (activeKeys.includes("ArrowRight")) inv = translateMatrix4(inv, 0.03, 0, 0);
        if (activeKeys.includes("KeyA")) inv = rotateMatrix4(inv, -0.01, 0, 1, 0);
        if (activeKeys.includes("KeyD")) inv = rotateMatrix4(inv, 0.01, 0, 1, 0);
        if (activeKeys.includes("KeyQ")) inv = rotateMatrix4(inv, 0.01, 0, 0, 1);
        if (activeKeys.includes("KeyE")) inv = rotateMatrix4(inv, -0.01, 0, 0, 1);
        if (activeKeys.includes("KeyW")) inv = rotateMatrix4(inv, 0.005, 1, 0, 0);
        if (activeKeys.includes("KeyS")) inv = rotateMatrix4(inv, -0.005, 1, 0, 0);

        if (selectionMode && selectedLabel !== NO_SELECTION) {
            const splatMoveSpeed = 0.1; // Adjust speed as needed

            if (activeKeys.includes("KeyH")) {
                updateSplatDisplacement(selectedLabel, -splatMoveSpeed, 0, 0);
            }
            if (activeKeys.includes("KeyK")) {
                updateSplatDisplacement(selectedLabel, splatMoveSpeed, 0, 0);
            }

            // Up/Down
            if (activeKeys.includes("KeyJ")) {
                updateSplatDisplacement(selectedLabel, 0, splatMoveSpeed, 0);
            }
            if (activeKeys.includes("KeyU")) {
                updateSplatDisplacement(selectedLabel, 0, -splatMoveSpeed, 0);
            }

        }

        const gamepads = navigator.getGamepads ? navigator.getGamepads() : [];
        let isJumping = activeKeys.includes("Space");
        for (let gamepad of gamepads) {
            if (!gamepad) continue;

            const axisThreshold = 0.1; // Threshold to detect when the axis is intentionally moved
            const moveSpeed = 0.06;
            const rotateSpeed = 0.02;

            // Assuming the left stick controls translation (axes 0 and 1)
            if (Math.abs(gamepad.axes[0]) > axisThreshold) {
                inv = translateMatrix4(inv, moveSpeed * gamepad.axes[0], 0, 0);
                carousel = false;
            }
            if (Math.abs(gamepad.axes[1]) > axisThreshold) {
                inv = translateMatrix4(inv, 0, 0, -moveSpeed * gamepad.axes[1]);
                carousel = false;
            }
            if (gamepad.buttons[12].pressed || gamepad.buttons[13].pressed) {
                inv = translateMatrix4(inv, 0, -moveSpeed * (gamepad.buttons[12].pressed - gamepad.buttons[13].pressed), 0,);
                carousel = false;
            }

            if (gamepad.buttons[14].pressed || gamepad.buttons[15].pressed) {
                inv = translateMatrix4(inv, -moveSpeed * (gamepad.buttons[14].pressed - gamepad.buttons[15].pressed), 0, 0,);
                carousel = false;
            }

            // Assuming the right stick controls rotation (axes 2 and 3)
            if (Math.abs(gamepad.axes[2]) > axisThreshold) {
                inv = rotateMatrix4(inv, rotateSpeed * gamepad.axes[2], 0, 1, 0);
                carousel = false;
            }
            if (Math.abs(gamepad.axes[3]) > axisThreshold) {
                inv = rotateMatrix4(inv, -rotateSpeed * gamepad.axes[3], 1, 0, 0);
                carousel = false;
            }

            let tiltAxis = gamepad.buttons[6].value - gamepad.buttons[7].value;
            if (Math.abs(tiltAxis) > axisThreshold) {
                inv = rotateMatrix4(inv, rotateSpeed * tiltAxis, 0, 0, 1);
                carousel = false;
            }
            if (gamepad.buttons[4].pressed && !leftGamepadTrigger) {
                camera = cameras[(cameras.indexOf(camera) + 1) % cameras.length];
                inv = invert4(getViewMatrix(camera));
                carousel = false;
            }
            if (gamepad.buttons[5].pressed && !rightGamepadTrigger) {
                camera = cameras[(cameras.indexOf(camera) + cameras.length - 1) % cameras.length];
                inv = invert4(getViewMatrix(camera));
                carousel = false;
            }
            leftGamepadTrigger = gamepad.buttons[4].pressed;
            rightGamepadTrigger = gamepad.buttons[5].pressed;
            if (gamepad.buttons[0].pressed) {
                isJumping = true;
                carousel = false;
            }
            if (gamepad.buttons[3].pressed) {
                carousel = true;
            }
        }

        /*         if (["KeyJ", "KeyK", "KeyL", "KeyI"].some((k) => activeKeys.includes(k))) {
                    let d = 4;
                    inv = translateMatrix4(inv, 0, 0, d);
                    inv = rotateMatrix4(inv, activeKeys.includes("KeyJ") ? -0.05 : activeKeys.includes("KeyL") ? 0.05 : 0, 0, 1, 0,);
                    inv = rotateMatrix4(inv, activeKeys.includes("KeyI") ? 0.05 : activeKeys.includes("KeyK") ? -0.05 : 0, 1, 0, 0,);
                    inv = translateMatrix4(inv, 0, 0, -d);
                }
         */
        viewMatrix = invert4(inv);

        if (carousel) {
            let inv = invert4(defaultViewMatrix);

            const t = Math.sin((Date.now() - start) / 5000);
            inv = translateMatrix4(inv, 2.5 * t, 0, 6 * (1 - Math.cos(t)));
            inv = rotateMatrix4(inv, -0.6 * t, 0, 1, 0);

            viewMatrix = invert4(inv);
        }

        if (isJumping) {
            jumpDelta = Math.min(1, jumpDelta + 0.05);
        } else {
            jumpDelta = Math.max(0, jumpDelta - 0.05);
        }

        let inv2 = invert4(viewMatrix);
        inv2 = translateMatrix4(inv2, 0, -jumpDelta, 0);
        inv2 = rotateMatrix4(inv2, -0.1 * jumpDelta, 1, 0, 0);
        actualViewMatrix = invert4(inv2);

        const viewProj = multiply4(projectionMatrix, actualViewMatrix);
        worker.postMessage({view: viewProj});

        const currentFps = 1000 / (now - lastFrame) || 0;
        avgFps = avgFps * 0.9 + currentFps * 0.1;

        // Inside frame function, where we handle rendering
        if (vertexCount > 0) {
            gl.bindFramebuffer(gl.FRAMEBUFFER, null);
            gl.uniform1i(u_selectionMode, selectionMode ? 1 : 0);
            gl.uniform1i(u_selectedLabel, selectedLabel);
            gl.uniformMatrix4fv(u_view, false, actualViewMatrix);


            colorManager.updateColorUniforms();


            // Handling displacement for currently selected label
            if (selectedLabel !== NO_SELECTION && selectionMode) {
                const displacement = displacements.get(selectedLabel) || [0, 0, 0];
                gl.uniform1i(u_enableDisplacement, 1);
                gl.uniform3fv(u_displacement, new Float32Array(displacement));
            } else {
                if (!selectionMode) {
                    gl.uniform1i(u_enableDisplacement, 0);
                    gl.uniform3fv(u_displacement, new Float32Array([0, 0, 0]));
                }
            }

            gl.clear(gl.COLOR_BUFFER_BIT);
            gl.drawArraysInstanced(gl.TRIANGLE_FAN, 0, 4, vertexCount);
            //  debugState()
            document.getElementById("spinner").style.display = "none";
            document.getElementById("message").style.display = "none"; // Added this for cleaner UI
            gl.bindFramebuffer(gl.FRAMEBUFFER, null);
            gl.uniform1i(u_selectionMode, selectionMode ? 1 : 0);
            gl.uniform1i(u_selectedLabel, selectedLabel);
            gl.uniformMatrix4fv(u_view, false, actualViewMatrix);
            gl.clear(gl.COLOR_BUFFER_BIT);
            //console.log("Drawing", vertexCount, "vertices");
            gl.drawArraysInstanced(gl.TRIANGLE_FAN, 0, 4, vertexCount);
            //  debugState()
        } else {
            gl.clear(gl.COLOR_BUFFER_BIT);
            document.getElementById("spinner").style.display = "";
            start = Date.now() + 2000;
        }
        if (splatData) {
            const progress = (100 * vertexCount) / (splatData.length / rowLength);
            if (progress < 100) {
                document.getElementById("progress").style.width = progress + "%";
            } else {
                document.getElementById("progress").style.display = "none";
            }
        }
        fps.innerText = Math.round(avgFps) + " fps";
        if (isNaN(currentCameraIndex)) {
            camid.innerText = "";
        }
        lastFrame = now;
        requestAnimationFrame(frame);
    };
    loadDisplacements();
    frame();

    const isPly = (splatData) => splatData[0] == 112 && splatData[1] == 108 && splatData[2] == 121 && splatData[3] == 10;

    const selectFile = (file) => {
        const fr = new FileReader();
        if (/\.json$/i.test(file.name)) {
            fr.onload = () => {
                cameras = JSON.parse(fr.result);
                viewMatrix = getViewMatrix(cameras[0]);
                projectionMatrix = calculateProjectionMatrix(camera.fx / downsample, camera.fy / downsample, canvas.width, canvas.height,);
                gl.uniformMatrix4fv(u_projection, false, projectionMatrix);

                console.log("Loaded Cameras");
            };
            fr.readAsText(file);
        } else {
            stopLoading = true;
            fr.onload = () => {
                splatData = new Uint8Array(fr.result);
                console.log("Loaded file data:", {
                    fileSize: fr.result.byteLength
                });

                if (isPly(splatData)) {
                    // ply file magic header means it should be handled differently
                    worker.postMessage({ply: splatData.buffer});
                } else {
                    worker.postMessage({
                        buffer: splatData.buffer, vertexCount: Math.floor(splatData.length / rowLength),
                    });
                }
                loadDisplacements();

            };
            fr.readAsArrayBuffer(file);
        }
    };

    window.addEventListener("hashchange", (e) => {
        try {
            viewMatrix = JSON.parse(decodeURIComponent(location.hash.slice(1)));
            carousel = false;
        } catch (err) {
        }
    });

    const preventDefault = (e) => {
        e.preventDefault();
        e.stopPropagation();
    };
    document.addEventListener("dragenter", preventDefault);
    document.addEventListener("dragover", preventDefault);
    document.addEventListener("dragleave", preventDefault);
    document.addEventListener("drop", (e) => {
        e.preventDefault();
        e.stopPropagation();
        selectFile(e.dataTransfer.files[0]);
    });
}


let labels;
let cameras = [
    {
        id: 0,
        img_name: "00001",
        width: 1959,
        height: 1090,
        position: [
            -3.0089893469241797, -0.11086489695181866, -3.7527640949141428,
        ],
        rotation: [
            [0.876134201218856, 0.06925962026449776, 0.47706599800804744],
            [-0.04747421839895102, 0.9972110940209488, -0.057586739349882114],
            [-0.4797239414934443, 0.027805376500959853, 0.8769787916452908],
        ],
        fy: 1164.6601287484507,
        fx: 1159.5880733038064,
    },
    {
        id: 1,
        img_name: "00009",
        width: 1959,
        height: 1090,
        position: [
            -2.5199776022057296, -0.09704735754873686, -3.6247725540304545,
        ],
        rotation: [
            [0.9982731285632193, -0.011928707708098955, -0.05751927260507243],
            [0.0065061360949636325, 0.9955928229282383, -0.09355533724430458],
            [0.058381769258182864, 0.09301955098900708, 0.9939511719154457],
        ],
        fy: 1164.6601287484507,
        fx: 1159.5880733038064,
    },
    {
        id: 2,
        img_name: "00017",
        width: 1959,
        height: 1090,
        position: [
            -0.7737533667465242, -0.3364271945329695, -2.9358969417573753,
        ],
        rotation: [
            [0.9998813418672372, 0.013742375651625236, -0.0069605529394208224],
            [-0.014268370388586709, 0.996512943252834, -0.08220929105659476],
            [0.00580653013657589, 0.08229885200307129, 0.9965907801935302],
        ],
        fy: 1164.6601287484507,
        fx: 1159.5880733038064,
    },
    {
        id: 3,
        img_name: "00025",
        width: 1959,
        height: 1090,
        position: [
            1.2198221749590001, -0.2196687861401182, -2.3183162007028453,
        ],
        rotation: [
            [0.9208648867765482, 0.0012010625395201253, 0.389880004297208],
            [-0.06298204172269357, 0.987319521752825, 0.14571693239364383],
            [-0.3847611242348369, -0.1587410451475895, 0.9092635249821667],
        ],
        fy: 1164.6601287484507,
        fx: 1159.5880733038064,
    },
    {
        id: 4,
        img_name: "00033",
        width: 1959,
        height: 1090,
        position: [
            1.742387858893817, -0.13848225198886954, -2.0566370113193146,
        ],
        rotation: [
            [0.24669889292141334, -0.08370189346592856, -0.9654706879349405],
            [0.11343747891376445, 0.9919082664242816, -0.05700815184573074],
            [0.9624300466054861, -0.09545671285663988, 0.2541976029815521],
        ],
        fy: 1164.6601287484507,
        fx: 1159.5880733038064,
    },
    {
        id: 5,
        img_name: "00041",
        width: 1959,
        height: 1090,
        position: [
            3.6567309419223935, -0.16470990600750707, -1.3458085590422042,
        ],
        rotation: [
            [0.2341293058324528, -0.02968330457755884, -0.9717522161434825],
            [0.10270823606832301, 0.99469554638321, -0.005638106875665722],
            [0.9667649592295676, -0.09848690996657204, 0.2359360976431732],
        ],
        fy: 1164.6601287484507,
        fx: 1159.5880733038064,
    },
    {
        id: 6,
        img_name: "00049",
        width: 1959,
        height: 1090,
        position: [
            3.9013554243203497, -0.2597500978038105, -0.8106154188297828,
        ],
        rotation: [
            [0.6717235545638952, -0.015718162115524837, -0.7406351366386528],
            [0.055627354673906296, 0.9980224478387622, 0.029270992841185218],
            [0.7387104058127439, -0.060861588786650656, 0.6712695459756353],
        ],
        fy: 1164.6601287484507,
        fx: 1159.5880733038064,
    },
    {
        id: 7,
        img_name: "00057",
        width: 1959,
        height: 1090,
        position: [4.742994605467533, -0.05591660945412069, 0.9500365976084458],
        rotation: [
            [-0.17042655709210375, 0.01207080756938, -0.9852964448542146],
            [0.1165090336695526, 0.9931575292530063, -0.00798543433078162],
            [0.9784581921120181, -0.1161568667478904, -0.1706667764862097],
        ],
        fy: 1164.6601287484507,
        fx: 1159.5880733038064,
    },
    {
        id: 8,
        img_name: "00065",
        width: 1959,
        height: 1090,
        position: [4.34676307626522, 0.08168160516967145, 1.0876221470355405],
        rotation: [
            [-0.003575447631888379, -0.044792503246552894, -0.9989899137764799],
            [0.10770152645126597, 0.9931680875192705, -0.04491693593046672],
            [0.9941768441149182, -0.10775333677534978, 0.0012732004866391048],
        ],
        fy: 1164.6601287484507,
        fx: 1159.5880733038064,
    },
    {
        id: 9,
        img_name: "00073",
        width: 1959,
        height: 1090,
        position: [3.264984351114202, 0.078974937336732, 1.0117200284114904],
        rotation: [
            [-0.026919994628162257, -0.1565891128261527, -0.9872968974090509],
            [0.08444552208239385, 0.983768234577625, -0.1583319754069128],
            [0.9960643893290491, -0.0876350978794554, -0.013259786205163005],
        ],
        fy: 1164.6601287484507,
        fx: 1159.5880733038064,
    },
];

let camera = cameras[0]
/*let cameras, camera, labels
Promise.all([fetch('cameras.json'), fetch('ade20k-id2label.json')]).then(responses =>
    Promise.all(responses.map(response => response.json())).then(data => {
        cameras = data[0];
        camera = cameras[0];
        labels = data[1];
        main()
    })).catch((err) => {
    document.getElementById("spinner").style.display = "none";
    document.getElementById("message").innerText = err.toString();
})*/

fetch('ade20k-id2label.json').then(response => response.json()).then(data => {
    labels = data;
    main();
}).catch((err) => {
    document.getElementById("spinner").style.display = "none";
    document.getElementById("message").innerText = err.toString();
});
