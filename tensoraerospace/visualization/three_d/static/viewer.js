/**
 * F-16 flight visualizer (Phase B — placeholder cube).
 *
 * Reads the inline `window.FLIGHT_LOG` populated by the HTML template,
 * sets up a three.js scene with OrbitControls, and animates a placeholder
 * cube along the trajectory. Phase C replaces the cube with a parametric
 * F-16 built from the flight log's geometry.sections array.
 */
(function () {
    "use strict";

    const log = window.FLIGHT_LOG;
    if (!log) {
        document.body.innerHTML = "<div style='padding:24px;color:#f88'>No FLIGHT_LOG injected.</div>";
        return;
    }
    const MODEL_NAME = ((log.metadata && log.metadata.model) || "").toLowerCase();
    const AIRCRAFT_TYPE = ((log.metadata && log.metadata.aircraft_type)
        || (log.geometry && log.geometry.aircraft_type)
        || MODEL_NAME);
    const IS_B747 = AIRCRAFT_TYPE.toLowerCase().includes("b747")
        || AIRCRAFT_TYPE.toLowerCase().includes("b-747");
    if (IS_B747) document.body.classList.add("aircraft-b747");

    const traj = log.trajectory;
    const T = traj.time.length;

    // ---- Scene setup ----
    const sceneEl = document.getElementById("scene");
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.domElement.style.touchAction = "none";
    // Cap DPR at 2 — going to 3+ on retina screens triples pixel count
    // for negligible visual gain on this geometry density (Three.js
    // best practice: render-pixel-ratio).
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(sceneEl.clientWidth, sceneEl.clientHeight);
    // PBR-correct tone mapping for HDR-friendly output
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.0;
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    sceneEl.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x4a6a9a);

    // ---- Environment map (PMREMGenerator sky) ----
    // Procedurally-generated equirect sky → PMREM cube → scene.environment.
    // Gives PBR metalness/roughness something to reflect: the canopy looks
    // glassy, fuselage gains proper sky highlights, missiles read as metal.
    // No external assets required. (Three.js rule: lighting-environment)
    function _buildSkyEnvMap(rend) {
        const pmrem = new THREE.PMREMGenerator(rend);
        pmrem.compileEquirectangularShader();

        const c = document.createElement("canvas");
        c.width = 512; c.height = 256;
        const ctx = c.getContext("2d");
        const grad = ctx.createLinearGradient(0, 0, 0, 256);
        grad.addColorStop(0.00, "#5a7fb5");   // upper sky
        grad.addColorStop(0.45, "#7a9bc8");   // horizon haze
        grad.addColorStop(0.50, "#8a8270");   // horizon line
        grad.addColorStop(1.00, "#3a3530");   // ground / desert
        ctx.fillStyle = grad;
        ctx.fillRect(0, 0, 512, 256);
        // Faint "sun" highlight to bias reflections
        const sunGrad = ctx.createRadialGradient(380, 70, 4, 380, 70, 60);
        sunGrad.addColorStop(0, "rgba(255,240,200,0.95)");
        sunGrad.addColorStop(1, "rgba(255,240,200,0)");
        ctx.fillStyle = sunGrad;
        ctx.fillRect(0, 0, 512, 256);

        const tex = new THREE.CanvasTexture(c);
        tex.mapping = THREE.EquirectangularReflectionMapping;
        tex.colorSpace = THREE.SRGBColorSpace;
        const envRT = pmrem.fromEquirectangular(tex);
        tex.dispose();
        pmrem.dispose();
        return envRT.texture;
    }
    scene.environment = _buildSkyEnvMap(renderer);

    // Three lights total (lighting-limit-lights rule: ≤3 active lights).
    // Lower ambient now that env map provides ambient reflections.
    scene.add(new THREE.AmbientLight(0xffffff, 0.25));
    const sun = new THREE.DirectionalLight(0xfff0d0, 0.9);
    sun.position.set(40, 60, 30);
    scene.add(sun);
    // Hemisphere fill: sky colour from above, ground from below
    const fillLight = new THREE.HemisphereLight(0x6a7fb5, 0x3a3530, 0.4);
    scene.add(fillLight);

    // Ground grid. B-747 scenarios can cover tens of kilometres, so the
    // grid/far plane are scaled up when the log declares a wide-body model.
    const GRID_SIZE = IS_B747 ? 80000 : 2000;
    const GRID_DIVS = IS_B747 ? 80 : 40;
    const grid = new THREE.GridHelper(GRID_SIZE, GRID_DIVS, 0x303048, 0x202030);
    grid.position.y = 0;
    scene.add(grid);

    // ---- Camera + controls ----
    // Near=1m: we never look from < 1 m away. Far=8000m for long trails.
    // Tighter near plane improves depth-buffer precision (camera-near-far rule).
    const camera = new THREE.PerspectiveCamera(
        55, sceneEl.clientWidth / sceneEl.clientHeight, 1,
        IS_B747 ? 150000 : 8000,
    );
    if (IS_B747) camera.position.set(90, 55, 95);
    else camera.position.set(20, 14, 20);
    camera.lookAt(0, 0, 0);

    const controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.rotateSpeed = IS_B747 ? 0.9 : 1.0;
    controls.zoomSpeed = IS_B747 ? 1.4 : 1.0;
    controls.panSpeed = IS_B747 ? 1.1 : 1.0;
    controls.minDistance = IS_B747 ? 20 : 5;
    controls.maxDistance = IS_B747 ? 12000 : 1500;

    // Camera modes: each frame, after the aircraft is repositioned,
    // updateCamera() places the camera so the view follows the aircraft.
    //  - "3d":    OrbitControls drives the camera; we just keep target at
    //             aircraft so the user's orbit angle is preserved while
    //             the centre of orbit moves with the plane.
    //  - "top":   straight overhead, fixed offset (+y in world).
    //  - "left":  body's left flank (-z in world).
    //  - "right": body's right flank (+z in world).
    let cameraMode = "3d";

    // Camera offsets in the aircraft's LOCAL three.js frame. After we
    // build the aircraft via bodyToThree(bx, by, bz) → [bx, -bz, by], the
    // aircraft Object3D's local axes align as:
    //   local +x = body +x = forward
    //   local +y = body -z = up
    //   local +z = body +y = right
    // So the offsets below are intuitive: +x = ahead, +y = above, +z = right.
    // applyEuler(aircraft.rotation) maps them to world coords each frame
    // so the camera rolls/pitches/yaws with the aircraft.
    const FOLLOW_OFFSETS = IS_B747 ? {
        top:   { offset: new THREE.Vector3(0,  220,   0),
                 up:     new THREE.Vector3(1,    0,   0) },
        left:  { offset: new THREE.Vector3(0,   30, -140),
                 up:     new THREE.Vector3(0,    1,   0) },
        right: { offset: new THREE.Vector3(0,   30,  140),
                 up:     new THREE.Vector3(0,    1,   0) },
        "3d":  { offset: new THREE.Vector3(90, 55, 95),
                 up:     new THREE.Vector3(0,   1,   0) },
    } : {
        // Above the cockpit, looking down. "Up" on screen is aircraft fwd.
        top:   { offset: new THREE.Vector3(0,  60,   0),
                 up:     new THREE.Vector3(1,   0,   0) },
        // To the pilot's left. "Up" on screen is aircraft up.
        left:  { offset: new THREE.Vector3(0,   5, -35),
                 up:     new THREE.Vector3(0,   1,   0) },
        // To the pilot's right.
        right: { offset: new THREE.Vector3(0,   5,  35),
                 up:     new THREE.Vector3(0,   1,   0) },
        // 3D offset is the initial drop-in for OrbitControls; afterwards
        // the user's mouse drives the orbit angle around the aircraft.
        "3d":  { offset: new THREE.Vector3(20, 14,  20),
                 up:     new THREE.Vector3(0,   1,   0) },
    };

    function updateCamera() {
        const p = aircraft.position;
        if (cameraMode === "3d") {
            // Move both controls.target AND camera.position by the same
            // delta. OrbitControls preserves camera's offset from target
            // through its internal spherical state, so without moving
            // camera explicitly the orbit point would drift away from
            // the aircraft as the plane flies. With this delta-shift the
            // orbit ring travels with the aircraft and the user's
            // current orbit angle / radius is preserved.
            const delta = p.clone().sub(controls.target);
            controls.target.copy(p);
            camera.position.add(delta);
            camera.up.set(0, 1, 0);
            return;
        }

        // Strap the camera to the aircraft body. The offset is given in
        // the aircraft's local three.js frame; rotate by aircraft.rotation
        // to get world-frame placement that follows roll / pitch / yaw.
        const cfg = FOLLOW_OFFSETS[cameraMode];
        const worldOffset = cfg.offset.clone().applyEuler(aircraft.rotation);
        camera.position.copy(p).add(worldOffset);
        // Camera "up" also rotates with the aircraft so e.g. on barrel
        // roll the side view tracks the roll naturally.
        camera.up.copy(cfg.up).applyEuler(aircraft.rotation);
        camera.lookAt(p);
    }

    function preset3DCamera() {
        cameraMode = "3d";
        controls.enabled = true;
        const p = aircraft.position;
        const off = FOLLOW_OFFSETS["3d"].offset;
        camera.position.set(p.x + off.x, p.y + off.y, p.z + off.z);
        controls.target.copy(p);
        camera.up.set(0, 1, 0);
        controls.update();
        syncCameraButtons();
    }

    function presetTopDown() {
        cameraMode = "top";
        controls.enabled = false;
        updateCamera();
        syncCameraButtons();
    }

    function presetLeftSide() {
        cameraMode = "left";
        controls.enabled = false;
        updateCamera();
        syncCameraButtons();
    }

    function presetRightSide() {
        cameraMode = "right";
        controls.enabled = false;
        updateCamera();
        syncCameraButtons();
    }

    function syncCameraButtons() {
        const ids = {
            "3d": "btn-cam-3d",
            top: "btn-cam-top",
            left: "btn-cam-left",
            right: "btn-cam-right",
        };
        for (const [mode, id] of Object.entries(ids)) {
            const btn = document.getElementById(id);
            if (btn) btn.classList.toggle("active", cameraMode === mode);
        }
    }

    // If the user starts dragging the 3D canvas while a fixed camera
    // preset is active, switch back to free orbit immediately. This avoids
    // the "mouse drag does nothing" failure mode.
    renderer.domElement.addEventListener("pointerdown", () => {
        if (cameraMode !== "3d") preset3DCamera();
    }, { capture: true });

    // ---- Procedural F-16 mesh ----
    // Hand-tuned to look recognisably like an F-16 while preserving
    // section-level addressability for damage visualization. Each YAML
    // section in log.geometry.sections is rendered as a named Object3D
    // child of the aircraft Group.
    function bodyToThree(bx, by, bz) {
        return [bx, -bz, by];  // body (x-fwd, y-right, z-down) → three (x-fwd, y-up, z-right)
    }

    const F16_COLORS = {
        fuselage:  0xa0a8b0,   // light grey
        canopy:    0x2c3e50,   // dark blue glass
        intake:    0x4a4a4a,   // darker grey
        wing:      0x8a92a0,   // wing grey
        wing_edge: 0x484848,   // leading-edge dark line
        stab:      0x808890,
        vtail:     0x6a7280,
        rudder:    0x5a6068,
        aileron:   0x707880,
        nozzle:    0x303030,
    };

    // Material cache (Three.js best practice: material-reuse). Many
    // aircraft sub-parts share colors. Sharing the same Material instance
    // lets the renderer batch state changes and roughly halves the unique
    // material count for this scene.
    //
    // CAVEAT: meshes whose material gets MUTATED at runtime (damage tint
    // on wing sections / control surfaces, exhaust fade) need their own
    // material so the mutation doesn't leak. Pass {unique: true} for them.
    const _stdMatCache = new Map();
    function _stdMat(color, opts) {
        opts = opts || {};
        const unique = opts.unique === true;
        const metalness = opts.metalness ?? 0.55;
        const roughness = opts.roughness ?? 0.4;
        const opacity = opts.opacity ?? 1.0;
        const transparent = !!opts.transparent;
        const emissive = opts.emissive ?? 0;
        const emissiveIntensity = opts.emissiveIntensity ?? 1.0;
        const cacheKey = [
            color, metalness, roughness, opacity, transparent, emissive,
            emissiveIntensity,
        ].join("|");
        if (!unique) {
            const hit = _stdMatCache.get(cacheKey);
            if (hit) return hit;
        }
        const cleanOpts = Object.assign({}, opts);
        delete cleanOpts.unique;
        const mat = new THREE.MeshStandardMaterial({
            color, metalness, roughness, opacity, transparent,
            side: THREE.DoubleSide,
            ...(emissive ? { emissive, emissiveIntensity } : {}),
            ...cleanOpts,
        });
        if (!unique) _stdMatCache.set(cacheKey, mat);
        return mat;
    }

    // ---- Manual geometry merge helper ----
    // BufferGeometryUtils.mergeGeometries is not shipped in the UMD build,
    // so we do it manually. Takes an array of {geom, matrix?} pairs and
    // concatenates position + normal attributes into one BufferGeometry.
    // (Three.js rule: geometry-merge-static)
    function _mergeGeoms(pairs) {
        let totalVerts = 0;
        for (const { geom } of pairs) {
            totalVerts += geom.attributes.position.count;
        }
        const posArr = new Float32Array(totalVerts * 3);
        const nrmArr = new Float32Array(totalVerts * 3);
        let off = 0;
        const _tmp = new THREE.Vector3();
        const _nrm = new THREE.Vector3();
        for (const { geom, matrix } of pairs) {
            const pos = geom.attributes.position;
            const nrm = geom.attributes.normal;
            const normalMatrix = matrix
                ? new THREE.Matrix3().getNormalMatrix(matrix)
                : null;
            for (let i = 0; i < pos.count; i++) {
                _tmp.fromBufferAttribute(pos, i);
                if (matrix) _tmp.applyMatrix4(matrix);
                posArr[(off + i) * 3]     = _tmp.x;
                posArr[(off + i) * 3 + 1] = _tmp.y;
                posArr[(off + i) * 3 + 2] = _tmp.z;
                if (nrm) {
                    _nrm.fromBufferAttribute(nrm, i);
                    if (normalMatrix) _nrm.applyNormalMatrix(normalMatrix).normalize();
                    nrmArr[(off + i) * 3]     = _nrm.x;
                    nrmArr[(off + i) * 3 + 1] = _nrm.y;
                    nrmArr[(off + i) * 3 + 2] = _nrm.z;
                }
            }
            off += pos.count;
        }
        const merged = new THREE.BufferGeometry();
        merged.setAttribute("position", new THREE.BufferAttribute(posArr, 3));
        merged.setAttribute("normal",   new THREE.BufferAttribute(nrmArr, 3));
        return merged;
    }

    function _quadGeom(c0, c1, c2, c3) {
        // 4 corners → 2 triangles. Corners in three.js coords directly.
        const positions = new Float32Array([
            ...c0, ...c1, ...c2,
            ...c0, ...c2, ...c3,
        ]);
        const geom = new THREE.BufferGeometry();
        geom.setAttribute("position", new THREE.BufferAttribute(positions, 3));
        geom.computeVertexNormals();
        return geom;
    }

    // F-16 wing planform (right side, body frame). Half-span ~ 4.7 m,
    // root chord ~ 5.0 m, leading edge sweep ~ 40°, trailing edge slight
    // forward sweep. Sections share boundary edges so the assembled wing
    // is continuous.
    const RIGHT_WING_POLYGONS = {
        right_root: [
            // [body.x, body.y]
            [+1.5,  0.7],   // LE_inboard (at fuselage)
            [+0.4,  2.0],   // LE_outboard
            [-3.4,  2.0],   // TE_outboard
            [-3.5,  0.7],   // TE_inboard
        ],
        right_mid: [
            [+0.4,  2.0],
            [-1.0,  3.5],
            [-3.5,  3.5],
            [-3.4,  2.0],
        ],
        right_tip: [
            [-1.0,  3.5],
            [-2.0,  4.7],
            [-3.5,  4.7],
            [-3.5,  3.5],
        ],
    };

    // Stabilator hinge: at the inboard leading-edge corner. Polygon
    // vertices are relative to this hinge so we can wrap the mesh in a
    // group positioned at the hinge and rotate the group to apply the
    // commanded stabilator deflection.
    const STAB_RIGHT_HINGE_BODY = [-3.5, 0.4, 0];   // body (x, y, z)
    const STAB_RIGHT_POLY_REL = [
        // [bx_rel, by_rel] relative to hinge
        [ 0.0,  0.0],            // LE_in (hinge)
        [-1.2,  1.4],            // LE_out
        [-2.7,  1.4],            // TE_out
        [-2.0,  0.0],            // TE_in
    ];

    // Aileron hinge: at the inboard leading-edge corner of the right
    // aileron flap.
    const AILERON_RIGHT_HINGE_BODY = [-3.4, 3.2, 0];
    const AILERON_RIGHT_POLY_REL = [
        [ 0.0,  0.0],
        [-0.3,  1.0],
        [-0.5,  1.0],
        [-0.1,  0.0],
    ];

    function _flatSectionMesh(corners2d_body, color, name) {
        // Build a thick wing section with top and bottom skins.
        // Each skin is a quad offset in body z (z < 0 = up in three.js y).
        // Top skin at bz = -0.06 (body up), bottom at bz = +0.06.
        const SKIN_HALF = 0.055;
        const top = corners2d_body.map(([bx, by]) => bodyToThree(bx, by, -SKIN_HALF));
        const bot = corners2d_body.map(([bx, by]) => bodyToThree(bx, by, +SKIN_HALF));

        // Triangulate as: top quad (T0 T1 T2, T0 T2 T3)
        //                 bottom quad reversed (B0 B2 B1, B0 B3 B2)
        //                 LE edge: T0-B0-B1, T0-B1-T1
        //                 TE edge: T2-T3-B3, T2-B3-B2
        const positions = new Float32Array([
            // top face
            ...top[0], ...top[1], ...top[2],
            ...top[0], ...top[2], ...top[3],
            // bottom face (winding reversed)
            ...bot[0], ...bot[2], ...bot[1],
            ...bot[0], ...bot[3], ...bot[2],
            // leading edge cap (vertices 0-1 shared by both sides)
            ...top[0], ...bot[0], ...bot[1],
            ...top[0], ...bot[1], ...top[1],
            // trailing edge cap
            ...top[2], ...top[3], ...bot[3],
            ...top[2], ...bot[3], ...bot[2],
            // inboard edge cap
            ...top[0], ...top[3], ...bot[3],
            ...top[0], ...bot[3], ...bot[0],
            // outboard edge cap
            ...top[1], ...bot[1], ...bot[2],
            ...top[1], ...bot[2], ...top[2],
        ]);
        const geom = new THREE.BufferGeometry();
        geom.setAttribute("position", new THREE.BufferAttribute(positions, 3));
        geom.computeVertexNormals();
        const mesh = new THREE.Mesh(geom, _stdMat(color));
        mesh.name = name;
        return mesh;
    }

    function _mirrorY(poly) {
        return poly.map(([x, y]) => [x, -y]);
    }

    // ---- Helper: build an oval cross-section ring ----
    // rx = half-width (body y direction), ry = half-height (body z direction)
    // Returns an array of N [by, bz] pairs forming the ring perimeter.
    function _ovalRing(rx, ry, N) {
        const pts = [];
        for (let i = 0; i < N; i++) {
            const a = (i / N) * 2 * Math.PI;
            pts.push([rx * Math.cos(a), ry * Math.sin(a)]);
        }
        return pts;
    }

    // Build a fuselage segment between two axial stations.
    // stationA/B: { bx, ring: [[by,bz]...] } (same N vertices)
    // Returns a BufferGeometry of triangles.
    function _fuseSegment(stA, stB) {
        const N = stA.ring.length;
        const verts = [];
        for (let i = 0; i < N; i++) {
            const j = (i + 1) % N;
            const [ayL, azL] = stA.ring[i];
            const [ayR, azR] = stA.ring[j];
            const [byL, bzL] = stB.ring[i];
            const [byR, bzR] = stB.ring[j];
            const A0 = bodyToThree(stA.bx, ayL, azL);
            const A1 = bodyToThree(stA.bx, ayR, azR);
            const B0 = bodyToThree(stB.bx, byL, bzL);
            const B1 = bodyToThree(stB.bx, byR, bzR);
            // Two triangles per quad panel
            verts.push(...A0, ...A1, ...B0);
            verts.push(...A1, ...B1, ...B0);
        }
        const geom = new THREE.BufferGeometry();
        geom.setAttribute("position",
            new THREE.BufferAttribute(new Float32Array(verts), 3));
        geom.computeVertexNormals();
        return geom;
    }

    // Cap a station with a triangle fan to a centre point.
    function _fuseCap(st, capBx, sign) {
        // sign = +1 for nose cap (cap is forward), -1 for tail cap
        const N = st.ring.length;
        const verts = [];
        const centre = bodyToThree(capBx, 0, 0);
        for (let i = 0; i < N; i++) {
            const j = (i + 1) % N;
            const [ayL, azL] = st.ring[i];
            const [ayR, azR] = st.ring[j];
            const A0 = bodyToThree(st.bx, ayL, azL);
            const A1 = bodyToThree(st.bx, ayR, azR);
            if (sign > 0) {
                verts.push(...centre, ...A1, ...A0);
            } else {
                verts.push(...centre, ...A0, ...A1);
            }
        }
        const geom = new THREE.BufferGeometry();
        geom.setAttribute("position",
            new THREE.BufferAttribute(new Float32Array(verts), 3));
        geom.computeVertexNormals();
        return geom;
    }

    function _fuselageGroup() {
        const group = new THREE.Group();
        const N = 20;   // ring vertex count — enough for smooth oval

        // F-16 cross-section stations along body x.
        // Each station: bx (fwd +), rx (half-width in y), ry_top/ry_bot for
        // asymmetric upper/lower half-height (chine flattening).
        // The F-16 forebody is flattened below (flat-bottomed chin for intake)
        // and slightly humped above. Aft body tapers to circular.
        //
        // rx  = half-width (symmetrical left-right)
        // ryT = upper half-height (body z negative = up)
        // ryB = lower half-height (body z positive = down)
        //
        //  bx      rx    ryT   ryB
        const STATIONS_DEF = [
            [ +8.0,  0.00, 0.00, 0.00],  // nose tip (cap)
            [ +7.0,  0.08, 0.08, 0.08],  // radome tip
            [ +6.0,  0.18, 0.18, 0.18],  // narrow radome
            [ +5.0,  0.32, 0.30, 0.28],  // start of chine
            [ +3.8,  0.52, 0.44, 0.35],  // forebody chine — flatter below
            [ +2.8,  0.68, 0.52, 0.40],  // below cockpit starts
            [ +1.5,  0.78, 0.60, 0.48],  // cockpit mid — widest cross section
            [ +0.0,  0.82, 0.58, 0.52],  // widest — inlet bay
            [ -1.5,  0.80, 0.55, 0.55],  // aft of inlet
            [ -3.0,  0.75, 0.54, 0.54],  // mid fuselage
            [ -4.5,  0.68, 0.52, 0.52],  // toward wing trailing edge
            [ -5.8,  0.60, 0.50, 0.50],  // aft fuselage
            [ -7.0,  0.52, 0.46, 0.46],  // engine casing
            [ -8.0,  0.45, 0.42, 0.42],  // nozzle section
            [ -8.3,  0.42, 0.40, 0.40],  // nozzle exit
        ];

        // Build oval rings per station. We use a modified oval: top arc uses
        // ryT, bottom arc uses ryB, giving the chin flattening.
        function _buildRing(rx, ryT, ryB, N) {
            const pts = [];
            for (let i = 0; i < N; i++) {
                const a = (i / N) * 2 * Math.PI;
                const sinA = Math.sin(a);
                const cosA = Math.cos(a);
                const by = rx * cosA;
                // body z: negative = up. sinA < 0 means upper half.
                const ry = sinA < 0 ? ryT : ryB;
                const bz = ry * sinA;
                pts.push([by, bz]);
            }
            return pts;
        }

        const stations = STATIONS_DEF.map(([bx, rx, ryT, ryB]) => ({
            bx, ring: _buildRing(rx, ryT, ryB, N),
        }));

        // Stitch fuselage segments (skip the cap station at index 0)
        const fuseGeomsVerts = [];
        for (let i = 1; i < stations.length; i++) {
            const segGeom = _fuseSegment(stations[i - 1], stations[i]);
            fuseGeomsVerts.push(...segGeom.attributes.position.array);
        }
        // Nose cap: fan from station[0] to tip
        const noseCapGeom = _fuseCap(stations[1], stations[0].bx, +1);
        fuseGeomsVerts.push(...noseCapGeom.attributes.position.array);

        const allFuseGeom = new THREE.BufferGeometry();
        allFuseGeom.setAttribute("position",
            new THREE.BufferAttribute(new Float32Array(fuseGeomsVerts), 3));
        allFuseGeom.computeVertexNormals();
        const fuselage = new THREE.Mesh(allFuseGeom, _stdMat(F16_COLORS.fuselage));
        fuselage.name = "fuselage_main";
        group.add(fuselage);

        // ---- COCKPIT HUMP ----
        // A streamlined blister above station bx ~+1.5 to +3.5 to suggest
        // the cockpit turtledeck and sill.
        const humpGeom = new THREE.SphereGeometry(0.35, 16, 10, 0, Math.PI * 2, 0, Math.PI * 0.55);
        humpGeom.scale(2.2, 1.0, 0.85);
        const hump = new THREE.Mesh(humpGeom, _stdMat(F16_COLORS.fuselage));
        hump.position.set(2.0, 0.62, 0);
        group.add(hump);

        // ---- CANOPY — single bubble teardrop (F-16 frameless canopy) ----
        // One smooth tapered dome: starts low/angled at x≈+3.5 (windscreen),
        // peaks at x≈+2.0, tapers to x≈-1.5 along the fuselage spine.
        // Built as a SphereGeometry upper hemisphere with per-vertex x-stretch
        // so the shape is a proper teardrop rather than a symmetric dome.
        const canopyGeom = new THREE.SphereGeometry(
            0.95, 28, 18,           // radius, widthSeg, heightSeg
            0, Math.PI * 2,         // phiStart, phiLength (full revolution)
            0, Math.PI * 0.52,      // thetaStart, thetaLength (upper ~half)
        );
        {
            const cpos = canopyGeom.attributes.position;
            for (let i = 0; i < cpos.count; ++i) {
                const sx = cpos.getX(i);   // sphere-local x in [-0.95, +0.95]
                const sy = cpos.getY(i);   // sphere y (up)
                const sz = cpos.getZ(i);   // sphere z (right)
                // Map sphere x-range [-0.95, +0.95] → body x [+3.5, -1.5]
                const t = (sx + 0.95) / 1.9;          // 0=front, 1=back
                const newBx = 3.5 - 5.0 * t;
                // Taper width and height toward both ends (sin gives 0 at ends)
                const taper = Math.sin(Math.PI * t);
                const widthScale  = 0.35 + 0.65 * taper;
                const heightScale = 0.42 + 0.58 * taper;
                // In three.js space: X=body-fwd, Y=body-up, Z=body-right.
                // Sphere's Y is "height" (up), Z is "sideways".
                cpos.setXYZ(
                    i,
                    newBx,
                    sy * heightScale + 0.55,    // offset up so base sits at y≈+0.55
                    sz * widthScale,
                );
            }
            cpos.needsUpdate = true;
        }
        canopyGeom.computeVertexNormals();

        const bubble = new THREE.Mesh(
            canopyGeom,
            _stdMat(0x182838, { metalness: 0.88, roughness: 0.04,
                                 transparent: true, opacity: 0.70 }),
        );
        bubble.name = "fuselage_canopy";
        group.add(bubble);

        // Canopy sill band — thin dark torus at the canopy's base where it
        // meets the fuselage. Slightly elongated to follow canopy footprint.
        const sillGeom = new THREE.TorusGeometry(0.80, 0.04, 8, 32);
        const sill = new THREE.Mesh(
            sillGeom,
            _stdMat(0x111820, { metalness: 0.5, roughness: 0.4 }),
        );
        sill.rotation.x = Math.PI / 2;
        sill.scale.set(2.2, 1.0, 0.75);    // elongate to follow canopy footprint
        sill.position.set(1.0, 0.56, 0);   // centred under the canopy
        group.add(sill);

        // Pilot helmet — small dark sphere visible through the canopy.
        // Positioned at body (x=+2.0, y=0, z=-0.85) → three.js (2.0, +0.85, 0).
        const helmetGeom = new THREE.SphereGeometry(0.18, 10, 8);
        const helmet = new THREE.Mesh(
            helmetGeom,
            _stdMat(0x1a1a1f, { metalness: 0.3, roughness: 0.7 }),
        );
        helmet.position.set(2.0, 0.88, 0);
        group.add(helmet);

        // ---- INTAKE ----
        // Recessed oval-mouth chin intake with boundary-layer splitter plate.
        // The mouth is an oval at body x = +3.8, hanging ~0.95 m below
        // the centreline (body bz ≈ +0.95), ±0.80 wide × ±0.45 tall.
        function _intakePanelGeom(corners4_body) {
            const c = corners4_body.map(([bx, by, bz]) => bodyToThree(bx, by, bz));
            return _quadGeom(c[0], c[1], c[2], c[3]);
        }

        const intakeGroup = new THREE.Group();
        intakeGroup.name = "fuselage_intake";

        // Duct corners in body space
        const iFront = +3.8, iBack = +1.5;
        const iHalfW = 0.80, iTop = +0.60, iBot = +1.50;
        const iMat = _stdMat(F16_COLORS.intake, { metalness: 0.48, roughness: 0.45 });

        // Collect all static intake geometry for merging into one mesh
        const intakeDecorGeoms = [];

        // Oval mouth: parametric ellipse ring as front face
        // Build a closed oval ring at bx = iFront as a triangle fan
        const OVAL_SEGS = 24;
        const ovalCentBz = (iTop + iBot) * 0.5;   // ≈ +1.05
        const ovalHalfH  = (iBot - iTop) * 0.5;   // ≈ 0.45
        const ovalFaceVerts = [];
        const ovalCx = iFront, ovalCy = 0, ovalCz = ovalCentBz;
        const ovalRy = iHalfW, ovalRz = ovalHalfH;
        for (let i = 0; i < OVAL_SEGS; i++) {
            const a0 = (i / OVAL_SEGS) * Math.PI * 2;
            const a1 = ((i + 1) / OVAL_SEGS) * Math.PI * 2;
            const by0 = ovalRy * Math.cos(a0), bz0 = ovalCz + ovalRz * Math.sin(a0);
            const by1 = ovalRy * Math.cos(a1), bz1 = ovalCz + ovalRz * Math.sin(a1);
            // Two triangles per segment (inner quad from centre ring)
            ovalFaceVerts.push(
                ...bodyToThree(ovalCx, ovalCy, ovalCentBz),
                ...bodyToThree(ovalCx, by1, bz1),
                ...bodyToThree(ovalCx, by0, bz0),
            );
        }
        const ovalFaceGeom = new THREE.BufferGeometry();
        ovalFaceGeom.setAttribute("position",
            new THREE.BufferAttribute(new Float32Array(ovalFaceVerts), 3));
        ovalFaceGeom.computeVertexNormals();
        // Dark inner face — very dark (suggests deep duct)
        intakeGroup.add(new THREE.Mesh(
            ovalFaceGeom,
            _stdMat(0x080c10, { metalness: 0.15, roughness: 0.95 }),
        ));

        // Four duct walls running aft from mouth
        const panels = [
            // left wall
            [[iFront, -iHalfW, iTop], [iBack, -iHalfW * 0.8, iTop + 0.1],
             [iBack, -iHalfW * 0.8, iBot - 0.1], [iFront, -iHalfW, iBot]],
            // right wall
            [[iFront, +iHalfW, iTop], [iFront, +iHalfW, iBot],
             [iBack, +iHalfW * 0.8, iBot - 0.1], [iBack, +iHalfW * 0.8, iTop + 0.1]],
            // top ceiling
            [[iFront, -iHalfW, iTop], [iFront, +iHalfW, iTop],
             [iBack, +iHalfW * 0.8, iTop + 0.1], [iBack, -iHalfW * 0.8, iTop + 0.1]],
            // bottom floor
            [[iFront, -iHalfW, iBot], [iBack, -iHalfW * 0.8, iBot - 0.1],
             [iBack, +iHalfW * 0.8, iBot - 0.1], [iFront, +iHalfW, iBot]],
        ];
        for (const p of panels) {
            intakeGroup.add(new THREE.Mesh(_intakePanelGeom(p), iMat));
        }

        // Sharp intake lip — thin torus ring around the mouth opening
        // The lip is circular in the cross-section plane (body y-z plane)
        // and follows the oval outline.
        const lipTorusGeom = new THREE.TorusGeometry(0.70, 0.032, 8, OVAL_SEGS);
        const lip = new THREE.Mesh(
            lipTorusGeom,
            _stdMat(0x606870, { metalness: 0.65, roughness: 0.3 }),
        );
        lip.rotation.y = Math.PI / 2;   // orient ring in body y-z plane
        lip.scale.set(1.0, 1.12, 0.58); // squash to oval
        // Place at body (iFront, 0, ovalCentBz) → three.js
        const [lipX, lipY, lipZ] = bodyToThree(iFront, 0, ovalCentBz);
        lip.position.set(lipX, lipY, lipZ);
        intakeGroup.add(lip);

        // Boundary-layer splitter plate: thin horizontal panel between
        // intake top and fuselage belly — fills the gap visually.
        // Body: bx=[iFront..iBack], by=±iHalfW*0.9, bz = iTop (top of intake)
        const splitterVerts = new Float32Array([
            ...bodyToThree(iFront, -iHalfW * 0.9, iTop),
            ...bodyToThree(iBack,  -iHalfW * 0.75, iTop),
            ...bodyToThree(iBack,  +iHalfW * 0.75, iTop),
            ...bodyToThree(iFront, -iHalfW * 0.9, iTop),
            ...bodyToThree(iBack,  +iHalfW * 0.75, iTop),
            ...bodyToThree(iFront, +iHalfW * 0.9, iTop),
        ]);
        const splitterGeom = new THREE.BufferGeometry();
        splitterGeom.setAttribute("position",
            new THREE.BufferAttribute(splitterVerts, 3));
        splitterGeom.computeVertexNormals();
        intakeGroup.add(new THREE.Mesh(
            splitterGeom,
            _stdMat(0x505860, { metalness: 0.55, roughness: 0.45 }),
        ));

        group.add(intakeGroup);

        // ---- NOSE PITOT ----
        const pitotGeom = new THREE.ConeGeometry(0.055, 0.65, 10);
        const pitot = new THREE.Mesh(
            pitotGeom, _stdMat(F16_COLORS.nozzle, { metalness: 0.75 }),
        );
        pitot.rotation.z = -Math.PI / 2;
        pitot.position.set(8.55, 0, 0);
        group.add(pitot);

        // ---- ENGINE NOZZLE (afterburner can) ----
        // 12-petal nozzle (matches real F100/F110 nozzle count).
        // Petals are metallic with env-map reflections.
        // Inner throat has faint orange-red emissive for "lit engine" look.
        const nozzleX = -7.9;
        const nozzleOuterR = 0.48, nozzleLen = 1.1;
        const nozzleCanGeom = new THREE.CylinderGeometry(
            nozzleOuterR, nozzleOuterR * 0.88, nozzleLen, 16, 1, true,
        );
        const nozzleCan = new THREE.Mesh(
            nozzleCanGeom,
            _stdMat(0x3a3a3a, { metalness: 0.78, roughness: 0.25 }),
        );
        nozzleCan.rotation.z = Math.PI / 2;
        nozzleCan.position.set(nozzleX - nozzleLen / 2, 0, 0);
        group.add(nozzleCan);

        // 12 nozzle petals (up from 10) — shared material, merged geometry
        const PETAL_COUNT = 12;
        const petalGeomBase = new THREE.BoxGeometry(0.55, 0.045, 0.13);
        const petalMat = _stdMat(0x252525, { metalness: 0.85, roughness: 0.20 });
        const petalGeomPairs = [];
        for (let p = 0; p < PETAL_COUNT; p++) {
            const a = (p / PETAL_COUNT) * Math.PI * 2;
            const py = Math.cos(a) * nozzleOuterR * 0.88;
            const pz = Math.sin(a) * nozzleOuterR * 0.88;
            const [px, ptY, ptZ] = bodyToThree(nozzleX - nozzleLen, py, -pz);
            // Build a per-petal transform matrix
            const m = new THREE.Matrix4();
            m.makeTranslation(px, ptY, ptZ);
            // Rotate to face radially outward (petal's Z axis points outward)
            const rotMat = new THREE.Matrix4();
            rotMat.makeRotationY(Math.atan2(pz, py) + Math.PI / 2);
            m.multiply(rotMat);
            petalGeomPairs.push({ geom: petalGeomBase, matrix: m });
        }
        // Merge all petals into one draw call (geometry-merge-static rule)
        const mergedPetalsGeom = _mergeGeoms(petalGeomPairs);
        group.add(new THREE.Mesh(mergedPetalsGeom, petalMat));

        // Inner nozzle throat cone — faint orange emissive for "lit" look
        const throatGeom = new THREE.ConeGeometry(nozzleOuterR * 0.7, 0.5, 16, 1, false);
        const throat = new THREE.Mesh(
            throatGeom,
            _stdMat(0x111111, {
                metalness: 0.4, roughness: 0.8,
                emissive: 0x40180a, emissiveIntensity: 0.3,
            }),
        );
        throat.rotation.z = Math.PI / 2;
        throat.position.set(nozzleX - nozzleLen + 0.05, 0, 0);
        group.add(throat);

        // ---- LERX (Leading Edge Root Extensions) — 3D thin wedge ----
        // Sharp knife-edge strakes running from cockpit fairing to wing root.
        // Visually distinctive F-16 feature.
        const lerxRightShape = [
            [+3.0, +0.55],
            [+1.5, +0.95],
            [-0.4, +1.95],
            [+0.4, +1.95],
            [+1.5, +0.85],
        ];
        const lerxLeftShape = lerxRightShape.map(([x, y]) => [x, -y]);

        function _lerxMesh(poly, name) {
            // Top surface: bz = -0.14 (slightly above fuselage top)
            // Bottom surface: bz = +0.03 (flush with fuselage bottom)
            // LE knife-edge: where top and bottom meet at the outer edge.
            const top = poly.map(([bx, by]) => bodyToThree(bx, by, -0.14));
            const bot = poly.map(([bx, by]) => bodyToThree(bx, by, +0.03));
            const verts = [];
            const N2 = poly.length;
            // Top face (fan)
            for (let i = 1; i < N2 - 1; i++) {
                verts.push(...top[0], ...top[i], ...top[i + 1]);
            }
            // Bottom face (fan, reversed winding)
            for (let i = 1; i < N2 - 1; i++) {
                verts.push(...bot[0], ...bot[i + 1], ...bot[i]);
            }
            // Leading edge strip
            for (let i = 0; i < N2 - 1; i++) {
                verts.push(...top[i], ...bot[i], ...bot[i + 1]);
                verts.push(...top[i], ...bot[i + 1], ...top[i + 1]);
            }
            const geom = new THREE.BufferGeometry();
            geom.setAttribute("position",
                new THREE.BufferAttribute(new Float32Array(verts), 3));
            geom.computeVertexNormals();
            const mesh = new THREE.Mesh(geom, _stdMat(F16_COLORS.fuselage));
            mesh.name = name;
            return mesh;
        }
        group.add(_lerxMesh(lerxRightShape, "lerx_right"));
        group.add(_lerxMesh(lerxLeftShape,  "lerx_left"));

        // ---- VENTRAL FINS — 3D thin wedge ----
        function _ventralFinMesh(side, name) {
            const sign = side === "left" ? -1 : +1;
            const outerPts = [
                bodyToThree(-5.5, sign * 0.38, +0.68),
                bodyToThree(-7.0, sign * 0.38, +0.68),
                bodyToThree(-6.3, sign * 0.72, +1.65),
            ];
            const innerPts = [
                bodyToThree(-5.5, sign * 0.32, +0.72),
                bodyToThree(-7.0, sign * 0.32, +0.72),
                bodyToThree(-6.3, sign * 0.60, +1.60),
            ];
            const o = outerPts, inn = innerPts;
            const verts = new Float32Array([
                ...o[0], ...o[1], ...o[2],
                ...inn[0], ...inn[2], ...inn[1],
                ...o[0], ...inn[0], ...inn[1],
                ...o[0], ...inn[1], ...o[1],
                ...o[1], ...inn[1], ...inn[2],
                ...o[1], ...inn[2], ...o[2],
                ...o[2], ...inn[2], ...inn[0],
                ...o[2], ...inn[0], ...o[0],
            ]);
            const geom = new THREE.BufferGeometry();
            geom.setAttribute("position", new THREE.BufferAttribute(verts, 3));
            geom.computeVertexNormals();
            const mesh = new THREE.Mesh(geom, _stdMat(F16_COLORS.vtail));
            mesh.name = name;
            return mesh;
        }
        group.add(_ventralFinMesh("right", "ventral_fin_right"));
        group.add(_ventralFinMesh("left",  "ventral_fin_left"));

        // ---- SURFACE DECORATION (panel lines, tail flash) ----
        // All static decoration geometry is merged into a single draw call
        // per colour. (Three.js rule: geometry-merge-static)

        // Helper: build a cylindrical panel-line segment in body space.
        // Returns a BufferGeometry in three.js world coords.
        function _panelLineGeom(bx0, bx1, by, bz, r, segs) {
            const len = Math.abs(bx1 - bx0);
            const cg = new THREE.CylinderGeometry(r, r, len, segs);
            const cgeomTmp = new THREE.BufferGeometry();
            const cpArr = cg.attributes.position.array.slice();
            const outPos = new Float32Array(cpArr.length);
            const cnt = cg.attributes.position.count;
            const bxMid = (bx0 + bx1) / 2;
            for (let i = 0; i < cnt; i++) {
                // Cylinder Y-axis = length axis; rotate so length is along X
                const lx = cpArr[i * 3 + 1];  // cylinder's Y → body X offset
                const ly = cpArr[i * 3 + 0];  // cylinder's X → body Y offset
                const lz = cpArr[i * 3 + 2];  // Z unchanged
                const [tx, ty, tz] = bodyToThree(bxMid + lx, by + ly, bz + lz);
                outPos[i * 3]     = tx;
                outPos[i * 3 + 1] = ty;
                outPos[i * 3 + 2] = tz;
            }
            cgeomTmp.setAttribute("position",
                new THREE.BufferAttribute(outPos, 3));
            cgeomTmp.computeVertexNormals();
            cg.dispose();
            return cgeomTmp;
        }

        // Collect all dark-grey panel lines for merging
        const darkPanelGeoms = [];

        // Top spine seam from canopy back to vtail base (~x=-0.5 to -7.0)
        darkPanelGeoms.push({ geom: _panelLineGeom(-0.5, -7.0, 0, -0.50, 0.018, 6) });

        // Short cockpit-side door seam (each side)
        darkPanelGeoms.push({ geom: _panelLineGeom(+3.5, +1.5, +0.55, -0.30, 0.016, 6) });
        darkPanelGeoms.push({ geom: _panelLineGeom(+3.5, +1.5, -0.55, -0.30, 0.016, 6) });

        // Engine bay hoop (vertical cylinder around the nozzle forward of can)
        // Approximate as a short ring at bx ≈ -7.3
        const hoopGeom = new THREE.TorusGeometry(0.46, 0.015, 8, 20);
        hoopGeom.rotateZ(Math.PI / 2);
        // Translate to three.js position
        const hoopMat4 = new THREE.Matrix4().makeTranslation(-7.3, 0, 0);
        darkPanelGeoms.push({ geom: hoopGeom, matrix: hoopMat4 });

        // Merge all dark panel line strips into one mesh
        if (darkPanelGeoms.length > 0) {
            const mergedPanels = _mergeGeoms(darkPanelGeoms);
            const panelsMesh = new THREE.Mesh(
                mergedPanels,
                _stdMat(0x6a7278, { metalness: 0.3, roughness: 0.7 }),
            );
            panelsMesh.name = "_static_decor_6a7278";
            group.add(panelsMesh);
        }

        return group;
    }

    function _vtailMesh() {
        // F-16 vertical tail — thick foil with tip antenna and dorsal fillet.
        const HALF_T = 0.055;  // half-thickness at root (m)
        const vtailGroup = new THREE.Group();
        vtailGroup.name = "vtail";

        // 4 corner points in body frame (bx, bz). by varies for thickness.
        // LE_root=(-3.5, -0.6), LE_tip=(-5.5, -3.4), TE_tip=(-6.5,-3.4), TE_root=(-6.5,-0.6)
        const corners = [
            { bx: -3.5, bz: -0.6 },
            { bx: -5.5, bz: -3.4 },
            { bx: -6.5, bz: -3.4 },
            { bx: -6.5, bz: -0.6 },
        ];
        // Thickness tapers toward tip
        function _ht(idx) { return idx <= 1 ? HALF_T * 0.4 : HALF_T; }
        const left = corners.map((c, i) => bodyToThree(c.bx, -_ht(i), c.bz));
        const right = corners.map((c, i) => bodyToThree(c.bx, +_ht(i), c.bz));

        // Build all faces
        const verts = [];
        const addQuad = (a, b, c, d) => {
            verts.push(...a, ...b, ...c, ...a, ...c, ...d);
        };
        // Left face, right face
        addQuad(left[0], left[1], left[2], left[3]);
        addQuad(right[0], right[3], right[2], right[1]);
        // LE edge (0-1)
        addQuad(left[0], right[0], right[1], left[1]);
        // TE edge (3-2)
        addQuad(left[3], left[2], right[2], right[3]);
        // Root (bottom) cap
        addQuad(left[0], left[3], right[3], right[0]);
        // Tip cap
        addQuad(left[1], right[1], right[2], left[2]);

        const geom = new THREE.BufferGeometry();
        geom.setAttribute("position", new THREE.BufferAttribute(new Float32Array(verts), 3));
        geom.computeVertexNormals();
        vtailGroup.add(new THREE.Mesh(geom, _stdMat(F16_COLORS.vtail)));

        // Dorsal fillet at the tail root — a small triangular fairing
        const filletVerts = new Float32Array([
            ...bodyToThree(-3.5, -0.25, -0.6),
            ...bodyToThree(-3.5, +0.25, -0.6),
            ...bodyToThree(-4.2,  0.0, -1.0),
            ...bodyToThree(-4.2, -0.22, -1.0),
            ...bodyToThree(-4.2, +0.22, -1.0),
            ...bodyToThree(-3.5, -0.25, -0.6),
            // wrap sides
            ...bodyToThree(-3.5, -0.25, -0.6),
            ...bodyToThree(-4.2, -0.22, -1.0),
            ...bodyToThree(-4.2, +0.22, -1.0),
            ...bodyToThree(-3.5, -0.25, -0.6),
            ...bodyToThree(-4.2, +0.22, -1.0),
            ...bodyToThree(-3.5, +0.25, -0.6),
        ]);
        const filletGeom = new THREE.BufferGeometry();
        filletGeom.setAttribute("position", new THREE.BufferAttribute(filletVerts, 3));
        filletGeom.computeVertexNormals();
        vtailGroup.add(new THREE.Mesh(filletGeom, _stdMat(F16_COLORS.fuselage)));

        // Tip antenna stub (small slim box at the very tip)
        const antennaGeom = new THREE.BoxGeometry(0.08, 0.06, 0.32);
        const antenna = new THREE.Mesh(
            antennaGeom,
            _stdMat(0x909898, { metalness: 0.65, roughness: 0.3 }),
        );
        const [ax, ay, az] = bodyToThree(-5.7, 0, -3.55);
        antenna.position.set(ax, ay, az);
        vtailGroup.add(antenna);

        // Formation light strip (cyan) on the LE
        const formLightGeom = new THREE.BoxGeometry(0.04, 0.04, 1.6);
        const formLight = new THREE.Mesh(
            formLightGeom,
            new THREE.MeshBasicMaterial({ color: 0x00ffcc }),
        );
        const [flx, fly, flz] = bodyToThree(-4.6, 0.0, -2.1);
        formLight.position.set(flx, fly, flz);
        // Tilt to follow the LE sweep: LE goes from (-3.5,-0.6) to (-5.5,-3.4)
        // => run=-2, rise=-2.8, angle from vertical ≈ atan2(2,2.8)
        formLight.rotation.x = -Math.atan2(2.0, 2.8);
        vtailGroup.add(formLight);

        // Tail flash — darker grey horizontal band near the top of the vtail
        // (both left and right faces), imitating squadron / USAF markings.
        // Body coords: bx=-4.5 to -6.5, bz=-2.6 to -3.0, each lateral surface.
        const flashMat = _stdMat(0x4a5260, { metalness: 0.4, roughness: 0.5 });
        const flashBands = [
            // Right face (by = +HALF_T): bx=-4.5→-6.5, bz=-2.6→-3.0
            [bodyToThree(-4.5, +HALF_T + 0.001, -2.6),
             bodyToThree(-6.5, +HALF_T + 0.001, -2.6),
             bodyToThree(-6.5, +HALF_T + 0.001, -3.0),
             bodyToThree(-4.5, +HALF_T + 0.001, -3.0)],
            // Left face (by = -HALF_T)
            [bodyToThree(-4.5, -HALF_T - 0.001, -2.6),
             bodyToThree(-4.5, -HALF_T - 0.001, -3.0),
             bodyToThree(-6.5, -HALF_T - 0.001, -3.0),
             bodyToThree(-6.5, -HALF_T - 0.001, -2.6)],
        ];
        for (const corners of flashBands) {
            vtailGroup.add(new THREE.Mesh(
                _quadGeom(corners[0], corners[1], corners[2], corners[3]),
                flashMat,
            ));
        }

        return vtailGroup;
    }

    // Rudder hinge: at the LE_top corner (the forward-upper edge of the
    // rudder flap, where it pivots vertically against the vtail).
    const RUDDER_HINGE_BODY = [-6.4, 0, -0.6];
    // Rudder polygon vertices relative to hinge in body (x, z) plane
    // (rudder lies in the body x-z plane, y = 0 throughout).
    const RUDDER_POLY_REL = [
        // [bx_rel, bz_rel] relative to hinge
        [ 0.0,   0.0],          // LE_top (hinge)
        [ 0.0,  -2.8],          // LE_bottom
        [-0.3,  -2.8],          // TE_bottom
        [-0.3,   0.0],          // TE_top
    ];

    function _rudderHingeGroup() {
        const group = new THREE.Group();
        group.name = "rudder";
        const [hx, hy, hz] = bodyToThree(...RUDDER_HINGE_BODY);
        group.position.set(hx, hy, hz);
        // Build thick foil mesh relative to hinge in body x-z plane
        const HALF_T = 0.040;
        const c = RUDDER_POLY_REL.map(([bx, bz]) => ({ bx, bz }));
        const left  = c.map(p => bodyToThree(p.bx, -HALF_T, p.bz));
        const right = c.map(p => bodyToThree(p.bx, +HALF_T, p.bz));
        const verts = [];
        const addQ = (a,b,cc,d) => verts.push(...a,...b,...cc,...a,...cc,...d);
        addQ(left[0], left[1], left[2], left[3]);
        addQ(right[0], right[3], right[2], right[1]);
        addQ(left[0], right[0], right[1], left[1]);    // LE
        addQ(left[2], left[3], right[3], right[2]);    // TE
        addQ(left[0], left[3], right[3], right[0]);    // top edge
        addQ(left[1], right[1], right[2], left[2]);    // bot edge
        const geom = new THREE.BufferGeometry();
        geom.setAttribute("position", new THREE.BufferAttribute(new Float32Array(verts), 3));
        geom.computeVertexNormals();
        group.add(new THREE.Mesh(geom, _stdMat(F16_COLORS.rudder)));
        return group;
    }

    function _stabHingeGroup(side) {
        const group = new THREE.Group();
        group.name = side === "left" ? "stab_left" : "stab_right";
        const sign = side === "left" ? -1 : +1;
        const [hbx, hby, hbz] = STAB_RIGHT_HINGE_BODY;
        const [hx, hy, hz] = bodyToThree(hbx, sign * hby, hbz);
        group.position.set(hx, hy, hz);
        // Thick foil: top skin at bz=-0.055, bottom at bz=+0.055
        const SKIN = 0.050;
        const top = STAB_RIGHT_POLY_REL.map(([bx, by]) => bodyToThree(bx, sign * by, -SKIN));
        const bot = STAB_RIGHT_POLY_REL.map(([bx, by]) => bodyToThree(bx, sign * by, +SKIN));
        const verts = [];
        // top face
        verts.push(...top[0],...top[1],...top[2], ...top[0],...top[2],...top[3]);
        // bottom face (reversed winding)
        verts.push(...bot[0],...bot[2],...bot[1], ...bot[0],...bot[3],...bot[2]);
        // LE edge (0-1)
        verts.push(...top[0],...bot[0],...bot[1], ...top[0],...bot[1],...top[1]);
        // TE edge (3-2)
        verts.push(...top[3],...top[2],...bot[2], ...top[3],...bot[2],...bot[3]);
        // inboard cap (0-3)
        verts.push(...top[0],...top[3],...bot[3], ...top[0],...bot[3],...bot[0]);
        // outboard cap (1-2)
        verts.push(...top[1],...bot[1],...bot[2], ...top[1],...bot[2],...top[2]);
        const geom = new THREE.BufferGeometry();
        geom.setAttribute("position", new THREE.BufferAttribute(new Float32Array(verts), 3));
        geom.computeVertexNormals();
        group.add(new THREE.Mesh(geom, _stdMat(F16_COLORS.stab)));
        return group;
    }

    function _aileronHingeGroup(side) {
        const group = new THREE.Group();
        group.name = side === "left" ? "aileron_left" : "aileron_right";
        const sign = side === "left" ? -1 : +1;
        const [hbx, hby, hbz] = AILERON_RIGHT_HINGE_BODY;
        const [hx, hy, hz] = bodyToThree(hbx, sign * hby, hbz);
        group.position.set(hx, hy, hz);
        // Thick foil
        const SKIN = 0.040;
        const top = AILERON_RIGHT_POLY_REL.map(([bx, by]) => bodyToThree(bx, sign * by, -SKIN));
        const bot = AILERON_RIGHT_POLY_REL.map(([bx, by]) => bodyToThree(bx, sign * by, +SKIN));
        const verts = [];
        verts.push(...top[0],...top[1],...top[2], ...top[0],...top[2],...top[3]);
        verts.push(...bot[0],...bot[2],...bot[1], ...bot[0],...bot[3],...bot[2]);
        verts.push(...top[0],...bot[0],...bot[1], ...top[0],...bot[1],...top[1]);
        verts.push(...top[3],...top[2],...bot[2], ...top[3],...bot[2],...bot[3]);
        verts.push(...top[0],...top[3],...bot[3], ...top[0],...bot[3],...bot[0]);
        verts.push(...top[1],...bot[1],...bot[2], ...top[1],...bot[2],...top[2]);
        const geom = new THREE.BufferGeometry();
        geom.setAttribute("position", new THREE.BufferAttribute(new Float32Array(verts), 3));
        geom.computeVertexNormals();
        group.add(new THREE.Mesh(geom, _stdMat(F16_COLORS.aileron)));
        return group;
    }

    function _wingtipLauncher(side, name) {
        // AIM-9 Sidewinder on wingtip launcher rail.
        // All AIM-9 sub-meshes are merged into ONE mesh per launcher to cut
        // ~8 draw calls per side down to 2 (one for rail, one for missile+fins).
        // (Three.js rule: geometry-merge-static)
        const sign = side === "left" ? -1 : +1;

        const launcherGroup = new THREE.Group();
        launcherGroup.name = name;

        // Launcher rail: thin cylinder below the missile
        const railGeom = new THREE.CylinderGeometry(0.045, 0.045, 2.6, 8);
        railGeom.rotateZ(Math.PI / 2);
        const rail = new THREE.Mesh(
            railGeom, _stdMat(0x707878, { metalness: 0.6, roughness: 0.35 }),
        );
        launcherGroup.add(rail);

        // Collect all AIM-9 body geometry for merging.
        // Each sub-shape has a local transform applied, then we merge all
        // into a single BufferGeometry under one mesh.
        const missileGeomPairs = [];

        // AIM-9 body cylinder
        const bodyG = new THREE.CylinderGeometry(0.065, 0.065, 2.85, 14);
        bodyG.rotateZ(Math.PI / 2);
        const bm4 = new THREE.Matrix4().makeTranslation(0, 0.115, 0);
        missileGeomPairs.push({ geom: bodyG, matrix: bm4 });

        // Seeker nose cone
        const seekerG = new THREE.ConeGeometry(0.065, 0.22, 14);
        seekerG.rotateZ(-Math.PI / 2);
        const sm4 = new THREE.Matrix4().makeTranslation(1.535, 0.115, 0);
        missileGeomPairs.push({ geom: seekerG, matrix: sm4 });

        // Tail nozzle cone
        const tailG = new THREE.ConeGeometry(0.065, 0.18, 14);
        tailG.rotateZ(Math.PI / 2);
        const tm4 = new THREE.Matrix4().makeTranslation(-1.515, 0.115, 0);
        missileGeomPairs.push({ geom: tailG, matrix: tm4 });

        // Cruciform fins helper — pushes 4 fin geometries into the pair list
        function _cruciformFinGeoms(bxCentre, span, chord, thick, offsetY) {
            for (let k = 0; k < 4; k++) {
                const finW = (k % 2 === 0) ? span : thick;
                const finH = (k % 2 === 0) ? thick : span;
                const finG = new THREE.BoxGeometry(chord, finH, finW);
                const fy = (k % 2 === 1) ? (k === 1 ? span * 0.5 : -span * 0.5) : 0;
                const fz = (k % 2 === 0) ? (k === 0 ? span * 0.5 : -span * 0.5) : 0;
                const fm4 = new THREE.Matrix4().makeTranslation(
                    bxCentre, offsetY + fy, fz * sign,
                );
                missileGeomPairs.push({ geom: finG, matrix: fm4 });
            }
        }

        // Mid-body delta fins
        _cruciformFinGeoms(-0.5, 0.30, 0.45, 0.018, 0.115);
        // Forward canard fins
        _cruciformFinGeoms(+0.9, 0.14, 0.22, 0.014, 0.115);

        // Merge everything into a single AIM-9 mesh
        const mergedMissile = _mergeGeoms(missileGeomPairs);
        const missileMesh = new THREE.Mesh(
            mergedMissile,
            _stdMat(0xb8c0c8, { metalness: 0.55, roughness: 0.30 }),
        );
        missileMesh.name = name;    // damage system looks this up via getObjectByName
        launcherGroup.add(missileMesh);

        // Dispose intermediate geometries
        for (const { geom } of missileGeomPairs) geom.dispose();

        // Position whole launcher group at wingtip LE
        const [tx, ty, tz] = bodyToThree(-1.85, sign * 4.62, -0.08);
        launcherGroup.position.set(tx, ty, tz);
        return launcherGroup;
    }

    const B747_COLORS = {
        // Healthy nacelle tint (used by engine-damage lerp). Other colours
        // come from the GLB material directly.
        engine_healthy: 0xc8cbcd,
        engine_failed:  0x404040,
    };

    // Procedural disk texture used by the per-engine failure marker.
    // A small, hard-edged solid black dot. Generated once and shared.
    let _b747SmokeTex = null;
    function _b747SmokeTexture() {
        if (_b747SmokeTex) return _b747SmokeTex;
        const size = 64;
        const cv = document.createElement("canvas");
        cv.width = cv.height = size;
        const ctx = cv.getContext("2d");
        ctx.clearRect(0, 0, size, size);
        ctx.beginPath();
        ctx.arc(size / 2, size / 2, size / 2 - 2, 0, 2 * Math.PI);
        ctx.fillStyle = "#000000";
        ctx.fill();
        const tex = new THREE.CanvasTexture(cv);
        _b747SmokeTex = tex;
        return tex;
    }

    // Helper: a small black-dot marker anchored to an engine. Pops on
    // engine failure (driven by applyDamageState via engines_mu).
    function _b747SmokeSprite() {
        const smoke = new THREE.Sprite(new THREE.SpriteMaterial({
            map: _b747SmokeTexture(),
            color: 0x000000,
            transparent: true,
            opacity: 0.0,
            depthWrite: false,
        }));
        smoke.scale.set(2.5, 2.5, 1.0);
        smoke.visible = false;
        return smoke;
    }

    // ---- B-747 GLB loader + animation hooks ----
    //
    // The B-747 silhouette comes from a binary glTF (b747.glb) embedded in
    // the HTML as base64. The model has nine named nodes that we wire up
    // for animation:
    //
    //   aileron_L, aileron_R              — wing trailing-edge slabs
    //   elevator_L, elevator_R            — horizontal-stab trailing slabs
    //   rudder                             — vertical-stab trailing slab
    //   Engine 1, Engine 2, Engine 3, Engine 4
    //
    // The remaining ~3829 anonymous nodes are static fuselage / wing
    // geometry. The GLB axis convention (Blender export, glTF v2):
    //   +x = wingspan toward port (left); -x = starboard (right)
    //   +y = up
    //   +z = forward (toward tail); -z = forward toward nose
    // (Verified empirically: rudder bbox z∈[-36, -29] sits at the tail.)
    //
    // Three viewer convention (see bodyToThree at line ~250):
    //   local +x = body forward (nose direction)
    //   local +y = up
    //   local +z = right wing
    //
    // To map GLB → viewer: rotate -90° around three.y so GLB-(-z)→three(+x)
    // and GLB(-x)→three(+z). Concretely, GLB_to_three(gx, gy, gz) =
    // (-gz, gy, -gx). That's `aircraft.add(root)` with
    // `root.rotation.y = -Math.PI/2` and `root.scale.set(1, 1, -1)` — but
    // mirror-scaling flips winding and breaks lighting. Instead we apply
    // an Euler rotation: rot.y = +Math.PI/2 maps (gx, gy, gz) →
    // (gz, gy, -gx); then rot.x = 0 keeps y as up. We want (-gz, gy, -gx),
    // so rot.y = -Math.PI/2 maps (gx, gy, gz) → (-gz, gy, gx); then we
    // additionally need to flip the +x→-x sign. Easiest fix: combine
    // rot.y = -Math.PI/2 with rot.z = Math.PI to flip lateral axis. We
    // verify visually below (nose at +x, right wing at +z).
    //
    // Scale: bbox length is ~70.7 m and span is ~65.3 m at unit scale,
    // matching the B-747 in metres — no further scale needed.

    function _loadB747GlbInto(aircraft, hingeRefs, smokeRefs) {
        const b64 = (typeof window !== "undefined") ? window.B747_GLB_B64 : "";
        if (!b64) {
            console.warn("B-747 GLB missing; the named-node animation hooks will no-op.");
            return;
        }
        if (typeof THREE.GLTFLoader !== "function") {
            console.error("THREE.GLTFLoader is not loaded; B-747 GLB cannot be parsed.");
            return;
        }
        // Decode base64 → Uint8Array → Blob URL (avoids string-vs-binary
        // pitfalls; the GLB is binary).
        const bin = atob(b64);
        const bytes = new Uint8Array(bin.length);
        for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
        const blob = new Blob([bytes.buffer], { type: "model/gltf-binary" });
        const url = URL.createObjectURL(blob);
        const loader = new THREE.GLTFLoader();
        loader.load(
            url,
            (gltf) => {
                URL.revokeObjectURL(url);
                _onB747GlbLoaded(gltf, aircraft, hingeRefs, smokeRefs);
            },
            undefined,
            (err) => {
                URL.revokeObjectURL(url);
                console.error("B-747 GLTFLoader failed:", err);
            },
        );
    }

    function _onB747GlbLoaded(gltf, aircraft, hingeRefs, smokeRefs) {
        const root = gltf.scene;
        // GLB → viewer body frame. Verified empirically from the model's
        // bboxes (see comments above _loadB747GlbInto):
        //   GLB +z = nose (forward), GLB +y = up, GLB +x = lateral.
        // Viewer:
        //   +x = forward (nose); +y = up; +z = right wing.
        // A single −π/2 rotation about +y gives the correct mapping
        //   GLB +z → viewer +x (forward),  GLB +y → viewer +y (up).
        // Pure rotation (det = +1) — no winding flip, no DoubleSide
        // workaround required. The B-747 silhouette is left/right
        // symmetric, so any residual mirror in the lateral mapping is
        // visually undetectable; engine numbering is just programmatic.
        // NB: +π/2 (not −π/2) — earlier sign produced GLB +z → viewer −x
        // i.e. the plane flew tail-first.
        root.rotation.y = Math.PI / 2;
        root.scale.set(1, 1, 1);
        root.name = "_static_decor_b747_glb";
        aircraft.add(root);
        // We need worldMatrix to be current before searching for nodes.
        aircraft.updateMatrixWorld(true);

        // Find named nodes and wire them up.
        const NAMES = [
            "aileron_L", "aileron_R",
            "elevator_L", "elevator_R",
            "rudder",
            "Engine 1", "Engine 2", "Engine 3", "Engine 4",
        ];
        const found = {};
        root.traverse((obj) => {
            if (obj.name && NAMES.indexOf(obj.name) >= 0) {
                found[obj.name] = obj;
            }
        });
        const missing = NAMES.filter((n) => !(n in found));
        if (missing.length) {
            console.warn("B-747 GLB: missing named nodes:", missing);
        }

        // Wire control-surface hinges. Each named mesh becomes the *child*
        // of an existing placeholder hinge group (aileron_left etc.).
        // Re-parent + adjust pivot.
        const surfaceMap = [
            { glb: "aileron_L",  hinge: "aileron_left"  },
            { glb: "aileron_R",  hinge: "aileron_right" },
            { glb: "elevator_L", hinge: "stab_left"     },
            { glb: "elevator_R", hinge: "stab_right"    },
            { glb: "rudder",     hinge: "rudder"        },
        ];
        for (const { glb, hinge } of surfaceMap) {
            const meshNode = found[glb];
            const hingeGroup = hingeRefs[hinge];
            if (!meshNode || !hingeGroup) continue;
            // Compute world-space bbox of this mesh in aircraft-local
            // coords, find the hinge pivot, then reparent the mesh to a
            // new sub-group whose origin sits on the pivot.
            meshNode.geometry.computeBoundingBox();
            meshNode.updateWorldMatrix(true, false);
            aircraft.updateWorldMatrix(true, false);
            const invAir = new THREE.Matrix4()
                .copy(aircraft.matrixWorld).invert();
            const meshToAircraft = new THREE.Matrix4()
                .multiplyMatrices(invAir, meshNode.matrixWorld);
            const localBox = new THREE.Box3()
                .copy(meshNode.geometry.boundingBox)
                .applyMatrix4(meshToAircraft);
            // Determine pivot in aircraft-local frame.
            const c = new THREE.Vector3();
            localBox.getCenter(c);
            const pivot = c.clone();
            if (glb === "aileron_L" || glb === "aileron_R") {
                pivot.x = localBox.max.x;  // forward edge
                pivot.z = (Math.abs(localBox.min.z) < Math.abs(localBox.max.z))
                    ? localBox.min.z : localBox.max.z;  // inboard
            } else if (glb === "elevator_L" || glb === "elevator_R") {
                pivot.x = localBox.max.x;  // forward edge
                pivot.z = (Math.abs(localBox.min.z) < Math.abs(localBox.max.z))
                    ? localBox.min.z : localBox.max.z;  // inboard
            } else if (glb === "rudder") {
                pivot.x = localBox.max.x;  // forward edge
                pivot.y = localBox.min.y;  // bottom
                pivot.z = 0.5 * (localBox.min.z + localBox.max.z);
            }
            // Place the hinge group at the pivot in aircraft-local space.
            hingeGroup.position.copy(pivot);
            hingeGroup.updateMatrix();
            // Refresh world matrix before attach (see engine loop below).
            hingeGroup.updateWorldMatrix(false, false);
            // Re-parent the mesh under the hinge group, preserving its
            // world transform (so the geometry stays put visually).
            hingeGroup.attach(meshNode);
            meshNode.updateMatrix();
        }

        // Wire engine groups: re-parent each Engine N node to the matching
        // engine_<id> placeholder, preserving world transform. Also drop a
        // smoke sprite anchored to the engine's aft point.
        const engineMap = [
            ["Engine 1", "engine_1"],
            ["Engine 2", "engine_2"],
            ["Engine 3", "engine_3"],
            ["Engine 4", "engine_4"],
        ];
        // Engine positions in aircraft-local frame (after the +π/2 y-axis
        // root rotation). Derived offline from the GLB's accessor min/max
        // for each "Engine N" mesh, plus the airplane/world node
        // translations baked into the rotation. Hard-coded because the
        // run-time bbox path was placing the marker at the aircraft origin
        // for reasons not yet understood.
        const ENGINE_LOCAL = {
            engine_1: new THREE.Vector3(-2.22, -6.86, -21.09),
            engine_2: new THREE.Vector3( 5.72, -7.26, -12.08),
            engine_3: new THREE.Vector3( 5.72, -7.26,  12.07),
            engine_4: new THREE.Vector3(-2.22, -6.86,  21.08),
        };
        aircraft.updateWorldMatrix(true, true);
        for (const [glb, anchor] of engineMap) {
            const meshNode = found[glb];
            const anchorGroup = hingeRefs[anchor];
            if (!meshNode || !anchorGroup) continue;
            const localCen = ENGINE_LOCAL[anchor].clone();
            // Re-parent meshNode into anchorGroup (preserve world tx)
            anchorGroup.position.copy(localCen);
            anchorGroup.updateMatrix();
            anchorGroup.updateWorldMatrix(false, false);
            anchorGroup.attach(meshNode);
            meshNode.updateMatrix();
            // Move the smoke sprite OUT of the anchor chain and attach it
            // directly under aircraft at the engine's aircraft-local pos.
            const smoke = smokeRefs[anchor];
            if (smoke) {
                if (smoke.parent && smoke.parent !== aircraft) {
                    smoke.parent.remove(smoke);
                }
                aircraft.add(smoke);
                smoke.position.copy(localCen);
                smoke.updateMatrix();
                smoke.updateMatrixWorld(true);
            }
        }

        // Mark inert GLB descendants matrixAutoUpdate=false to avoid
        // recomputing thousands of static local matrices each frame.
        // We only freeze the static GLB silhouette (the
        // `_static_decor_b747_glb` subtree) — leaving hinge and engine
        // subtrees with their default matrixAutoUpdate=true so per-tick
        // .rotation / .position / .scale updates propagate.
        root.traverse((obj) => {
            obj.matrixAutoUpdate = false;
            obj.updateMatrix();
        });

        // Late-registration: the sectionMaterials cache was populated
        // synchronously during scene init — but the engine_<id> groups
        // were empty at that time, so they have no entry. Add them now.
        if (typeof _registerB747EngineMaterials === "function") {
            _registerB747EngineMaterials();
        }
    }

    function buildB747Aircraft(geometry) {
        const aircraft = new THREE.Group();
        aircraft.name = "aircraft";

        // ---- Hinge / engine placeholders ----
        // The viewer's per-tick animation looks up these groups by name and
        // sets .rotation.<axis> = deflection. The async GLB loader will
        // re-parent the matching GLB nodes into these groups (preserving
        // world transform) and bake the hinge pivot into geometry. Until
        // the GLB load finishes, these groups are empty and the animation
        // hooks no-op visually.
        function _hinge(name) {
            const g = new THREE.Group();
            g.name = name;
            return g;
        }
        const hingeRefs = {};
        const surfaceNames = [
            "aileron_left", "aileron_right",
            "stab_left", "stab_right", "rudder",
            "engine_1", "engine_2", "engine_3", "engine_4",
        ];
        for (const n of surfaceNames) {
            const g = _hinge(n);
            hingeRefs[n] = g;
            aircraft.add(g);
        }

        // Engine smoke sprites — pre-created and attached to each engine
        // anchor so applyDamageState can find them by `engine_<id>_smoke`.
        const smokeRefs = {};
        for (let i = 1; i <= 4; i++) {
            const anchor = hingeRefs["engine_" + i];
            const sprite = _b747SmokeSprite();
            sprite.name = "engine_" + i + "_smoke";
            anchor.add(sprite);
            smokeRefs["engine_" + i] = sprite;
        }

        // Kick off async GLB load. When done, _onB747GlbLoaded reparents
        // the named GLB nodes into hingeRefs and adds the static silhouette
        // under aircraft as `_static_decor_b747_glb`.
        _loadB747GlbInto(aircraft, hingeRefs, smokeRefs);

        return aircraft;
    }

    function buildF16Aircraft(geometry) {
        const aircraft = new THREE.Group();
        aircraft.name = "aircraft";

        // Fuselage (always present; uses the geometry's "fuselage_main"
        // section if it exists, but visual is fixed).
        const fuse = _fuselageGroup();
        aircraft.add(fuse);

        // Wings: 6 sections (3 per side) defined by hardcoded polygons
        // that abut along their inner edges so the assembled wing is
        // continuous. Mesh names match the YAML section names so damage
        // viz can find them.
        for (const [name, poly] of Object.entries(RIGHT_WING_POLYGONS)) {
            aircraft.add(_flatSectionMesh(poly, F16_COLORS.wing, name));
            const leftName = name.replace("right_", "left_");
            aircraft.add(_flatSectionMesh(_mirrorY(poly),
                                           F16_COLORS.wing, leftName));
        }

        // Stabilators: hinge-pivoted groups so their pitch deflection
        // can be animated from traj.stab[idx] each frame.
        aircraft.add(_stabHingeGroup("right"));
        aircraft.add(_stabHingeGroup("left"));

        // Vertical tail + rudder (rudder is a hinge-pivoted group so its
        // yaw deflection animates from traj.dir[idx]).
        aircraft.add(_vtailMesh());
        aircraft.add(_rudderHingeGroup());

        // Ailerons (hinge-pivoted; differential deflection from traj.ail).
        aircraft.add(_aileronHingeGroup("right"));
        aircraft.add(_aileronHingeGroup("left"));

        // Wingtip launchers
        aircraft.add(_wingtipLauncher("right", "launcher_right"));
        aircraft.add(_wingtipLauncher("left",  "launcher_left"));

        // Underwing pylons (static, no names matching protected list).
        // Merge all pylon geometry into a single mesh (geometry-merge-static).
        const pylonGeomPairs = [];
        const pylonConfigs = [
            [+1, 2.6], [+1, 3.3], [-1, 2.6], [-1, 3.3],
        ];
        for (const [psign, pylonY] of pylonConfigs) {
            const [px, py, pz] = bodyToThree(-1.6, psign * pylonY, +0.22);
            const strutG = new THREE.BoxGeometry(0.55, 0.38, 0.10);
            pylonGeomPairs.push({
                geom: strutG,
                matrix: new THREE.Matrix4().makeTranslation(px, py, pz),
            });
            const shelfG = new THREE.BoxGeometry(0.45, 0.065, 0.22);
            pylonGeomPairs.push({
                geom: shelfG,
                matrix: new THREE.Matrix4().makeTranslation(px, py - 0.22, pz),
            });
        }
        const mergedPylons = _mergeGeoms(pylonGeomPairs);
        const pylonsMesh = new THREE.Mesh(
            mergedPylons,
            _stdMat(0x7a8288, { metalness: 0.5, roughness: 0.45 }),
        );
        pylonsMesh.name = "_static_decor_pylons";
        aircraft.add(pylonsMesh);
        for (const { geom } of pylonGeomPairs) geom.dispose();

        // Wing leading-edge accent strips — thin dark strip along the LE of
        // each wing section, suggesting the LE-droop / honeycomb edge.
        // (One merged dark-edge mesh per wing side for efficiency.)
        function _wingLEStrips(polys, sign) {
            const stripGeoms = [];
            for (const poly of polys) {
                // LE = edge from corner[0] to corner[1]
                const bx0 = poly[0][0], by0 = poly[0][1];
                const bx1 = poly[1][0], by1 = poly[1][1];
                // Thin flat quad just above the wing surface along the LE
                const bxMid = (bx0 + bx1) / 2, byMid = (by0 + by1) / 2;
                const c0 = bodyToThree(bx0, by0, -0.07);
                const c1 = bodyToThree(bx1, by1, -0.07);
                const c2 = bodyToThree(bx1, by1, +0.07);
                const c3 = bodyToThree(bx0, by0, +0.07);
                stripGeoms.push({ geom: _quadGeom(c0, c1, c2, c3) });
            }
            return _mergeGeoms(stripGeoms);
        }
        const rightPolys = Object.values(RIGHT_WING_POLYGONS);
        const leftPolys  = rightPolys.map(p => _mirrorY(p));
        const wingEdgeMat = _stdMat(0x404850, { metalness: 0.4, roughness: 0.6 });
        const rStripMesh = new THREE.Mesh(_wingLEStrips(rightPolys, +1), wingEdgeMat);
        rStripMesh.name = "_static_decor_wing_le_r";
        aircraft.add(rStripMesh);
        const lStripMesh = new THREE.Mesh(_wingLEStrips(leftPolys, -1), wingEdgeMat);
        lStripMesh.name = "_static_decor_wing_le_l";
        aircraft.add(lStripMesh);

        return aircraft;
    }

    function buildAircraft(geometry) {
        return IS_B747 ? buildB747Aircraft(geometry) : buildF16Aircraft(geometry);
    }

    const aircraft = buildAircraft(log.geometry);
    scene.add(aircraft);

    // Static-decoration meshes never move relative to the aircraft, so
    // we can disable per-frame local-matrix recomputation on them. The
    // aircraft Group itself still updates each frame; children inherit
    // its world matrix automatically. Hinge-pivoted control surfaces
    // (stab_*, aileron_*, rudder) DO change their local rotation, so
    // they must keep matrixAutoUpdate=true.
    const _MOVING_NAMES = new Set([
        "stab_left", "stab_right", "aileron_left", "aileron_right",
        "rudder", "exhaust",
        // B-747: engine_<id> placeholders are repositioned once after the
        // async GLB load, and the per-engine smoke Sprite under each
        // group is scaled / faded by applyDamageState().
        "engine_1", "engine_2", "engine_3", "engine_4",
    ]);
    aircraft.traverse((obj) => {
        if (obj === aircraft) return;
        // Walk up to find which top-level child this belongs to so we
        // can skip the entire hinge-group subtree.
        let p = obj;
        while (p && p.parent && p.parent !== aircraft) p = p.parent;
        if (p && _MOVING_NAMES.has(p.name)) return;
        obj.matrixAutoUpdate = false;
        obj.updateMatrix();
    });

    // ---- Instrument-panel helpers ----
    function _setNeedle(id, value, vmin, vmax, degMin, degMax) {
        const el = document.getElementById(id);
        if (!el) return;
        const t = (value - vmin) / (vmax - vmin);
        const tc = Math.max(0, Math.min(1, t));
        const deg = degMin + tc * (degMax - degMin);
        el.setAttribute("transform", "rotate(" + deg.toFixed(1) + ")");
    }

    function _setText(id, txt) {
        const el = document.getElementById(id);
        if (el) el.textContent = txt;
    }

    // Build static dial decorations (ticks + labels + ADI pitch rungs)
    // once at scene init.
    (function _buildPanelStatics() {
        // ADI pitch ladder rungs at ±5°, ±10°, ±15°, ±20° (1.6 px per °).
        const ladder = document.getElementById("adi-pitch-ladder");
        if (ladder) {
            const ADI_PIXELS_PER_DEG = 1.6;
            for (let d = -20; d <= 20; d += 5) {
                if (d === 0) continue;
                const y = -d * ADI_PIXELS_PER_DEG;
                const w = (Math.abs(d) % 10 === 0) ? 14 : 7;
                const segL = document.createElementNS(
                    "http://www.w3.org/2000/svg", "line");
                segL.setAttribute("x1", -w);
                segL.setAttribute("y1", y);
                segL.setAttribute("x2", -3);
                segL.setAttribute("y2", y);
                segL.setAttribute("class", "adi-pitch-rung");
                ladder.appendChild(segL);
                const segR = document.createElementNS(
                    "http://www.w3.org/2000/svg", "line");
                segR.setAttribute("x1", 3);
                segR.setAttribute("y1", y);
                segR.setAttribute("x2", w);
                segR.setAttribute("y2", y);
                segR.setAttribute("class", "adi-pitch-rung");
                ladder.appendChild(segR);
                if (Math.abs(d) % 10 === 0) {
                    const lbl = document.createElementNS(
                        "http://www.w3.org/2000/svg", "text");
                    lbl.setAttribute("x", w + 3);
                    lbl.setAttribute("y", y + 1.8);
                    lbl.setAttribute("class", "adi-pitch-label");
                    lbl.textContent = String(Math.abs(d));
                    ladder.appendChild(lbl);
                }
            }
        }

        // Helper: tick mark for round dials.
        function _makeRoundDialTicks(ticksId, labelsId, vmin, vmax,
                                    degMin, degMax, majorEvery, minorStep,
                                    formatLabel) {
            const ticks = document.getElementById(ticksId);
            const labels = document.getElementById(labelsId);
            if (!ticks) return;
            for (let v = vmin; v <= vmax + 1e-6; v += minorStep) {
                const t = (v - vmin) / (vmax - vmin);
                const deg = degMin + t * (degMax - degMin);
                const isMajor = ((v - vmin) % majorEvery === 0)
                              || Math.abs((v - vmin) % majorEvery) < 1e-6;
                const len = isMajor ? 5 : 2.5;
                const r1 = 42, r2 = 42 - len;
                const a = (deg - 90) * Math.PI / 180;  // SVG: 0° = 3 o'clock
                const x1 = Math.cos(a) * r1;
                const y1 = Math.sin(a) * r1;
                const x2 = Math.cos(a) * r2;
                const y2 = Math.sin(a) * r2;
                const tick = document.createElementNS(
                    "http://www.w3.org/2000/svg", "line");
                tick.setAttribute("x1", x1.toFixed(2));
                tick.setAttribute("y1", y1.toFixed(2));
                tick.setAttribute("x2", x2.toFixed(2));
                tick.setAttribute("y2", y2.toFixed(2));
                if (isMajor) tick.setAttribute("class", "major");
                ticks.appendChild(tick);
                if (isMajor && labels && formatLabel) {
                    const txt = formatLabel(v);
                    if (txt !== null && txt !== undefined && txt !== "") {
                        const lr = 30;
                        const lx = Math.cos(a) * lr;
                        const ly = Math.sin(a) * lr + 2.0;
                        const lbl = document.createElementNS(
                            "http://www.w3.org/2000/svg", "text");
                        lbl.setAttribute("x", lx.toFixed(2));
                        lbl.setAttribute("y", ly.toFixed(2));
                        lbl.textContent = txt;
                        labels.appendChild(lbl);
                    }
                }
            }
        }

        // Airspeed: 0–600 KIAS, major 100, minor 20
        _makeRoundDialTicks("airspeed-ticks", "airspeed-labels",
                            0, 600, -150, 150,
                            100, 20,
                            (v) => v % 100 === 0 ? String(v / 100) : null);
        // Altimeter: 0–30000 ft, major 5000, minor 1000 (label as ÷1000)
        _makeRoundDialTicks("altimeter-ticks", "altimeter-labels",
                            0, 30000, 0, 360,
                            5000, 1000,
                            (v) => v < 30000 && v % 5000 === 0
                                   ? String(v / 1000) : null);
        // VVI: -6000 to +6000, major 2000, minor 500
        _makeRoundDialTicks("vvi-ticks", "vvi-labels",
                            -6000, 6000, -150, 150,
                            2000, 500,
                            (v) => Math.abs(v) === 6000 || Math.abs(v) === 4000
                                   || Math.abs(v) === 2000 || v === 0
                                   ? String(v / 1000) : null);
        // G-meter: 0..10 G, major 1, minor 0.5
        _makeRoundDialTicks("g-ticks", "g-labels",
                            0, 10, -150, 150,
                            1, 0.5,
                            (v) => v % 1 === 0 ? String(v) : null);

        // HSI compass card: ticks every 10°, labels for cardinal/30° intervals
        const card = document.getElementById("hsi-card");
        if (card) {
            for (let d = 0; d < 360; d += 10) {
                const a = (d - 90) * Math.PI / 180;
                const isMajor = d % 30 === 0;
                const len = isMajor ? 6 : 3;
                const r1 = 42, r2 = 42 - len;
                const x1 = Math.cos(a) * r1, y1 = Math.sin(a) * r1;
                const x2 = Math.cos(a) * r2, y2 = Math.sin(a) * r2;
                const tk = document.createElementNS(
                    "http://www.w3.org/2000/svg", "line");
                tk.setAttribute("x1", x1.toFixed(2));
                tk.setAttribute("y1", y1.toFixed(2));
                tk.setAttribute("x2", x2.toFixed(2));
                tk.setAttribute("y2", y2.toFixed(2));
                tk.setAttribute("class", "hsi-card-tick" + (isMajor ? " major" : ""));
                card.appendChild(tk);
                if (d % 90 === 0 || d === 30 || d === 60 || d === 120
                    || d === 150 || d === 210 || d === 240 || d === 300
                    || d === 330) {
                    const lr = 32;
                    const lx = Math.cos(a) * lr;
                    const ly = Math.sin(a) * lr + 2.5;
                    const lbl = document.createElementNS(
                        "http://www.w3.org/2000/svg", "text");
                    lbl.setAttribute("x", lx.toFixed(2));
                    lbl.setAttribute("y", ly.toFixed(2));
                    let lbltxt;
                    if (d === 0) lbltxt = "N";
                    else if (d === 90) lbltxt = "E";
                    else if (d === 180) lbltxt = "S";
                    else if (d === 270) lbltxt = "W";
                    else lbltxt = String(d / 10);  // numeric (3=30°, 6=60°, ...)
                    lbl.textContent = lbltxt;
                    let cls = "hsi-card-label";
                    if (d === 0 || d === 90 || d === 180 || d === 270) {
                        cls += " hsi-card-cardinal";
                    }
                    lbl.setAttribute("class", cls);
                    card.appendChild(lbl);
                }
            }
        }
    })();

    // Afterburner exhaust glow — hot orange cone trailing aft of the
    // engine nozzle.
    const exhaustMat = new THREE.MeshBasicMaterial({
        color: 0xff8844, transparent: true, opacity: 0.85,
    });
    const exhaustGeom = new THREE.ConeGeometry(0.55, 4.0, 16, 1, true);
    const exhaust = new THREE.Mesh(exhaustGeom, exhaustMat);
    exhaust.name = "exhaust";
    exhaust.rotation.z = Math.PI / 2;
    exhaust.position.set(-9.0, 0, 0);
    exhaust.visible = !IS_B747;
    aircraft.add(exhaust);

    // ---- Damage state machinery ----
    // damage_state_history is sorted by time at export; binary-search
    // the latest entry with .time <= target.
    const dsh = log.damage_state_history;

    function damageStateAt(t) {
        if (!dsh || dsh.length === 0) return null;
        let lo = 0, hi = dsh.length - 1, best = 0;
        while (lo <= hi) {
            const mid = (lo + hi) >> 1;
            if (dsh[mid].time <= t) { best = mid; lo = mid + 1; }
            else { hi = mid - 1; }
        }
        return dsh[best].state;
    }

    // Cache original materials so we can restore on rewind. Each section
    // mesh's material is cloned so per-section opacity / colour edits do
    // not bleed across instances of _materialFor() that share types.
    const sectionMaterials = new Map();
    aircraft.children.forEach((obj) => {
        if (!obj.name || obj.name === "exhaust") return;
        // Find the mesh whose material we'll edit. If `obj` IS a mesh,
        // use its material directly. If it's a Group (hinge-pivoted
        // control surface), use the first descendant mesh.
        let meshNode = null;
        if (obj.isMesh && obj.material) {
            meshNode = obj;
        } else if (obj.isGroup || obj.children) {
            obj.traverse((c) => {
                if (!meshNode && c.isMesh && c.material) meshNode = c;
            });
        }
        if (!meshNode) return;
        meshNode.material = meshNode.material.clone();
        meshNode.material.transparent = true;
        sectionMaterials.set(obj.name, {
            color: meshNode.material.color.clone(),
            opacity: 1.0,
            mesh: meshNode,   // the actual mesh (may be a child of obj)
        });
    });

    // Late-registration hook for B-747 engines. The async GLB loader
    // re-parents the GLB Engine N nodes into the empty engine_<id>
    // placeholder groups; once that finishes, this function walks each
    // engine subtree, clones the mesh material(s), and registers them
    // in sectionMaterials so applyDamageState() can tint them per
    // engines_mu.
    function _registerB747EngineMaterials() {
        for (let eid = 1; eid <= 4; eid++) {
            const name = "engine_" + eid;
            if (sectionMaterials.has(name)) continue;
            const grp = aircraft.getObjectByName(name);
            if (!grp) continue;
            let meshNode = null;
            grp.traverse((c) => {
                if (!meshNode && c.isMesh && c.material && !c.isSprite) {
                    meshNode = c;
                }
            });
            if (!meshNode) continue;
            meshNode.material = meshNode.material.clone();
            meshNode.material.transparent = true;
            sectionMaterials.set(name, {
                color: meshNode.material.color.clone(),
                opacity: 1.0,
                mesh: meshNode,
            });
        }
    }

    const HEALTHY_COLOR = new THREE.Color(0xffffff);  // not used directly;
                                                       // we lerp from base
    const DAMAGE_RED = new THREE.Color(0xc0392b);
    const JAM_YELLOW = new THREE.Color(0xf1c40f);
    const ENGINE_AMBER = new THREE.Color(0xff8c2a);

    // Per-section damage animation state.
    //   when:  sim time at which the section first became damaged
    //   side:  "left"/"right"/"center" — drives the breakaway direction
    //   pos0:  original local position (always 0,0,0 for our static meshes)
    //   rot0:  original local rotation
    // When loss_fraction goes from 0 → >0 between frames, an entry is
    // created. setFrame() then calls advanceDamageAnimations() which
    // tweens position / rotation over BREAKAWAY_DURATION seconds.
    const damageAnim = new Map();
    const BREAKAWAY_DURATION = 0.8;  // seconds (sim time)

    function _sectionSign(name) {
        // -1 for left, +1 for right, 0 for centre. Used as breakaway
        // direction along the aircraft's local +z axis (right wing).
        if (name.startsWith("left_") || name.endsWith("_left")) return -1;
        if (name.startsWith("right_") || name.endsWith("_right")) return +1;
        return 0;
    }

    function _resetSection(mesh) {
        mesh.position.set(0, 0, 0);
        mesh.rotation.set(0, 0, 0);
        // Damage-tracked meshes have matrixAutoUpdate=false (set after
        // buildAircraft), so we need to recompute the local matrix
        // explicitly after mutating position / rotation.
        mesh.updateMatrix();
    }

    function advanceDamageAnimations(currentTime, lossMap) {
        // Sections currently animating
        for (const [name, anim] of damageAnim.entries()) {
            const mesh = aircraft.getObjectByName(name);
            if (!mesh) continue;

            const dt = currentTime - anim.when;
            if (dt < 0) {
                // Rewound past event: snap back, drop animation.
                _resetSection(mesh);
                damageAnim.delete(name);
                continue;
            }
            const phase = Math.min(dt / BREAKAWAY_DURATION, 1.0);
            // Translate outward (sideways) and downward in aircraft local
            // frame (its local +z = right; -y = down).
            mesh.position.set(
                -2.0 * phase,                 // drift backward
                -3.0 * phase,                 // fall (local -y = down)
                anim.sign * 6.0 * phase,      // outward
            );
            // Tumble: roll about local x and yaw about local y. Add to
            // anim.rotBase (the deflection at the moment damage fired)
            // so we don't clobber control-surface deflections from the
            // trajectory.
            mesh.rotation.set(
                anim.rotBase.x + anim.sign * phase * 1.4,
                anim.rotBase.y + phase * 0.8,
                anim.rotBase.z + phase * 1.2,
            );
            // matrixAutoUpdate is off on this mesh — recompute manually
            mesh.updateMatrix();
        }

        // Detect new damage transitions and start animations.
        for (const name of sectionMaterials.keys()) {
            const f = lossMap[name] || 0;
            if (f > 0 && !damageAnim.has(name)) {
                const mesh = aircraft.getObjectByName(name);
                damageAnim.set(name, {
                    when: currentTime,
                    sign: _sectionSign(name),
                    // Capture rotation at event time so the breakaway
                    // animation adds to it (preserves any active control-
                    // surface deflection).
                    rotBase: mesh
                        ? mesh.rotation.clone()
                        : new THREE.Euler(0, 0, 0),
                });
            }
        }

        // Detect heal-back (rewind): if a section is now at f=0 but had
        // an animation, reset it.
        for (const name of [...damageAnim.keys()]) {
            const f = lossMap[name] || 0;
            if (f === 0) {
                const mesh = aircraft.getObjectByName(name);
                if (mesh) _resetSection(mesh);
                damageAnim.delete(name);
            }
        }
    }

    function applyDamageState(state, currentTime) {
        const lossMap = (state && state.section_loss) || {};
        advanceDamageAnimations(currentTime, lossMap);
        if (!state) {
            // Reset all sections to their original material
            for (const [name, ref] of sectionMaterials.entries()) {
                const m = aircraft.getObjectByName(name);
                if (!m) continue;
                m.visible = true;
                ref.mesh.material.color.copy(ref.color);
                ref.mesh.material.opacity = ref.opacity;
                ref.mesh.material.emissive = new THREE.Color(0x000000);
            }
            exhaust.visible = !IS_B747;
            exhaust.material.opacity = 0.7;
            exhaust.scale.set(1, 1, 1);
            for (let eid = 1; eid <= 4; eid++) {
                const plume = aircraft.getObjectByName("engine_" + eid + "_exhaust");
                if (plume) {
                    plume.visible = IS_B747;
                    plume.material.opacity = 0.35;
                    plume.scale.set(1, 1, 1);
                }
                const smoke = aircraft.getObjectByName("engine_" + eid + "_smoke");
                if (smoke) {
                    smoke.visible = false;
                    smoke.material.opacity = 0.0;
                }
            }
            return;
        }

        // Section loss → red tint + fade
        // lossMap is already set above via (state && state.section_loss) || {}
        for (const [name, ref] of sectionMaterials.entries()) {
            const m = aircraft.getObjectByName(name);
            if (!m) continue;
            const f = lossMap[name] || 0.0;
            if (f <= 0) {
                m.visible = true;
                ref.mesh.material.color.copy(ref.color);
                ref.mesh.material.opacity = ref.opacity;
                ref.mesh.material.emissive = new THREE.Color(0x000000);
            } else if (f >= 1) {
                m.visible = false;
            } else {
                m.visible = true;
                // Lerp colour toward red, opacity toward 0
                ref.mesh.material.color.copy(ref.color).lerp(DAMAGE_RED, f);
                ref.mesh.material.opacity = (1 - f) * ref.opacity;
                ref.mesh.material.emissive = new THREE.Color(0x000000);
            }
        }

        // Control failures → yellow emissive outline
        const failures = state.control_failures || {};
        for (const surface in failures) {
            const ref = sectionMaterials.get(surface);
            if (!ref) continue;
            const failure = failures[surface];
            if (failure.mode === "healthy") continue;
            ref.mesh.material.emissive = JAM_YELLOW.clone().multiplyScalar(0.6);
        }

        // Engine state → exhaust intensity / visibility
        const engine = state.engine || { thrust_factor: 1.0, hard_failure: false };
        if (engine.hard_failure) {
            exhaust.visible = false;
        } else {
            exhaust.visible = !IS_B747;
            const tf = Math.max(0.0, Math.min(1.0, engine.thrust_factor));
            exhaust.material.opacity = 0.2 + 0.5 * tf;
            exhaust.scale.set(tf, 1, 1);
        }

        // B-747 engine-out state uses per-engine multipliers rather than
        // the F-16 single-engine {thrust_factor, hard_failure} payload.
        const enginesMu = state.engines_mu || null;
        if (enginesMu) {
            for (let eid = 1; eid <= 4; eid++) {
                const key = String(eid);
                const mu = Math.max(0.0, Math.min(1.0,
                    enginesMu[key] ?? enginesMu[eid] ?? 1.0));
                const ref = sectionMaterials.get("engine_" + eid);
                if (ref) {
                    ref.mesh.material.color.copy(ref.color).lerp(DAMAGE_RED, 1.0 - mu);
                    ref.mesh.material.emissive = ENGINE_AMBER.clone()
                        .multiplyScalar(mu < 0.99 ? 0.7 * (1.0 - mu) : 0.0);
                }
                const plume = aircraft.getObjectByName("engine_" + eid + "_exhaust");
                if (plume) {
                    plume.visible = mu > 0.02;
                    plume.material.opacity = 0.10 + 0.35 * mu;
                    plume.scale.set(Math.max(0.15, mu), 1, 1);
                }
                // Failure marker — small black dot at the failed engine.
                // mu=1.0  → hidden; mu<0.98 → solid black dot, fixed size.
                const smoke = aircraft.getObjectByName("engine_" + eid + "_smoke");
                if (smoke) {
                    const dmg = 1.0 - mu;
                    if (dmg > 0.02) {
                        smoke.visible = true;
                        smoke.material.opacity = 1.0;
                        smoke.scale.set(2.5, 2.5, 1.0);
                    } else {
                        smoke.visible = false;
                        smoke.material.opacity = 0.0;
                    }
                }
            }
        }
    }

    // ---- Trajectory trail ----
    // Volumetric tube along the path. CatmullRomCurve3 + TubeGeometry
    // gives a smooth, visible 3D ribbon (THREE.Line's `linewidth` is
    // ignored by most browsers, hence the rebuild-as-tube approach).
    //
    // Decimate the path so the tube doesn't blow up to ~6000 spline
    // points on long episodes; one point every TRAIL_SAMPLE_STRIDE
    // frames (plus the latest frame, always) keeps the curve smooth.
    const TRAIL_SAMPLE_STRIDE = 5;
    const TRAIL_RADIUS = IS_B747 ? 6.0 : 0.6;
    const TRAIL_RADIAL_SEGMENTS = 6;
    const trailMat = new THREE.MeshBasicMaterial({
        color: 0x4a90e2, transparent: true, opacity: 0.65,
    });
    let trailMesh = null;

    // Cache the path-point three.Vector3s so we don't reallocate each
    // frame; they're populated lazily on first reference.
    const trailPathCache = traj.position.map(p => new THREE.Vector3(p[0], -p[2], p[1]));

    // Track which frame the trail was last rebuilt at so we don't pay
    // the geometry cost every single tick at high FPS.
    let trailLastBuiltAt = -1;
    const TRAIL_REBUILD_EVERY = 5;  // frames

    function updateTrail(idx) {
        // Only rebuild when the trail has grown enough OR we rewound.
        if (idx === trailLastBuiltAt) return;
        if (idx > trailLastBuiltAt
            && idx - trailLastBuiltAt < TRAIL_REBUILD_EVERY
            && idx < traj.time.length - 1) {
            return;
        }
        trailLastBuiltAt = idx;

        // Build sampled point list: every TRAIL_SAMPLE_STRIDE-th from
        // the start up to idx, plus idx itself.
        const points = [];
        for (let k = 0; k <= idx; k += TRAIL_SAMPLE_STRIDE) {
            points.push(trailPathCache[k]);
        }
        if (points.length < 2 || points[points.length - 1] !== trailPathCache[idx]) {
            points.push(trailPathCache[idx]);
        }
        if (points.length < 2) return;  // nothing to render yet

        const curve = new THREE.CatmullRomCurve3(points);
        const tubeSegments = Math.max(8, points.length - 1);
        const tubeGeom = new THREE.TubeGeometry(
            curve, tubeSegments, TRAIL_RADIUS, TRAIL_RADIAL_SEGMENTS, false,
        );

        if (trailMesh) {
            scene.remove(trailMesh);
            trailMesh.geometry.dispose();
        }
        trailMesh = new THREE.Mesh(tubeGeom, trailMat);
        scene.add(trailMesh);
    }

    // ---- Time-series charts panel ----
    // Each spec produces one stacked line chart. `extract` reads a value
    // per frame from the trajectory; `unit` is the suffix on the live
    // readout. `series` is filled in below.
    const RAD2DEG = 180 / Math.PI;
    const chartSpecs = [
        { key: "V",     label: "AIRSPEED",  unit: "m/s",
          extract: (i) => traj.airspeed_mps
              ? traj.airspeed_mps[i]
              : (log.metadata && log.metadata.airspeed) || 0,
          fmt: (v) => v.toFixed(1) },
        { key: "h",     label: "ALTITUDE",  unit: "m",
          extract: (i) => traj.altitude_m
              ? traj.altitude_m[i]
              : ((log.metadata.params || {}).Oy || 0) - traj.position[i][2],
          fmt: (v) => v.toFixed(0) },
        { key: "theta", label: "PITCH θ",   unit: "°",
          extract: (i) => traj.attitude[i][1] * RAD2DEG,
          fmt: (v) => v.toFixed(2) },
        { key: "roll",  label: "ROLL γ",    unit: "°",
          extract: (i) => traj.attitude[i][0] * RAD2DEG,
          fmt: (v) => v.toFixed(2) },
        { key: "yaw",   label: "YAW ψ",     unit: "°",
          extract: (i) => traj.attitude[i][2] * RAD2DEG,
          fmt: (v) => v.toFixed(2) },
        { key: "alpha", label: "AOA α",     unit: "°",
          extract: (i) => traj.alpha[i] * RAD2DEG,
          fmt: (v) => v.toFixed(2) },
        { key: "beta",  label: "SIDESLIP β", unit: "°",
          extract: (i) => traj.beta[i] * RAD2DEG,
          fmt: (v) => v.toFixed(2) },
        { key: "wz",    label: "PITCH RATE ωz", unit: "°/s",
          extract: (i) => traj.wz[i] * RAD2DEG,
          fmt: (v) => v.toFixed(2) },
        { key: "stab",  label: "ELEVATOR (stab)", unit: "°",
          extract: (i) => traj.stab[i] * RAD2DEG,
          fmt: (v) => v.toFixed(2) },
        { key: "ail",   label: "AILERON",   unit: "°",
          extract: (i) => traj.ail[i] * RAD2DEG,
          fmt: (v) => v.toFixed(2) },
        { key: "dir",   label: "RUDDER",    unit: "°",
          extract: (i) => traj.dir[i] * RAD2DEG,
          fmt: (v) => v.toFixed(2) },
    ];

    // Chart geometry (in viewBox units, matched to the SVG aspect via
    // preserveAspectRatio="none"). The SVG itself stretches to fill the
    // panel width and the height set by --chart-height (S/M/L preset).
    const CHART_W = 340, CHART_H = 56;
    const CHART_PAD_L = 4, CHART_PAD_R = 4;
    const CHART_PAD_T = 14, CHART_PAD_B = 4;
    const CHART_HEIGHTS = { S: 40, M: 56, L: 90 };
    const CHARTS_STORAGE_KEY = "f16-viewer-charts-v1";

    function _buildChart(spec) {
        // Sample the series — full resolution is 6000 pts on a 60 s run,
        // far more than the chart pixel width. Decimate to ≈340 points.
        const stride = Math.max(1, Math.floor(T / CHART_W));
        const ys = [];
        const xs = [];
        for (let i = 0; i < T; i += stride) {
            ys.push(spec.extract(i));
            xs.push(i);
        }
        if (xs[xs.length - 1] !== T - 1) {
            ys.push(spec.extract(T - 1));
            xs.push(T - 1);
        }
        spec._values = ys;
        spec._indices = xs;

        // Optional reference/commanded signal — overlaid as a dashed line
        // and used to widen the y-range so both signals stay on screen.
        const refsAll = (traj.references || {});
        const refSeq = refsAll[spec.key];
        const hasRef = Array.isArray(refSeq) && refSeq.length > 0;
        const refYs = [];
        if (hasRef) {
            for (const i of xs) refYs.push(refSeq[i]);
        }
        spec._hasRef = hasRef;

        let yMin = Infinity, yMax = -Infinity;
        const _track = (v) => {
            if (Number.isFinite(v)) {
                if (v < yMin) yMin = v;
                if (v > yMax) yMax = v;
            }
        };
        for (const v of ys) _track(v);
        for (const v of refYs) _track(v);
        if (!Number.isFinite(yMin) || yMin === yMax) {
            // Constant signal — pad so the line is visible.
            const c = Number.isFinite(yMin) ? yMin : 0;
            yMin = c - 1; yMax = c + 1;
        }
        const yPad = (yMax - yMin) * 0.08;
        yMin -= yPad; yMax += yPad;
        spec._yMin = yMin;
        spec._yMax = yMax;

        const plotW = CHART_W - CHART_PAD_L - CHART_PAD_R;
        const plotH = CHART_H - CHART_PAD_T - CHART_PAD_B;
        const yScale = (v) =>
            CHART_PAD_T + plotH * (1 - (v - yMin) / (yMax - yMin));
        const xScale = (i) =>
            CHART_PAD_L + plotW * (i / Math.max(1, T - 1));

        const _pathFromYs = (yArr) => {
            let p = "";
            for (let k = 0; k < yArr.length; ++k) {
                const x = xScale(xs[k]);
                const y = yScale(yArr[k]);
                p += (k === 0 ? "M" : "L") + x.toFixed(1) + "," + y.toFixed(1);
            }
            return p;
        };
        const path = _pathFromYs(ys);
        const refPath = hasRef ? _pathFromYs(refYs) : "";
        spec._xScale = xScale;
        spec._yScale = yScale;
        spec._plotTop = CHART_PAD_T;
        spec._plotBottom = CHART_PAD_T + plotH;

        // Build SVG. The reference path (if any) sits beneath the live
        // line so the actual signal stays on top. A second small text
        // shows the current ref value next to the live readout.
        const wrap = document.createElement("div");
        wrap.className = "chart";
        const refMarkup = hasRef
            ? `<path class="chart-ref-line" d="${refPath}"/>
               <text class="chart-ref-value" id="chart-ref-${spec.key}"
                     x="${CHART_W - CHART_PAD_R - 2}" y="20">--</text>`
            : "";
        wrap.innerHTML =
            `<svg viewBox="0 0 ${CHART_W} ${CHART_H}" preserveAspectRatio="none">
                <line class="chart-axis" x1="${CHART_PAD_L}" y1="${spec._plotBottom}"
                      x2="${CHART_W - CHART_PAD_R}" y2="${spec._plotBottom}"/>
                <text class="chart-label" x="${CHART_PAD_L + 2}" y="10">${spec.label}</text>
                <text class="chart-value" id="chart-val-${spec.key}"
                      x="${CHART_W - CHART_PAD_R - 2}" y="10">--</text>
                <text class="chart-extreme" x="${CHART_PAD_L + 2}" y="${spec._plotBottom - 1}">${spec.fmt(yMin)}</text>
                <text class="chart-extreme" x="${CHART_PAD_L + 2}" y="${CHART_PAD_T + 7}">${spec.fmt(yMax)}</text>
                ${refMarkup}
                <path class="chart-line" d="${path}"/>
                <line class="chart-cursor" id="chart-cur-${spec.key}"
                      x1="${CHART_PAD_L}" y1="${spec._plotTop}"
                      x2="${CHART_PAD_L}" y2="${spec._plotBottom}"/>
            </svg>`;
        return wrap;
    }

    function _loadChartsState() {
        try {
            const raw = localStorage.getItem(CHARTS_STORAGE_KEY);
            if (raw) return JSON.parse(raw);
        } catch (_) { /* localStorage may be unavailable in private mode */ }
        return {};
    }
    function _saveChartsState(state) {
        try {
            localStorage.setItem(CHARTS_STORAGE_KEY, JSON.stringify(state));
        } catch (_) {}
    }
    const chartsState = Object.assign(
        { width: 340, size: "M", hidden: [] },
        _loadChartsState(),
    );

    function _applyChartHeight(size) {
        const h = CHART_HEIGHTS[size] || CHART_HEIGHTS.M;
        const panel = document.getElementById("charts-panel");
        if (panel) panel.style.setProperty("--chart-height", h + "px");
        chartsState.size = size;
        _saveChartsState(chartsState);
        // Refresh active class on size buttons
        for (const k of Object.keys(CHART_HEIGHTS)) {
            const btn = document.getElementById("charts-size-" + k);
            if (btn) btn.classList.toggle("active", k === size);
        }
    }

    function _setChartHidden(key, hidden) {
        const wrap = document.getElementById("chart-wrap-" + key);
        if (wrap) wrap.classList.toggle("hidden", hidden);
        const idx = chartsState.hidden.indexOf(key);
        if (hidden && idx === -1) chartsState.hidden.push(key);
        if (!hidden && idx !== -1) chartsState.hidden.splice(idx, 1);
        _saveChartsState(chartsState);
        // Sync menu checkbox
        const cb = document.getElementById("chart-cb-" + key);
        if (cb) cb.checked = !hidden;
    }

    function _initChartsPanel() {
        const panel = document.getElementById("charts-panel");
        if (!panel) return;
        panel.style.display = "none";
        panel.style.width = chartsState.width + "px";

        // Resize handle (drag the left edge to widen / narrow the panel)
        const handle = document.createElement("div");
        handle.className = "charts-resize-handle";
        handle.title = "Drag to resize";
        panel.appendChild(handle);

        let dragStartX = 0, dragStartW = 0;
        const onMove = (e) => {
            // Drag left grows the panel (it's anchored to right: 12px).
            const dx = dragStartX - e.clientX;
            const w = Math.max(220, Math.min(window.innerWidth * 0.8,
                                             dragStartW + dx));
            panel.style.width = w + "px";
            chartsState.width = w;
        };
        const onUp = () => {
            window.removeEventListener("mousemove", onMove);
            window.removeEventListener("mouseup", onUp);
            panel.classList.remove("resizing");
            _saveChartsState(chartsState);
        };
        handle.addEventListener("mousedown", (e) => {
            dragStartX = e.clientX;
            dragStartW = panel.getBoundingClientRect().width;
            panel.classList.add("resizing");
            window.addEventListener("mousemove", onMove);
            window.addEventListener("mouseup", onUp);
            e.preventDefault();
        });

        // Title bar with size buttons + settings menu toggle
        const title = document.createElement("div");
        title.className = "charts-title";
        title.innerHTML = `
            <span class="charts-title-text">Flight parameters</span>
            <span class="charts-size-group">
                <button class="charts-size-btn" id="charts-size-S" title="Small">S</button>
                <button class="charts-size-btn" id="charts-size-M" title="Medium">M</button>
                <button class="charts-size-btn" id="charts-size-L" title="Large">L</button>
            </span>
            <button class="charts-menu-btn" id="charts-menu-btn" title="Show/hide charts">⋯</button>
        `;
        panel.appendChild(title);

        for (const k of Object.keys(CHART_HEIGHTS)) {
            const b = title.querySelector("#charts-size-" + k);
            if (b) b.addEventListener("click", () => _applyChartHeight(k));
        }

        // Visibility menu (one checkbox per chart)
        const menu = document.createElement("div");
        menu.className = "charts-menu";
        menu.id = "charts-menu";
        for (const spec of chartSpecs) {
            const lbl = document.createElement("label");
            lbl.innerHTML =
                `<input type="checkbox" id="chart-cb-${spec.key}"
                        ${chartsState.hidden.indexOf(spec.key) === -1 ? "checked" : ""}>
                 <span>${spec.label}</span>`;
            const cb = lbl.querySelector("input");
            cb.addEventListener("change", () =>
                _setChartHidden(spec.key, !cb.checked));
            menu.appendChild(lbl);
        }
        panel.appendChild(menu);

        const menuBtn = title.querySelector("#charts-menu-btn");
        if (menuBtn) {
            menuBtn.addEventListener("click", () =>
                menu.classList.toggle("open"));
        }

        // Build all charts
        for (const spec of chartSpecs) {
            const wrap = _buildChart(spec);
            wrap.id = "chart-wrap-" + spec.key;
            // Per-chart hide button (× in the corner, visible on hover)
            const hide = document.createElement("button");
            hide.className = "chart-hide-btn";
            hide.title = "Hide this chart";
            hide.textContent = "×";
            hide.addEventListener("click", () =>
                _setChartHidden(spec.key, true));
            wrap.appendChild(hide);
            if (chartsState.hidden.indexOf(spec.key) !== -1) {
                wrap.classList.add("hidden");
            }
            panel.appendChild(wrap);
        }

        // Apply persisted size last so the CSS variable is set after all
        // chart SVGs exist.
        _applyChartHeight(chartsState.size);
    }

    function _updateCharts(idx) {
        const panel = document.getElementById("charts-panel");
        if (!panel || panel.style.display === "none") return;
        const refs = traj.references || {};
        for (const spec of chartSpecs) {
            if (chartsState.hidden.indexOf(spec.key) !== -1) continue;
            const cur = document.getElementById("chart-cur-" + spec.key);
            const val = document.getElementById("chart-val-" + spec.key);
            if (cur) {
                const x = spec._xScale(idx);
                cur.setAttribute("x1", x.toFixed(1));
                cur.setAttribute("x2", x.toFixed(1));
            }
            if (val) {
                const v = spec.extract(idx);
                val.textContent = spec.fmt(v) + " " + spec.unit;
            }
            if (spec._hasRef) {
                const refEl = document.getElementById("chart-ref-" + spec.key);
                if (refEl) {
                    const r = refs[spec.key][idx];
                    refEl.textContent = "ref " + spec.fmt(r);
                }
            }
        }
    }

    _initChartsPanel();

    // ---- Animation state ----
    const dt = log.metadata.dt;
    let frame = 0;
    let playing = true;
    let speed = IS_B747 ? 8.0 : 1.0;
    let lastTickMs = performance.now();

    function setFrame(idx) {
        idx = Math.max(0, Math.min(T - 1, Math.floor(idx)));
        frame = idx;
        const pos = traj.position[idx];
        const att = traj.attitude[idx];
        // Three.js uses Y-up, right-handed; the env body frame is x-fwd,
        // y-right, z-down. Map: three.x = body.x, three.y = -body.z, three.z = body.y.
        aircraft.position.set(pos[0], -pos[2], pos[1]);
        // Attitude: (roll=gamma, pitch=theta, yaw=psi) about (x, y, z).
        // We'll apply roll-pitch-yaw via Euler with the same axis swap.
        aircraft.rotation.set(att[0], -att[2], att[1], "ZYX");

        // Update trail (TubeGeometry rebuild, throttled)
        updateTrail(idx);

        // Update parameter charts cursor + readouts
        _updateCharts(idx);

        // Apply control-surface deflections from the trajectory.
        // Stabilator and ailerons rotate about their hinge (lateral axis,
        // = aircraft's local +Z). Rudder rotates about its vertical hinge
        // (= aircraft's local -Y; we apply the sign to .rotation.y).
        // DEFLECTION_VISUAL_GAIN amplifies the mesh rotation for readability
        // without affecting the underlying physics values.
        const DEFLECTION_VISUAL_GAIN = 2.5;
        const stabDef = traj.stab[idx];
        const ailDef  = traj.ail[idx];
        const dirDef  = traj.dir[idx];
        const stabL = aircraft.getObjectByName("stab_left");
        const stabR = aircraft.getObjectByName("stab_right");
        if (stabL) stabL.rotation.z = stabDef * DEFLECTION_VISUAL_GAIN;
        if (stabR) stabR.rotation.z = stabDef * DEFLECTION_VISUAL_GAIN;
        const ailL = aircraft.getObjectByName("aileron_left");
        const ailR = aircraft.getObjectByName("aileron_right");
        if (ailL) ailL.rotation.z = -ailDef * DEFLECTION_VISUAL_GAIN;
        if (ailR) ailR.rotation.z = +ailDef * DEFLECTION_VISUAL_GAIN;
        const rud = aircraft.getObjectByName("rudder");
        if (rud) rud.rotation.y = -dirDef * DEFLECTION_VISUAL_GAIN;

        // HUD: surface deflections in degrees.
        const stabHud = document.getElementById("hud-stab");
        const ailHud  = document.getElementById("hud-ail");
        const dirHud  = document.getElementById("hud-dir");
        if (stabHud) stabHud.textContent =
            (stabDef * 180 / Math.PI).toFixed(1) + "°";
        if (ailHud)  ailHud.textContent  =
            (ailDef  * 180 / Math.PI).toFixed(1) + "°";
        if (dirHud)  dirHud.textContent  =
            (dirDef  * 180 / Math.PI).toFixed(1) + "°";

        // Apply damage state for this time
        applyDamageState(damageStateAt(traj.time[idx]), traj.time[idx]);

        // Camera tracks the aircraft each frame regardless of preset mode.
        updateCamera();

        // Update HUD
        document.getElementById("hud-time").textContent =
            traj.time[idx].toFixed(2) + " s";
        // Active damage events (events within 1.5 s of current frame)
        const eventsEl = document.getElementById("hud-events");
        if (eventsEl) {
            const t = traj.time[idx];
            const recent = (log.damage_events || []).filter(
                (e) => e.time <= t && t - e.time < 1.5,
            );
            eventsEl.textContent = recent.length
                ? recent.map((e) => e.label).join(", ")
                : "—";
        }
        document.getElementById("hud-alpha").textContent =
            (traj.alpha[idx] * 180 / Math.PI).toFixed(2) + "°";
        document.getElementById("hud-beta").textContent =
            (traj.beta[idx] * 180 / Math.PI).toFixed(2) + "°";
        document.getElementById("hud-wx").textContent =
            (traj.wx[idx] * 180 / Math.PI).toFixed(2) + "°/s";
        document.getElementById("hud-wz").textContent =
            (traj.wz[idx] * 180 / Math.PI).toFixed(2) + "°/s";

        // ---- Instrument-panel update ----
        const params = (log.metadata && log.metadata.params) || {};
        const airspeed_mps = params.V
            || (log.metadata && log.metadata.airspeed) || 0;
        const trimAlt_m = params.Oy || 0;
        const G_EARTH = params.g || 9.80665;

        const attHud   = traj.attitude[idx];
        const alphaRad = traj.alpha[idx];
        const betaRad  = traj.beta[idx];
        const wzRad    = traj.wz[idx];
        const rollDeg  = attHud[0] * 180 / Math.PI;
        const pitchDeg = attHud[1] * 180 / Math.PI;
        const yawDeg   = attHud[2] * 180 / Math.PI;
        const alphaDeg = alphaRad * 180 / Math.PI;

        // 1. ADI: bank rotation + pitch translation. ADI viewBox is
        // -50..+50; 1 unit ≈ 1° pitch (we scale 1.5x for readability).
        const ADI_PIXELS_PER_DEG = 1.6;
        const adiRotor = document.getElementById("adi-rotor");
        const adiPitch = document.getElementById("adi-pitch");
        if (adiRotor) {
            adiRotor.setAttribute(
                "transform", "rotate(" + (-rollDeg).toFixed(1) + ")",
            );
        }
        if (adiPitch) {
            const py = pitchDeg * ADI_PIXELS_PER_DEG;
            adiPitch.setAttribute(
                "transform", "translate(0 " + py.toFixed(2) + ")",
            );
        }

        // 2. Airspeed: needle 0..600 KIAS. Prefer the live airspeed from
        // the simulation if available (track_altitude=True envs export
        // traj.airspeed_mps); otherwise fall back to params.V (constant).
        const live_V = traj.airspeed_mps ? traj.airspeed_mps[idx] : airspeed_mps;
        const kias = live_V * 1.94384;
        _setNeedle("airspeed-needle", kias, 0, 600, -150, 150);
        _setText("airspeed-digital", String(Math.round(kias)));

        // 3. Altimeter: 0..30000 ft. Prefer simulation altitude when
        // available (track_altitude=True envs export traj.altitude_m);
        // otherwise compute kinematically from inertial position.
        const posHud = traj.position[idx];
        const alt_m = traj.altitude_m
            ? traj.altitude_m[idx]
            : trimAlt_m + (-posHud[2]);
        const alt_ft = alt_m * 3.28084;
        _setNeedle("altimeter-needle", alt_ft, 0, 30000, 0, 360);
        _setText("altimeter-digital", String(Math.round(alt_ft)));

        // 4. HSI: rotate compass card so the current heading sits at top.
        let hdg = yawDeg % 360;
        if (hdg < 0) hdg += 360;
        const hsiRotor = document.getElementById("hsi-rotor");
        if (hsiRotor) {
            hsiRotor.setAttribute(
                "transform", "rotate(" + (-hdg).toFixed(1) + ")",
            );
        }
        _setText("hsi-digital", String(Math.round(hdg)).padStart(3, "0"));

        // 5. VVI: vertical-speed needle ±6000 ft/min. Compute from a
        // 0.5 s lookback over altitude.
        const VSI_LOOKBACK = 50;
        const j = Math.max(0, idx - VSI_LOOKBACK);
        const altPrev_m = traj.altitude_m
            ? traj.altitude_m[j]
            : trimAlt_m + (-traj.position[j][2]);
        const dT_s = (idx - j) * dt || dt;
        const vvi_fpm = (alt_m - altPrev_m) * 3.28084 / dT_s * 60.0;
        // Map -6000..+6000 → -150..+150 (0 at top of dial).
        _setNeedle("vvi-needle", vvi_fpm, -6000, 6000, -150, 150);
        const vviSign = vvi_fpm >= 0 ? "+" : "";
        _setText("vvi-digital", vviSign + Math.round(vvi_fpm));

        // 6. AOA strip: 25° → +30, -25° → -30 (within the 80px tall strip).
        const aoaPointer = document.getElementById("aoa-pointer");
        if (aoaPointer) {
            const yPx = -alphaDeg / 25 * 30;   // clamp implicit via clip
            const ypc = Math.max(-30, Math.min(30, yPx));
            aoaPointer.setAttribute(
                "transform", "translate(0 " + ypc.toFixed(1) + ")",
            );
        }
        _setText("aoa-digital", alphaDeg.toFixed(1));

        // 7. G-meter: 0..10 G, 0 at bottom-left (-150°), 10 at bottom-right (+150°).
        const gLoad = 1.0 + Math.abs(wzRad * live_V) / G_EARTH;
        _setNeedle("g-needle", gLoad, 0, 10, -150, 150);
        _setText("g-digital", gLoad.toFixed(1));

        // 8. Master caution lamp on damage
        const ds = damageStateAt(traj.time[idx]);
        const lamp = document.getElementById("caution-lamp");
        if (lamp) {
            let any = false;
            if (ds && ds.section_loss) {
                for (const k in ds.section_loss) {
                    if (ds.section_loss[k] > 0) { any = true; break; }
                }
            }
            if (!any && ds && ds.engine && ds.engine.hard_failure) {
                any = true;
            }
            if (!any && ds && ds.control_failures) {
                for (const k in ds.control_failures) {
                    const cf = ds.control_failures[k];
                    if (cf && cf.mode && cf.mode !== "healthy") {
                        any = true; break;
                    }
                }
            }
            lamp.classList.toggle("active", any);
        }

        document.getElementById("timeline").value = idx;
    }

    // ---- UI wiring ----
    const timeline = document.getElementById("timeline");
    timeline.min = 0;
    timeline.max = T - 1;
    timeline.value = 0;
    timeline.addEventListener("input", (e) => {
        playing = false;
        document.getElementById("btn-play").textContent = "Play";
        setFrame(parseInt(e.target.value, 10));
    });

    const btnPlay = document.getElementById("btn-play");
    btnPlay.addEventListener("click", () => {
        if (!playing && frame >= T - 1) setFrame(0);
        playing = !playing;
        btnPlay.textContent = playing ? "Pause" : "Play";
        if (playing) lastTickMs = performance.now();
    });

    const speedSelect = document.getElementById("speed");
    speedSelect.value = String(speed);
    speedSelect.addEventListener("change", () => {
        speed = parseFloat(speedSelect.value);
    });

    document.getElementById("btn-cam-3d").addEventListener("click", preset3DCamera);
    document.getElementById("btn-cam-top").addEventListener("click", presetTopDown);
    document.getElementById("btn-cam-left").addEventListener("click", presetLeftSide);
    document.getElementById("btn-cam-right").addEventListener("click", presetRightSide);
    const fullscreenBtn = document.getElementById("btn-fullscreen");
    if (fullscreenBtn) {
        fullscreenBtn.classList.add("secondary");
        fullscreenBtn.addEventListener("click", async () => {
            const target = document.documentElement;
            try {
                if (!document.fullscreenElement) await target.requestFullscreen();
                else await document.exitFullscreen();
            } catch (_) {
                // Fullscreen may be blocked inside some notebook iframes.
            }
        });
    }

    window.addEventListener("resize", () => {
        camera.aspect = sceneEl.clientWidth / sceneEl.clientHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(sceneEl.clientWidth, sceneEl.clientHeight);
    });

    // ---- Keyboard shortcuts ----
    // Fires when the viewport (or anything not an input/select) has
    // focus. Avoid hijacking text-input keys.
    function _isTextInputTarget(t) {
        if (!t) return false;
        const tag = (t.tagName || "").toUpperCase();
        return tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT";
    }

    const SPEED_STEPS = [0.25, 0.5, 1, 2, 4, 8, 16, 32];
    function _changeSpeed(direction) {
        const i = SPEED_STEPS.indexOf(speed);
        const ni = Math.max(0, Math.min(SPEED_STEPS.length - 1,
                                        (i < 0 ? 2 : i) + direction));
        speed = SPEED_STEPS[ni];
        speedSelect.value = String(speed);
    }

    function _scrubBy(seconds) {
        playing = false;
        btnPlay.textContent = "Play";
        const dframes = Math.round(seconds / dt);
        setFrame(frame + dframes);
    }

    document.addEventListener("keydown", (e) => {
        if (_isTextInputTarget(e.target)) return;
        switch (e.code) {
            case "Space":
                e.preventDefault();
                playing = !playing;
                btnPlay.textContent = playing ? "Pause" : "Play";
                if (playing) lastTickMs = performance.now();
                return;
            case "ArrowLeft":
                e.preventDefault();
                _scrubBy(e.shiftKey ? -5.0 : -0.5);
                return;
            case "ArrowRight":
                e.preventDefault();
                _scrubBy(e.shiftKey ? +5.0 : +0.5);
                return;
            case "Home":
                e.preventDefault();
                playing = false;
                btnPlay.textContent = "Play";
                setFrame(0);
                return;
            case "End":
                e.preventDefault();
                playing = false;
                btnPlay.textContent = "Play";
                setFrame(T - 1);
                return;
            case "Digit1": preset3DCamera(); return;
            case "Digit2": presetTopDown(); return;
            case "Digit3": presetLeftSide(); return;
            case "Digit4": presetRightSide(); return;
            case "Equal":
            case "NumpadAdd":
                _changeSpeed(+1);
                return;
            case "Minus":
            case "NumpadSubtract":
                _changeSpeed(-1);
                return;
            case "Slash":   // ? on most layouts (with shift)
            case "KeyH": {
                const help = document.getElementById("keyboard-help");
                if (help) {
                    help.style.display =
                        help.style.display === "none" ? "block" : "none";
                }
                return;
            }
            case "KeyC": {
                const panel = document.getElementById("charts-panel");
                if (panel) {
                    const visible = panel.style.display !== "none"
                        && panel.style.display !== "";
                    panel.style.display = visible ? "none" : "block";
                    if (!visible) _updateCharts(frame);
                }
                return;
            }
        }
    });

    window.addEventListener("focus", () => {
        lastTickMs = performance.now();
    });
    document.addEventListener("visibilitychange", () => {
        lastTickMs = performance.now();
    });

    // ---- Animation loop ----
    function animate() {
        requestAnimationFrame(animate);
        if (playing) {
            const now = performance.now();
            const elapsed = Math.min((now - lastTickMs) / 1000.0, 0.25);
            const advance = (elapsed / dt) * speed;
            const next = frame + advance;
            if (next >= T - 1) {
                setFrame(T - 1);
                playing = false;
                btnPlay.textContent = "Play";
            } else {
                setFrame(next);
            }
            lastTickMs = now;
        }
        controls.update();
        renderer.render(scene, camera);
    }

    setFrame(0);
    syncCameraButtons();
    animate();
})();
