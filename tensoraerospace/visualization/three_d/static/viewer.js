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

    const traj = log.trajectory;
    const T = traj.time.length;

    // ---- Scene setup ----
    const sceneEl = document.getElementById("scene");
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    // Cap DPR at 2 — going to 3+ on retina screens triples pixel count
    // for negligible visual gain on this geometry density (Three.js
    // best practice: render-pixel-ratio).
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(sceneEl.clientWidth, sceneEl.clientHeight);
    sceneEl.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0a0a14);

    // Soft ambient + directional lights (sun-like)
    scene.add(new THREE.AmbientLight(0x404060, 0.6));
    const sun = new THREE.DirectionalLight(0xfff0d0, 1.0);
    sun.position.set(40, 60, 40);
    scene.add(sun);

    // Ground grid (10 x 10 units, 100 m subdivisions)
    const grid = new THREE.GridHelper(2000, 40, 0x303048, 0x202030);
    grid.position.y = 0;
    scene.add(grid);

    // ---- Camera + controls ----
    const camera = new THREE.PerspectiveCamera(
        55, sceneEl.clientWidth / sceneEl.clientHeight, 0.1, 5000,
    );
    camera.position.set(20, 14, 20);
    camera.lookAt(0, 0, 0);

    const controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.minDistance = 5;
    controls.maxDistance = 1500;

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
    const FOLLOW_OFFSETS = {
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
    }

    function presetTopDown() {
        cameraMode = "top";
        controls.enabled = false;
        updateCamera();
    }

    function presetLeftSide() {
        cameraMode = "left";
        controls.enabled = false;
        updateCamera();
    }

    function presetRightSide() {
        cameraMode = "right";
        controls.enabled = false;
        updateCamera();
    }

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
        const cacheKey = [
            color, metalness, roughness, opacity, transparent, emissive,
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
            ...(emissive ? { emissive } : {}),
            ...cleanOpts,
        });
        if (!unique) _stdMatCache.set(cacheKey, mat);
        return mat;
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
        const fuse = new THREE.Group();
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

        // ---- CANOPY ----
        // Two-piece: forward windscreen wedge + rear bubble.
        // Windscreen: a slightly tilted half-ellipsoid section
        const windscreenGeom = new THREE.SphereGeometry(0.9, 16, 12, 0, Math.PI * 2, 0, Math.PI * 0.55);
        windscreenGeom.scale(1.0, 0.5, 0.72);
        const windscreen = new THREE.Mesh(
            windscreenGeom,
            _stdMat(0x1a2a38, { metalness: 0.75, roughness: 0.05,
                                 transparent: true, opacity: 0.82 }),
        );
        windscreen.rotation.z = 0.28;   // lean forward ~16°
        windscreen.position.set(3.5, 0.88, 0);
        group.add(windscreen);

        // Rear bubble: main F-16 frameless canopy dome
        const bubbleGeom = new THREE.SphereGeometry(0.92, 20, 14, 0, Math.PI * 2, 0, Math.PI * 0.6);
        bubbleGeom.scale(1.55, 0.60, 0.80);
        const bubble = new THREE.Mesh(
            bubbleGeom,
            _stdMat(0x1f2f3e, { metalness: 0.72, roughness: 0.06,
                                 transparent: true, opacity: 0.80 }),
        );
        bubble.name = "fuselage_canopy";
        bubble.position.set(2.0, 0.90, 0);
        group.add(bubble);

        // Canopy frame / sill — slim dark ring around the canopy base
        const sillGeom = new THREE.TorusGeometry(0.82, 0.055, 7, 28);
        const sill = new THREE.Mesh(
            sillGeom,
            _stdMat(0x151f28, { metalness: 0.45, roughness: 0.35 }),
        );
        sill.rotation.x = Math.PI / 2;
        sill.scale.set(1.52, 1.0, 0.70);
        sill.position.set(2.0, 0.60, 0);
        group.add(sill);

        // ---- INTAKE ----
        // Build as four trapezoidal panels forming a rectangular duct,
        // with a visible darker cavity at the intake face.
        // Intake in body coords: centre at bx ~ +1.8, by = 0, bz = +1.1
        // (body +z = down → three.js -y). Duct width ≈ 1.4 m, height ≈ 0.7 m.
        function _intakePanelGeom(corners4_body) {
            const c = corners4_body.map(([bx, by, bz]) => bodyToThree(bx, by, bz));
            return _quadGeom(c[0], c[1], c[2], c[3]);
        }

        const intakeGroup = new THREE.Group();
        intakeGroup.name = "fuselage_intake";

        // Duct corners in body space (bx_front, bx_back, width, top_z, bot_z)
        const iFront = +4.2, iBack = -0.2;
        const iHalfW = 0.72, iTop = +0.78, iBot = +1.48;
        // Four panels: left wall, right wall, top ceiling, bottom floor
        const iMat = _stdMat(F16_COLORS.intake, { metalness: 0.45 });
        const panels = [
            // left wall (by = -iHalfW face)
            [[iFront, -iHalfW, iTop], [iBack, -iHalfW, iTop],
             [iBack, -iHalfW, iBot], [iFront, -iHalfW, iBot]],
            // right wall
            [[iFront, +iHalfW, iTop], [iFront, +iHalfW, iBot],
             [iBack, +iHalfW, iBot], [iBack, +iHalfW, iTop]],
            // top ceiling
            [[iFront, -iHalfW, iTop], [iFront, +iHalfW, iTop],
             [iBack, +iHalfW, iTop], [iBack, -iHalfW, iTop]],
            // bottom floor
            [[iFront, -iHalfW, iBot], [iBack, -iHalfW, iBot],
             [iBack, +iHalfW, iBot], [iFront, +iHalfW, iBot]],
        ];
        for (const p of panels) {
            intakeGroup.add(new THREE.Mesh(_intakePanelGeom(p), iMat));
        }
        // Front face (intake lip) — darker recessed plane
        const lipGVerts = new Float32Array([
            ...bodyToThree(iFront, -iHalfW, iTop),
            ...bodyToThree(iFront, -iHalfW, iBot),
            ...bodyToThree(iFront, +iHalfW, iBot),
            ...bodyToThree(iFront, -iHalfW, iTop),
            ...bodyToThree(iFront, +iHalfW, iBot),
            ...bodyToThree(iFront, +iHalfW, iTop),
        ]);
        const lipGeom = new THREE.BufferGeometry();
        lipGeom.setAttribute("position", new THREE.BufferAttribute(lipGVerts, 3));
        lipGeom.computeVertexNormals();
        intakeGroup.add(new THREE.Mesh(
            lipGeom,
            _stdMat(0x0c1014, { metalness: 0.2, roughness: 0.9 }),
        ));
        // Intake splitter plate — small horizontal plate inside the duct
        // (boundary layer splitter, F-16 characteristic)
        const splitterVerts = new Float32Array([
            ...bodyToThree(iFront - 0.1, -iHalfW + 0.1, iTop + 0.22),
            ...bodyToThree(iBack + 0.3, -iHalfW + 0.1, iTop + 0.22),
            ...bodyToThree(iBack + 0.3, +iHalfW - 0.1, iTop + 0.22),
            ...bodyToThree(iFront - 0.1, -iHalfW + 0.1, iTop + 0.22),
            ...bodyToThree(iBack + 0.3, +iHalfW - 0.1, iTop + 0.22),
            ...bodyToThree(iFront - 0.1, +iHalfW - 0.1, iTop + 0.22),
        ]);
        const splitterGeom = new THREE.BufferGeometry();
        splitterGeom.setAttribute("position", new THREE.BufferAttribute(splitterVerts, 3));
        splitterGeom.computeVertexNormals();
        intakeGroup.add(new THREE.Mesh(
            splitterGeom,
            _stdMat(0x606870, { metalness: 0.5, roughness: 0.4 }),
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
        // A cylinder with nozzle petals (thin radial strips around the exit)
        // and a darker inner cone.
        const nozzleX = -7.9;
        const nozzleOuterR = 0.48, nozzleLen = 1.1;
        const nozzleCanGeom = new THREE.CylinderGeometry(
            nozzleOuterR, nozzleOuterR * 0.88, nozzleLen, 16, 1, true,
        );
        const nozzleCan = new THREE.Mesh(
            nozzleCanGeom,
            _stdMat(0x3a3a3a, { metalness: 0.72, roughness: 0.3 }),
        );
        nozzleCan.rotation.z = Math.PI / 2;
        nozzleCan.position.set(nozzleX - nozzleLen / 2, 0, 0);
        group.add(nozzleCan);

        // Nozzle petals: 10 thin dark fins around the exit ring
        const PETAL_COUNT = 10;
        for (let p = 0; p < PETAL_COUNT; p++) {
            const a = (p / PETAL_COUNT) * Math.PI * 2;
            const py = Math.cos(a) * nozzleOuterR * 0.88;
            const pz = Math.sin(a) * nozzleOuterR * 0.88;
            // Each petal is a tiny flat rectangle oriented radially
            const petalGeom = new THREE.BoxGeometry(0.55, 0.045, 0.13);
            const petal = new THREE.Mesh(
                petalGeom,
                _stdMat(0x252525, { metalness: 0.8, roughness: 0.25 }),
            );
            // Position & orient radially at nozzle exit
            const [px, ptY, ptZ] = bodyToThree(nozzleX - nozzleLen, py, -pz);
            petal.position.set(px, ptY, ptZ);
            petal.lookAt(
                px - 1,
                ptY,
                ptZ,
            );
            petal.rotateY(Math.atan2(pz, py));
            group.add(petal);
        }

        // Inner dark nozzle throat cone
        const throatGeom = new THREE.ConeGeometry(nozzleOuterR * 0.7, 0.5, 16, 1, false);
        const throat = new THREE.Mesh(
            throatGeom,
            _stdMat(0x111111, { metalness: 0.4, roughness: 0.8 }),
        );
        throat.rotation.z = Math.PI / 2;
        throat.position.set(nozzleX - nozzleLen + 0.05, 0, 0);
        group.add(throat);

        // ---- LERX (Leading Edge Root Extensions) — 3D thin wedge ----
        // Top surface (slightly above fuselage top), bottom surface (flush),
        // meeting at a sharp leading edge.
        const lerxRightShape = [
            [+3.0, +0.55],
            [+1.5, +0.95],
            [-0.4, +1.95],
            [+0.4, +1.95],
            [+1.5, +0.85],
        ];
        const lerxLeftShape = lerxRightShape.map(([x, y]) => [x, -y]);

        function _lerxMesh(poly, name) {
            // Top surface: bz = -0.12 (slightly above fuselage top)
            // Bottom surface: bz = +0.02 (flush with fuselage bottom)
            // LE knife-edge: where top and bottom meet at the outer edge.
            const top = poly.map(([bx, by]) => bodyToThree(bx, by, -0.14));
            const bot = poly.map(([bx, by]) => bodyToThree(bx, by, +0.03));
            // Fan triangulate both surfaces + stitch edges
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
            // Leading edge strip: pair each consecutive edge
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
            // Fin outline in body coords (outboard face):
            const outerPts = [
                bodyToThree(-5.5, sign * 0.38, +0.68),
                bodyToThree(-7.0, sign * 0.38, +0.68),
                bodyToThree(-6.3, sign * 0.72, +1.65),
            ];
            // Inner face (inboard) slightly inset
            const innerPts = [
                bodyToThree(-5.5, sign * 0.32, +0.72),
                bodyToThree(-7.0, sign * 0.32, +0.72),
                bodyToThree(-6.3, sign * 0.60, +1.60),
            ];
            const o = outerPts, inn = innerPts;
            const verts = new Float32Array([
                // outer face
                ...o[0], ...o[1], ...o[2],
                // inner face (reversed)
                ...inn[0], ...inn[2], ...inn[1],
                // edge 0-1
                ...o[0], ...inn[0], ...inn[1],
                ...o[0], ...inn[1], ...o[1],
                // edge 1-2
                ...o[1], ...inn[1], ...inn[2],
                ...o[1], ...inn[2], ...o[2],
                // edge 2-0
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

        // ---- PANEL LINES (thin dark cylinders along the fuselage) ----
        // A few axial seam lines to break up the shiny surface.
        const panelLineMat = _stdMat(0x7a8088, { metalness: 0.3, roughness: 0.7 });
        // Top spine seam from cockpit aft
        const spineGeom = new THREE.CylinderGeometry(0.018, 0.018, 6.5, 6);
        const spine = new THREE.Mesh(spineGeom, panelLineMat);
        spine.rotation.z = Math.PI / 2;
        spine.position.set(-2.5, 0.82, 0);
        group.add(spine);
        // Lower chin panel seam
        const chinGeom = new THREE.CylinderGeometry(0.016, 0.016, 3.5, 6);
        const chin = new THREE.Mesh(chinGeom, panelLineMat);
        chin.rotation.z = Math.PI / 2;
        chin.position.set(0.0, 0, -0.72);  // body y=0, bz=+0.72 → three.js y=-0.72
        // recalculate: bodyToThree(0, 0, 0.72) = [0, -0.72, 0]
        chin.position.set(0.0, -0.72, 0);
        group.add(chin);

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
        // The missile body is ~2.85 m long, 0.127 m diameter.
        // 4 mid-body delta fins + 4 forward canard fins.
        const sign = side === "left" ? -1 : +1;

        // Root group named as required (launcher_left / launcher_right)
        const launcherGroup = new THREE.Group();
        launcherGroup.name = name;

        // Launcher rail: thin cylinder below the missile
        const railGeom = new THREE.CylinderGeometry(0.045, 0.045, 2.6, 8);
        railGeom.rotateZ(Math.PI / 2);
        const rail = new THREE.Mesh(
            railGeom, _stdMat(0x707878, { metalness: 0.6, roughness: 0.35 }),
        );
        launcherGroup.add(rail);

        // AIM-9 body (cylinder)
        const bodyGeom = new THREE.CylinderGeometry(0.065, 0.065, 2.85, 14);
        bodyGeom.rotateZ(Math.PI / 2);
        const missileBody = new THREE.Mesh(
            bodyGeom, _stdMat(0xb8c0c8, { metalness: 0.5, roughness: 0.35 }),
        );
        missileBody.position.set(0, 0.115, 0);  // above rail
        launcherGroup.add(missileBody);

        // Seeker nose (cone) — IR seeker dome (slightly darker)
        const seekerGeom = new THREE.ConeGeometry(0.065, 0.22, 14);
        seekerGeom.rotateZ(-Math.PI / 2);
        const seeker = new THREE.Mesh(
            seekerGeom,
            _stdMat(0x505860, { metalness: 0.55, roughness: 0.2 }),
        );
        seeker.position.set(1.535, 0.115, 0);
        launcherGroup.add(seeker);

        // Tail nozzle cone
        const tailGeom = new THREE.ConeGeometry(0.065, 0.18, 14);
        tailGeom.rotateZ(Math.PI / 2);
        const tailCone = new THREE.Mesh(
            tailGeom, _stdMat(0x404040, { metalness: 0.6, roughness: 0.3 }),
        );
        tailCone.position.set(-1.515, 0.115, 0);
        launcherGroup.add(tailCone);

        // Helper: 4 cruciform fins on the AIM-9
        function _addCruciformFins(bxCentre, span, chord, thick, offsetY, finSign) {
            // 4 fins at 0°, 90°, 180°, 270° around missile axis.
            for (let k = 0; k < 4; k++) {
                const a = k * Math.PI / 2;
                // Fin: a thin box. Fins alternate horizontal/vertical pair.
                const finW = (k % 2 === 0) ? span : thick;
                const finH = (k % 2 === 0) ? thick : span;
                const finGeom = new THREE.BoxGeometry(chord, finH, finW);
                const fin = new THREE.Mesh(
                    finGeom,
                    _stdMat(0xa8b0b8, { metalness: 0.5, roughness: 0.35 }),
                );
                // k=0: horizontal (Z), k=1: vertical (Y), k=2: horiz, k=3: vert
                const fy = (k % 2 === 1) ? (k === 1 ? span * 0.5 : -span * 0.5) : 0;
                const fz = (k % 2 === 0) ? (k === 0 ? span * 0.5 : -span * 0.5) : 0;
                fin.position.set(bxCentre, offsetY + fy, fz * finSign);
                launcherGroup.add(fin);
            }
        }

        // Mid-body delta fins (larger, at ~ x = -0.5 from missile centre)
        _addCruciformFins(-0.5, 0.30, 0.45, 0.018, 0.115, sign);
        // Forward canard fins (smaller, near seeker, ~ x = +1.0)
        _addCruciformFins(+0.9, 0.14, 0.22, 0.014, 0.115, sign);

        // Position whole launcher group at wingtip LE
        const [tx, ty, tz] = bodyToThree(-1.85, sign * 4.62, -0.08);
        launcherGroup.position.set(tx, ty, tz);
        return launcherGroup;
    }

    function buildAircraft(geometry) {
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

        // Underwing pylons (static, no names matching protected list)
        // Two per side at mid-span, hanging below the wing surface.
        function _uwPylon(sign, pylonY) {
            const pylonMat = _stdMat(0x7a8288, { metalness: 0.5, roughness: 0.45 });
            // Pylon strut: flat box hanging below wing
            const strutGeom = new THREE.BoxGeometry(0.55, 0.38, 0.10);
            const strut = new THREE.Mesh(strutGeom, pylonMat);
            const [px, py, pz] = bodyToThree(-1.6, sign * pylonY, +0.22);
            strut.position.set(px, py, pz);
            // Small rail/shelf at bottom of strut
            const shelfGeom = new THREE.BoxGeometry(0.45, 0.065, 0.22);
            const shelf = new THREE.Mesh(shelfGeom, pylonMat);
            shelf.position.set(px, py - 0.22, pz);
            const pGroup = new THREE.Group();
            pGroup.add(strut);
            pGroup.add(shelf);
            return pGroup;
        }
        aircraft.add(_uwPylon(+1, 2.6));  // right inboard
        aircraft.add(_uwPylon(+1, 3.3));  // right outboard
        aircraft.add(_uwPylon(-1, 2.6));  // left inboard
        aircraft.add(_uwPylon(-1, 3.3));  // left outboard

        return aircraft;
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

    const HEALTHY_COLOR = new THREE.Color(0xffffff);  // not used directly;
                                                       // we lerp from base
    const DAMAGE_RED = new THREE.Color(0xc0392b);
    const JAM_YELLOW = new THREE.Color(0xf1c40f);

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
            exhaust.visible = true;
            exhaust.material.opacity = 0.7;
            exhaust.scale.set(1, 1, 1);
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
            exhaust.visible = true;
            const tf = Math.max(0.0, Math.min(1.0, engine.thrust_factor));
            exhaust.material.opacity = 0.2 + 0.5 * tf;
            exhaust.scale.set(tf, 1, 1);
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
    const TRAIL_RADIUS = 0.6;
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
    let speed = 1.0;
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
        playing = !playing;
        btnPlay.textContent = playing ? "Pause" : "Play";
        if (playing) lastTickMs = performance.now();
    });

    const speedSelect = document.getElementById("speed");
    speedSelect.addEventListener("change", () => {
        speed = parseFloat(speedSelect.value);
    });

    document.getElementById("btn-cam-3d").addEventListener("click", preset3DCamera);
    document.getElementById("btn-cam-top").addEventListener("click", presetTopDown);
    document.getElementById("btn-cam-left").addEventListener("click", presetLeftSide);
    document.getElementById("btn-cam-right").addEventListener("click", presetRightSide);

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

    const SPEED_STEPS = [0.25, 0.5, 1, 2, 4];
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

    // ---- Animation loop ----
    function animate() {
        requestAnimationFrame(animate);
        if (playing) {
            const now = performance.now();
            const elapsed = (now - lastTickMs) / 1000.0;
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
    animate();
})();
