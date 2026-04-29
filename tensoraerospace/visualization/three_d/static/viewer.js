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
    renderer.setPixelRatio(window.devicePixelRatio);
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

    function _stdMat(color, opts) {
        return new THREE.MeshStandardMaterial({
            color, metalness: 0.35, roughness: 0.5,
            side: THREE.DoubleSide, ...opts,
        });
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
        // corners2d_body: array of [bx, by] in body frame; build a flat
        // quad in world coords using bodyToThree (z=0 in body → y=0 in
        // world, i.e. the wing lies on the centreline plane).
        const c = corners2d_body.map(([bx, by]) => bodyToThree(bx, by, 0));
        const geom = _quadGeom(c[0], c[1], c[2], c[3]);
        const mesh = new THREE.Mesh(geom, _stdMat(color));
        mesh.name = name;
        return mesh;
    }

    function _mirrorY(poly) {
        return poly.map(([x, y]) => [x, -y]);
    }

    function _fuselageGroup() {
        // F-16 fuselage. LatheGeometry from a hand-tuned profile that
        // captures the F-16's pointed radome, swelling cockpit area,
        // narrowing aft body, and engine nacelle.
        // Profile (radius, axial). Axial spans body x = +8 (nose) to -8 (tail).
        const profile = [
            new THREE.Vector2(0.00, +8.0),
            new THREE.Vector2(0.10, +7.5),    // radome point
            new THREE.Vector2(0.22, +6.5),
            new THREE.Vector2(0.40, +5.0),
            new THREE.Vector2(0.60, +3.5),
            new THREE.Vector2(0.78, +2.0),
            new THREE.Vector2(0.85, +0.5),    // widest at cockpit-mid
            new THREE.Vector2(0.85, -1.5),    // hold widest through midbody
            new THREE.Vector2(0.78, -3.5),
            new THREE.Vector2(0.62, -5.5),
            new THREE.Vector2(0.50, -7.0),
            new THREE.Vector2(0.42, -8.0),    // nozzle exit
        ];
        const fuselageGeom = new THREE.LatheGeometry(profile, 32);
        fuselageGeom.rotateZ(-Math.PI / 2);
        const fuselage = new THREE.Mesh(
            fuselageGeom, _stdMat(F16_COLORS.fuselage),
        );
        fuselage.name = "fuselage_main";

        const group = new THREE.Group();
        group.add(fuselage);

        // Bubble canopy — distinctive F-16 teardrop shape. Two-piece:
        //  - main bubble (rounded teardrop)
        //  - canopy frame (slightly darker thin band at the base)
        const canopyGeom = new THREE.SphereGeometry(1.0, 24, 16);
        canopyGeom.scale(2.0, 0.7, 1.0);  // long fore-aft, low, narrow
        const canopy = new THREE.Mesh(
            canopyGeom,
            _stdMat(F16_COLORS.canopy, { metalness: 0.7, roughness: 0.1 }),
        );
        canopy.name = "fuselage_canopy";
        canopy.position.set(2.6, 0.85, 0);  // forward of CG, on top of fuselage
        group.add(canopy);

        // Canopy frame band (slightly darker, sits at the canopy base)
        const frameGeom = new THREE.TorusGeometry(0.95, 0.09, 8, 32);
        const canopyFrame = new THREE.Mesh(
            frameGeom,
            _stdMat(0x1a2530, { metalness: 0.5, roughness: 0.3 }),
        );
        canopyFrame.rotation.x = Math.PI / 2;
        canopyFrame.scale.set(2.0, 1.0, 0.7);
        canopyFrame.position.set(2.6, 0.55, 0);
        group.add(canopyFrame);

        // Chin intake — distinctive F-16 single belly intake under the cockpit.
        // Wider than before, with a slight inlet ramp suggested by an inset
        // front face.
        const intakeBody = new THREE.Mesh(
            new THREE.BoxGeometry(4.5, 0.85, 1.5),
            _stdMat(F16_COLORS.intake),
        );
        intakeBody.name = "fuselage_intake";
        intakeBody.position.set(1.8, -0.95, 0);
        group.add(intakeBody);

        // Inlet ramp — a darker recessed face at the front of the intake to
        // suggest a lit inlet duct. Just a small inset plane.
        const inletGeom = new THREE.PlaneGeometry(1.3, 0.7);
        const inlet = new THREE.Mesh(
            inletGeom,
            _stdMat(0x101418, { metalness: 0.2, roughness: 1.0 }),
        );
        inlet.rotation.y = Math.PI / 2;
        inlet.position.set(4.0, -0.95, 0);
        group.add(inlet);

        // Nose pitot probe (slim cone forward of radome)
        const pitotGeom = new THREE.ConeGeometry(0.08, 0.7, 12);
        const pitot = new THREE.Mesh(
            pitotGeom, _stdMat(F16_COLORS.nozzle, { metalness: 0.7 }),
        );
        pitot.rotation.z = -Math.PI / 2;
        pitot.position.set(8.4, 0, 0);
        group.add(pitot);

        // Engine nozzle — proper afterburner-can shape.
        const nozzleGeom = new THREE.CylinderGeometry(
            0.50, 0.40, 1.0, 24, 1, true,
        );
        const nozzle = new THREE.Mesh(
            nozzleGeom, _stdMat(F16_COLORS.nozzle, { metalness: 0.65 }),
        );
        nozzle.rotation.z = Math.PI / 2;
        nozzle.position.set(-8.3, 0, 0);
        group.add(nozzle);

        // LERX — leading-edge root extensions. F-16's signature. Flat
        // wedges blending the forebody chine into the wing root, on each
        // side of the cockpit.
        const lerxRightShape = [
            // body coords (x, y) of the right LERX outline
            [+3.0, +0.55],   // front tip (near cockpit)
            [+1.5, +0.95],   // outer kink
            [-0.4, +1.95],   // joins wing root LE_outboard (matches RIGHT_WING_POLYGONS.right_root[1])
            [+0.4, +1.95],   // back to wing root LE_inboard adjacent
            [+1.5, +0.85],   // inboard back to fuselage line
        ];
        const lerxLeftShape = lerxRightShape.map(([x, y]) => [x, -y]);
        // Flat panels at z = 0 (centreline plane) — slightly above the
        // fuselage waterline at z = +0.4 looks more F-16-ish.
        function _lerxMesh(poly, name) {
            const c = poly.map(([bx, by]) => bodyToThree(bx, by, -0.15));
            // Build as a triangle fan from the first vertex
            const verts = [];
            for (let i = 1; i < c.length - 1; i++) {
                verts.push(...c[0], ...c[i], ...c[i + 1]);
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

        // Ventral fins — two small fins angled outward and down at the
        // aft fuselage (under the engine, between wing TE and nozzle).
        function _ventralFinMesh(side, name) {
            const sign = side === "left" ? -1 : +1;
            // Polygon in body x-z plane (y is sign * thickness offset),
            // flat triangle: top-fwd, top-aft, bottom-mid.
            const c = [
                bodyToThree(-5.5, sign * 0.4, +0.7),  // top-fwd
                bodyToThree(-7.0, sign * 0.4, +0.7),  // top-aft
                bodyToThree(-6.2, sign * 0.7, +1.6),  // bottom-mid (down + outward)
            ];
            const positions = new Float32Array([
                ...c[0], ...c[1], ...c[2],
            ]);
            const geom = new THREE.BufferGeometry();
            geom.setAttribute("position", new THREE.BufferAttribute(positions, 3));
            geom.computeVertexNormals();
            const mesh = new THREE.Mesh(
                geom, _stdMat(F16_COLORS.vtail),
            );
            mesh.name = name;
            return mesh;
        }
        group.add(_ventralFinMesh("right", "ventral_fin_right"));
        group.add(_ventralFinMesh("left",  "ventral_fin_left"));

        return group;
    }

    function _vtailMesh() {
        // F-16 vertical tail — swept fin centred on the tail upper surface.
        // Body coords (x, z): root at fuselage top (z = -0.6), tip up at z = -3.0
        // x_le_root = -3.5, x_te_root = -6.5; sweep ~45°.
        const c = [
            bodyToThree(-3.5, 0, -0.6),  // LE_root (lower forward)
            bodyToThree(-5.5, 0, -3.4),  // LE_tip (upper forward)
            bodyToThree(-6.5, 0, -3.4),  // TE_tip (upper aft)
            bodyToThree(-6.5, 0, -0.6),  // TE_root (lower aft)
        ];
        const geom = _quadGeom(c[0], c[1], c[2], c[3]);
        const mesh = new THREE.Mesh(geom, _stdMat(F16_COLORS.vtail));
        mesh.name = "vtail";
        return mesh;
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
        // Build mesh with vertices relative to hinge in body x-z plane
        const c = RUDDER_POLY_REL.map(
            ([bx, bz]) => bodyToThree(bx, 0, bz),
        );
        const geom = _quadGeom(c[0], c[1], c[2], c[3]);
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
        const c = STAB_RIGHT_POLY_REL.map(
            ([bx, by]) => bodyToThree(bx, sign * by, 0),
        );
        const geom = _quadGeom(c[0], c[1], c[2], c[3]);
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
        const c = AILERON_RIGHT_POLY_REL.map(
            ([bx, by]) => bodyToThree(bx, sign * by, 0),
        );
        const geom = _quadGeom(c[0], c[1], c[2], c[3]);
        group.add(new THREE.Mesh(geom, _stdMat(F16_COLORS.aileron)));
        return group;
    }

    function _wingtipLauncher(side, name) {
        // Short cylindrical rail on the wingtip's leading edge, where
        // an AIM-9 Sidewinder sits on a real F-16.
        const sign = side === "left" ? -1 : +1;
        // Tip body coords: leading edge of wing tip section ~(-2.0, ±4.7)
        const len = 1.6;
        const radius = 0.10;
        const railGeom = new THREE.CylinderGeometry(
            radius, radius, len, 12, 1, false,
        );
        // CylinderGeometry default axis is +Y; we want it along body x (forward).
        railGeom.rotateZ(Math.PI / 2);
        const rail = new THREE.Mesh(
            railGeom, _stdMat(0x808080, { metalness: 0.6, roughness: 0.4 }),
        );
        rail.name = name;
        // Position at body (~ -1.8, ±4.6, -0.1) — just forward of the
        // tip's LE corner, slightly above.
        const [tx, ty, tz] = bodyToThree(-1.8, sign * 4.6, -0.1);
        rail.position.set(tx, ty, tz);
        return rail;
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

        return aircraft;
    }

    const aircraft = buildAircraft(log.geometry);
    scene.add(aircraft);

    // ---- Build pitch ladder ----
    // Add horizontal pitch reference lines at every 5° from -25° to +25°.
    // Lines are static within the #hud-pitch-ladder group; the parent
    // groups translate (pitch) and rotate (bank) each frame.
    (function () {
        const ladder = document.getElementById("hud-pitch-ladder");
        if (!ladder) return;
        const PITCH_DEG_PER_PX = 5 / 30;   // 30 px per 5° pitch
        for (let d = -25; d <= 25; d += 5) {
            if (d === 0) continue;       // horizon drawn separately
            const y = -d / PITCH_DEG_PER_PX;
            const cls = d > 0 ? "hud-pitch-rung-pos" : "hud-pitch-rung-neg";
            // Rung: short segments either side of centre with the angle
            // label outboard.
            const segL = document.createElementNS(
                "http://www.w3.org/2000/svg", "line");
            segL.setAttribute("x1", "-100");
            segL.setAttribute("y1", y);
            segL.setAttribute("x2", "-30");
            segL.setAttribute("y2", y);
            segL.setAttribute("class", cls);
            ladder.appendChild(segL);
            const segR = document.createElementNS(
                "http://www.w3.org/2000/svg", "line");
            segR.setAttribute("x1", "30");
            segR.setAttribute("y1", y);
            segR.setAttribute("x2", "100");
            segR.setAttribute("y2", y);
            segR.setAttribute("class", cls);
            ladder.appendChild(segR);
            const lbl = document.createElementNS(
                "http://www.w3.org/2000/svg", "text");
            lbl.setAttribute("x", "115");
            lbl.setAttribute("y", y + 4);
            lbl.setAttribute("class", "hud-pitch-label");
            lbl.textContent = (d > 0 ? "+" : "") + d;
            ladder.appendChild(lbl);
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

        // ---- F-16-style SVG HUD ----
        // Convert sim units to fighter conventions:
        //   alpha, beta (rad)             → degrees
        //   airspeed (m/s, fixed in env)  → KIAS (knots) and Mach
        //   altitude (m, from initial Oy minus -position.z)
        //                                 → feet, plus VSI ft/min
        //   yaw / pitch / bank            → from traj.attitude
        //   load factor                   → approximated from cy at α
        const ALPHA_PX_PER_DEG = 6;     // FPM travels 6 px per °α
        const BETA_PX_PER_DEG  = 6;
        const PITCH_DEG_PER_PX_INV = 30 / 5;   // 30 px per 5°

        const att = traj.attitude[idx];   // (roll/gamma, pitch/theta, yaw/psi)
        const alphaRad = traj.alpha[idx];
        const betaRad  = traj.beta[idx];
        const wzRad    = traj.wz[idx];

        const rollDeg  = att[0] * 180 / Math.PI;
        const pitchDeg = att[1] * 180 / Math.PI;
        const yawDeg   = att[2] * 180 / Math.PI;
        const alphaDeg = alphaRad * 180 / Math.PI;
        const betaDeg  = betaRad  * 180 / Math.PI;

        // Heading: yaw in [0, 360)
        let hdg = yawDeg % 360;
        if (hdg < 0) hdg += 360;
        const hdgEl = document.getElementById("hud-hdg-value");
        if (hdgEl) hdgEl.textContent = String(Math.round(hdg)).padStart(3, "0");

        // Airspeed → KIAS, Mach. Prefer the value from the env's F-16
        // model (params.V is the simulation's true airspeed); fall back
        // to log.metadata.airspeed if the env didn't expose params.
        const params = (log.metadata && log.metadata.params) || {};
        const airspeed_mps = params.V
            || (log.metadata && log.metadata.airspeed) || 0;
        const kias = airspeed_mps * 1.94384;
        const mach = airspeed_mps / 340.0;
        const kiasEl = document.getElementById("hud-kias-value");
        const machEl = document.getElementById("hud-mach-value");
        if (kiasEl) kiasEl.textContent = String(Math.round(kias)).padStart(3, "0");
        if (machEl) machEl.textContent = "M " + mach.toFixed(2);

        // Altitude: trim altitude from the F-16 model parameters (Oy in
        // metres) plus the inertial-z displacement from the trajectory.
        // -bz is up in the simulation's body frame.
        const trimAlt_m = params.Oy || 0;
        const pos = traj.position[idx];
        const alt_m = trimAlt_m + (-pos[2]);
        const alt_ft = alt_m * 3.28084;
        const altEl = document.getElementById("hud-alt-value");
        if (altEl) altEl.textContent = String(Math.round(alt_ft)).padStart(5, "0");

        // VSI: ft/min, from finite difference on altitude over a
        // lookback window.
        const VSI_LOOKBACK = 50;   // 50 frames ≈ 0.5 s at dt=0.01
        const j = Math.max(0, idx - VSI_LOOKBACK);
        const altPrev_m = trimAlt_m + (-traj.position[j][2]);
        const dT_s = (idx - j) * dt || dt;
        const vsi_fps = (alt_m - altPrev_m) * 3.28084 / dT_s;
        const vsi_fpm = vsi_fps * 60.0;
        const vsiEl = document.getElementById("hud-vsi-value");
        if (vsiEl) {
            const sign = vsi_fpm >= 0 ? "+" : "-";
            vsiEl.textContent = "VS " + sign +
                String(Math.round(Math.abs(vsi_fpm))).padStart(4, "0");
        }

        // AOA / β
        const aoaEl = document.getElementById("hud-aoa-value");
        if (aoaEl) aoaEl.textContent = alphaDeg.toFixed(1);
        const betaValEl = document.getElementById("hud-beta-value");
        if (betaValEl) betaValEl.textContent = betaDeg.toFixed(1);

        // G-load: estimate from pitch rate × airspeed / g (centripetal
        // approximation in the pitch plane). Adds 1 G for steady level
        // flight. Uses the F-16 model's g constant if available so the
        // calc agrees with the physics module's gravity.
        const G_EARTH = params.g || 9.80665;
        const gLoad = 1.0 + Math.abs(wzRad * airspeed_mps) / G_EARTH;
        const gEl = document.getElementById("hud-g-value");
        if (gEl) gEl.textContent = gLoad.toFixed(1);

        // Bank pointer rotation (top scale): bank in degrees, around the
        // pivot at (600, 110). The HTML transform was set there already.
        const bankPointer = document.getElementById("hud-bank-pointer");
        if (bankPointer) {
            // Pointer rotates -bank (so positive bank tilts left as on
            // the real F-16 attitude indicator).
            bankPointer.setAttribute(
                "transform", "rotate(" + (-rollDeg).toFixed(1) + ")",
            );
        }

        // Pitch ladder + horizon: bank rotation around screen centre
        // then pitch translation along screen y. 30 px = 5° pitch.
        const bankRot = document.getElementById("hud-bank-rot");
        const pitchTrans = document.getElementById("hud-pitch-trans");
        if (bankRot) {
            bankRot.setAttribute(
                "transform",
                "translate(600 350) rotate(" + (-rollDeg).toFixed(1) + ")",
            );
        }
        if (pitchTrans) {
            const pitchPx = pitchDeg * PITCH_DEG_PER_PX_INV;
            pitchTrans.setAttribute(
                "transform", "translate(0 " + pitchPx.toFixed(1) + ")",
            );
        }

        // FPM (Flight Path Marker): offset from screen centre by α / β.
        // On a real HUD the FPM sits where the velocity vector points;
        // it's offset down by α (positive α = nose above velocity vector
        // = FPM below reticle) and offset opposite to β (positive β =
        // wind from right = velocity vector points slightly right of
        // nose = FPM right of reticle).
        const fpm = document.getElementById("hud-fpm");
        if (fpm) {
            const fx = 600 - betaDeg * BETA_PX_PER_DEG;
            const fy = 350 + alphaDeg * ALPHA_PX_PER_DEG;
            fpm.setAttribute(
                "transform", "translate(" + fx.toFixed(1) + " " + fy.toFixed(1) + ")",
            );
        }

        // Time + speed status (bottom-right)
        const timeValEl = document.getElementById("hud-time-value");
        if (timeValEl) timeValEl.textContent = traj.time[idx].toFixed(1);
        const speedValEl = document.getElementById("hud-speed-value");
        if (speedValEl) speedValEl.textContent = speed + "×";

        // Damage caution: visible if any section is currently lost.
        const ds = damageStateAt(traj.time[idx]);
        const cautionEl = document.getElementById("hud-caution");
        if (cautionEl) {
            let anyDamage = false;
            if (ds && ds.section_loss) {
                for (const k in ds.section_loss) {
                    if (ds.section_loss[k] > 0) { anyDamage = true; break; }
                }
            }
            if (!anyDamage && ds && ds.engine && ds.engine.hard_failure) {
                anyDamage = true;
            }
            if (!anyDamage && ds && ds.control_failures) {
                for (const k in ds.control_failures) {
                    const cf = ds.control_failures[k];
                    if (cf && cf.mode && cf.mode !== "healthy") {
                        anyDamage = true; break;
                    }
                }
            }
            cautionEl.setAttribute(
                "visibility", anyDamage ? "visible" : "hidden",
            );
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
