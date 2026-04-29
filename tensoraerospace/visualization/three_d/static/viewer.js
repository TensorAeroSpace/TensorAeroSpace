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
    camera.position.set(45, 30, 45);
    camera.lookAt(0, 0, 0);

    const controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.minDistance = 5;
    controls.maxDistance = 1500;

    function preset3DCamera() {
        controls.enabled = true;
        const p = aircraft.position;
        camera.position.set(p.x + 30, p.y + 20, p.z + 30);
        controls.target.copy(p);
        controls.update();
    }

    function presetTopDown() {
        controls.enabled = true;
        const p = aircraft.position;
        camera.position.set(p.x, p.y + 60, p.z);
        controls.target.copy(p);
        controls.update();
    }

    function presetLeftSide() {
        controls.enabled = true;
        const p = aircraft.position;
        // body.y < 0 = aircraft's left side; in three.js z = body.y, so left
        // side is at three.z < 0. Camera there looks at the left flank.
        camera.position.set(p.x, p.y + 5, p.z - 35);
        controls.target.copy(p);
        controls.update();
    }

    function presetRightSide() {
        controls.enabled = true;
        const p = aircraft.position;
        // body.y > 0 = aircraft's right side
        camera.position.set(p.x, p.y + 5, p.z + 35);
        controls.target.copy(p);
        controls.update();
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

    // Stabilator (horizontal tail): one trapezoid per side, smaller and
    // further aft. Same coord system.
    const RIGHT_STAB_POLY = [
        [-3.5,  0.4],   // LE_in
        [-4.7,  1.8],   // LE_out
        [-6.2,  1.8],   // TE_out
        [-5.5,  0.4],   // TE_in
    ];

    // Aileron (right): trailing-edge flap at outer wing. Body coords.
    const RIGHT_AILERON_POLY = [
        [-3.4,  3.2],
        [-3.7,  4.2],
        [-3.9,  4.2],
        [-3.5,  3.2],
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
        // F-16 fuselage as a body of revolution (LatheGeometry).
        // Profile points are (radius, axial_position) in body frame; after
        // lathing, the geometry is rotated so its axial direction aligns
        // with three.x (body forward).
        //
        // Profile shape:
        //   nose tip at body.x = +7.5, radius 0
        //   forward fuselage tapers up to ~0.75 m radius at body.x = +1
        //   centre body holds ~0.75 m radius from x=+1 to x=-3
        //   tail tapers to ~0.35 m radius at the nozzle (body.x = -7.5)
        const profile = [
            new THREE.Vector2(0.0,  +7.5),
            new THREE.Vector2(0.18, +6.5),
            new THREE.Vector2(0.32, +5.5),
            new THREE.Vector2(0.50, +4.0),
            new THREE.Vector2(0.65, +2.5),
            new THREE.Vector2(0.75, +1.0),
            new THREE.Vector2(0.78, -1.0),
            new THREE.Vector2(0.78, -3.0),
            new THREE.Vector2(0.65, -5.0),
            new THREE.Vector2(0.50, -6.5),
            new THREE.Vector2(0.35, -7.5),
        ];
        const geom = new THREE.LatheGeometry(profile, 24);
        // LatheGeometry's axial axis is local +Y. Rotate so that +Y → +X.
        // Rotation around Z by -π/2 gives: (x,y,z) → (y,-x,z).
        // Body's x-forward is rendered as three.js +X, which matches.
        geom.rotateZ(-Math.PI / 2);
        const fuselage = new THREE.Mesh(geom, _stdMat(F16_COLORS.fuselage));
        fuselage.name = "fuselage_main";

        const group = new THREE.Group();
        group.add(fuselage);

        // Cockpit canopy — a flattened sphere on top of the cockpit.
        // body coords: x ≈ +3.0 (forward of CG), z ≈ -0.6 (above fuselage axis)
        const canopyGeom = new THREE.SphereGeometry(0.9, 16, 12);
        canopyGeom.scale(1.6, 0.55, 0.95);  // long fore-aft, low, narrow
        const canopy = new THREE.Mesh(
            canopyGeom,
            _stdMat(F16_COLORS.canopy, { metalness: 0.6, roughness: 0.15 }),
        );
        canopy.name = "fuselage_canopy";
        // Position in body frame: +3.0 forward, 0 lateral, -0.6 up (in body z),
        // i.e. world coords (3.0, 0.6, 0).
        canopy.position.set(3.0, 0.6, 0);
        group.add(canopy);

        // Engine intake — flattened box mounted under the belly.
        // F-16 has the iconic single ventral chin intake.
        const intakeGeom = new THREE.BoxGeometry(3.5, 0.55, 1.1);
        const intake = new THREE.Mesh(intakeGeom, _stdMat(F16_COLORS.intake));
        intake.name = "fuselage_intake";
        // body x ≈ +1.5 (under cockpit), body z ≈ +0.8 (below axis)
        // → world (1.5, -0.8, 0)
        intake.position.set(1.5, -0.8, 0);
        group.add(intake);

        // Nose pitot tube — small cone at the very tip.
        const pitotGeom = new THREE.ConeGeometry(0.08, 0.6, 8);
        const pitot = new THREE.Mesh(
            pitotGeom, _stdMat(F16_COLORS.nozzle, { metalness: 0.7 }),
        );
        // ConeGeometry default axis is +Y; we want it pointing along +X (forward).
        pitot.rotation.z = -Math.PI / 2;
        pitot.position.set(7.8, 0, 0);
        group.add(pitot);

        // Engine nozzle — dark ring at the tail.
        const nozzleGeom = new THREE.CylinderGeometry(
            0.40, 0.32, 0.6, 16, 1, true,
        );
        const nozzle = new THREE.Mesh(
            nozzleGeom, _stdMat(F16_COLORS.nozzle, { metalness: 0.6 }),
        );
        nozzle.rotation.z = Math.PI / 2;
        nozzle.position.set(-7.7, 0, 0);
        group.add(nozzle);

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

    function _rudderMesh() {
        // Rudder — small flap at the trailing edge of the vtail.
        const c = [
            bodyToThree(-6.4, 0, -0.6),
            bodyToThree(-6.4, 0, -3.4),
            bodyToThree(-6.7, 0, -3.4),
            bodyToThree(-6.7, 0, -0.6),
        ];
        const geom = _quadGeom(c[0], c[1], c[2], c[3]);
        const mesh = new THREE.Mesh(geom, _stdMat(F16_COLORS.rudder));
        mesh.name = "rudder";
        return mesh;
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

        // Stabilator: one trapezoid per side.
        aircraft.add(_flatSectionMesh(RIGHT_STAB_POLY,
                                       F16_COLORS.stab, "stab_right"));
        aircraft.add(_flatSectionMesh(_mirrorY(RIGHT_STAB_POLY),
                                       F16_COLORS.stab, "stab_left"));

        // Vertical tail + rudder
        aircraft.add(_vtailMesh());
        aircraft.add(_rudderMesh());

        // Ailerons (small flaps at the trailing edge of the outer wing).
        aircraft.add(_flatSectionMesh(RIGHT_AILERON_POLY,
                                       F16_COLORS.aileron, "aileron_right"));
        aircraft.add(_flatSectionMesh(_mirrorY(RIGHT_AILERON_POLY),
                                       F16_COLORS.aileron, "aileron_left"));

        return aircraft;
    }

    const aircraft = buildAircraft(log.geometry);
    scene.add(aircraft);

    // Engine exhaust glow — a small orange cone trailing behind the
    // fuselage. Phase E damage handling scales it with engine thrust_factor
    // and hides it on hard_failure.
    const exhaustMat = new THREE.MeshBasicMaterial({
        color: 0xff6633, transparent: true, opacity: 0.7,
    });
    const exhaustGeom = new THREE.ConeGeometry(0.5, 3.0, 12, 1, true);
    const exhaust = new THREE.Mesh(exhaustGeom, exhaustMat);
    exhaust.name = "exhaust";
    // Cone default axis is +Y; rotate so it points along -X (aft of fuselage)
    exhaust.rotation.z = Math.PI / 2;
    // Mount aft of fuselage centre. Body x = -7 means 7 m aft of CG; in
    // three coords that's -7 along X.
    exhaust.position.set(-7, 0, 0);
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
    aircraft.traverse((obj) => {
        if (obj.isMesh && obj.material && obj.name && obj.name !== "exhaust") {
            obj.material = obj.material.clone();
            obj.material.transparent = true;
            sectionMaterials.set(obj.name, {
                color: obj.material.color.clone(),
                opacity: 1.0,
            });
        }
    });

    const HEALTHY_COLOR = new THREE.Color(0xffffff);  // not used directly;
                                                       // we lerp from base
    const DAMAGE_RED = new THREE.Color(0xc0392b);
    const JAM_YELLOW = new THREE.Color(0xf1c40f);

    function applyDamageState(state) {
        if (!state) {
            // Reset all sections to their original material
            for (const [name, ref] of sectionMaterials.entries()) {
                const m = aircraft.getObjectByName(name);
                if (!m) continue;
                m.visible = true;
                m.material.color.copy(ref.color);
                m.material.opacity = ref.opacity;
                m.material.emissive = new THREE.Color(0x000000);
            }
            exhaust.visible = true;
            exhaust.material.opacity = 0.7;
            exhaust.scale.set(1, 1, 1);
            return;
        }

        // Section loss → red tint + fade
        const lossMap = state.section_loss || {};
        for (const [name, ref] of sectionMaterials.entries()) {
            const m = aircraft.getObjectByName(name);
            if (!m) continue;
            const f = lossMap[name] || 0.0;
            if (f <= 0) {
                m.visible = true;
                m.material.color.copy(ref.color);
                m.material.opacity = ref.opacity;
                m.material.emissive = new THREE.Color(0x000000);
            } else if (f >= 1) {
                m.visible = false;
            } else {
                m.visible = true;
                // Lerp colour toward red, opacity toward 0
                m.material.color.copy(ref.color).lerp(DAMAGE_RED, f);
                m.material.opacity = (1 - f) * ref.opacity;
                m.material.emissive = new THREE.Color(0x000000);
            }
        }

        // Control failures → yellow emissive outline
        const failures = state.control_failures || {};
        for (const surface in failures) {
            const m = aircraft.getObjectByName(surface);
            if (!m) continue;
            const failure = failures[surface];
            if (failure.mode === "healthy") continue;
            m.material.emissive = JAM_YELLOW.clone().multiplyScalar(0.6);
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
    const trailMat = new THREE.LineBasicMaterial({ color: 0x4a90e2, linewidth: 2 });
    const trailGeom = new THREE.BufferGeometry();
    const trailPositions = new Float32Array(T * 3);
    trailGeom.setAttribute("position", new THREE.BufferAttribute(trailPositions, 3));
    trailGeom.setDrawRange(0, 0);
    const trailLine = new THREE.Line(trailGeom, trailMat);
    scene.add(trailLine);

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

        // Update trail
        for (let k = 0; k <= idx; k++) {
            const p = traj.position[k];
            trailPositions[k * 3 + 0] = p[0];
            trailPositions[k * 3 + 1] = -p[2];
            trailPositions[k * 3 + 2] = p[1];
        }
        trailGeom.attributes.position.needsUpdate = true;
        trailGeom.setDrawRange(0, idx + 1);

        // Apply damage state for this time
        applyDamageState(damageStateAt(traj.time[idx]));

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
