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

    // ---- Procedural F-16 from log.geometry.sections ----
    function bodyToThree(bx, by, bz) {
        // Body frame (x-fwd, y-right, z-down) → three.js (x-fwd, y-up, z-right)
        return [bx, -bz, by];
    }

    function _materialFor(type, side) {
        // Slight per-type tinting for readability without being garish
        const base = {
            wing:     0x2a7fb8,
            stab:     0x3d8e57,
            vtail:    0x9b59b6,
            control:  0x27ae60,
            fuselage: 0x95a5a6,
        }[type] || 0xaaaaaa;
        return new THREE.MeshStandardMaterial({
            color: base, metalness: 0.3, roughness: 0.55,
            side: THREE.DoubleSide,
        });
    }

    function _trapezoidPanel(spanCenter, spanExtent, chord, xCenter, sweepRad) {
        // Build a flat trapezoidal panel for a wing/stab section in body
        // frame (x-fwd, y-right, z-down). The panel lies in the body x-y
        // plane (z = 0). Sweep tilts the leading edge backward as we move
        // outboard (|y| grows).
        // Returns a BufferGeometry with two triangles (4 vertices).
        //
        // spanCenter   : y-coord of section centroid
        // spanExtent   : total span (m) covered by the section
        // chord        : streamwise extent at section centre
        // xCenter      : x-coord of section centroid (= aero_x_arm)
        // sweepRad     : leading-edge sweep angle (rad)
        const yIn  = spanCenter - 0.5 * Math.sign(spanCenter || 1) * spanExtent;
        const yOut = spanCenter + 0.5 * Math.sign(spanCenter || 1) * spanExtent;
        // Outboard sweep offset: tip moves backward (-x) by tan(sweep) * |Δy|
        const outboardDX = -Math.abs(yOut - spanCenter) * Math.tan(sweepRad);
        // Four corners (in body frame, z=0):
        //   LE-inboard, LE-outboard, TE-outboard, TE-inboard
        const xLE_in  = xCenter + 0.5 * chord;
        const xTE_in  = xCenter - 0.5 * chord;
        const xLE_out = xLE_in + outboardDX;
        const xTE_out = xTE_in + outboardDX;
        const corners = [
            bodyToThree(xLE_in,  yIn,  0),
            bodyToThree(xLE_out, yOut, 0),
            bodyToThree(xTE_out, yOut, 0),
            bodyToThree(xTE_in,  yIn,  0),
        ];
        const positions = new Float32Array([
            ...corners[0], ...corners[1], ...corners[2],
            ...corners[0], ...corners[2], ...corners[3],
        ]);
        const geom = new THREE.BufferGeometry();
        geom.setAttribute("position", new THREE.BufferAttribute(positions, 3));
        geom.computeVertexNormals();
        return geom;
    }

    function _vtailPanel(s) {
        // Vertical fin: flat trapezoid in body x-z plane, mounted at
        // span_position=0. The "span" of a vtail is in -z direction (up
        // in three.js).
        const xC = s.aero_x_arm;
        const c = s.chord;
        const heightBody = s.area / Math.max(c, 0.1);  // vertical extent
        // Body z grows downward, so the vtail extends in -z (up)
        const zRoot = 0;                  // attached at fuselage top
        const zTip  = -heightBody;        // upward
        const sweepRad = s.sweep || 0;
        const tipDX = -heightBody * Math.tan(sweepRad);
        const xLE_root = xC + 0.5 * c;
        const xTE_root = xC - 0.5 * c;
        const xLE_tip = xLE_root + tipDX;
        const xTE_tip = xTE_root + tipDX;
        const corners = [
            bodyToThree(xLE_root, 0, zRoot),
            bodyToThree(xLE_tip,  0, zTip),
            bodyToThree(xTE_tip,  0, zTip),
            bodyToThree(xTE_root, 0, zRoot),
        ];
        const positions = new Float32Array([
            ...corners[0], ...corners[1], ...corners[2],
            ...corners[0], ...corners[2], ...corners[3],
        ]);
        const geom = new THREE.BufferGeometry();
        geom.setAttribute("position", new THREE.BufferAttribute(positions, 3));
        geom.computeVertexNormals();
        return geom;
    }

    function _fuselageMesh(s) {
        // F-16 fuselage: stretched ellipsoid approximating ~14 m length,
        // ~1.2 m diameter. SphereGeometry scaled along the body x-axis.
        // Centred at (0, 0, 0) in body frame, then mapped into three.
        const sphereGeom = new THREE.SphereGeometry(0.7, 16, 12);
        const mat = _materialFor("fuselage", "center");
        const mesh = new THREE.Mesh(sphereGeom, mat);
        // Stretch X (forward axis); body-x in three.js is also +X.
        mesh.scale.set(10.0, 1.0, 1.0);
        return mesh;
    }

    function _controlSurfaceMesh(s) {
        // Small flap/rudder/aileron — thin trapezoid attached at the
        // trailing edge of its parent surface. We just render it as a
        // smaller flat panel for visual reference.
        const c = Math.max(s.chord, 0.3);
        const yC = s.span_position;
        const xC = s.aero_x_arm;
        const span = s.area / Math.max(c, 0.1);
        const corners = [
            bodyToThree(xC + 0.5 * c, yC - 0.5 * span, 0),
            bodyToThree(xC + 0.5 * c, yC + 0.5 * span, 0),
            bodyToThree(xC - 0.5 * c, yC + 0.5 * span, 0),
            bodyToThree(xC - 0.5 * c, yC - 0.5 * span, 0),
        ];
        const positions = new Float32Array([
            ...corners[0], ...corners[1], ...corners[2],
            ...corners[0], ...corners[2], ...corners[3],
        ]);
        const geom = new THREE.BufferGeometry();
        geom.setAttribute("position", new THREE.BufferAttribute(positions, 3));
        geom.computeVertexNormals();
        return new THREE.Mesh(geom, _materialFor("control", s.side));
    }

    function _wingOrStabMesh(s) {
        const span = s.area / Math.max(s.chord, 0.1);
        const geom = _trapezoidPanel(
            s.span_position, span, s.chord, s.aero_x_arm, s.sweep || 0,
        );
        return new THREE.Mesh(geom, _materialFor(s.type, s.side));
    }

    function buildAircraft(geometry) {
        const group = new THREE.Group();
        group.name = "aircraft";
        for (const s of geometry.sections) {
            let mesh;
            if (s.type === "wing" || s.type === "stab") {
                mesh = _wingOrStabMesh(s);
            } else if (s.type === "vtail") {
                const geom = _vtailPanel(s);
                mesh = new THREE.Mesh(geom, _materialFor("vtail", "center"));
            } else if (s.type === "control") {
                mesh = _controlSurfaceMesh(s);
            } else if (s.type === "fuselage") {
                mesh = _fuselageMesh(s);
            } else {
                continue;
            }
            mesh.name = s.name;
            mesh.userData = { type: s.type, side: s.side };
            group.add(mesh);
        }
        return group;
    }

    const aircraft = buildAircraft(log.geometry);
    scene.add(aircraft);

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

        // Update HUD
        document.getElementById("hud-time").textContent =
            traj.time[idx].toFixed(2) + " s";
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
