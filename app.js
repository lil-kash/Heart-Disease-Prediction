/* =====================================================
   CardioSense — Main Application Script
   - Three.js hero background with animated particles
   - Scroll/navbar effects
   - AOS (Animate On Scroll) manual implementation
   - Prediction form + results with animated gauge
===================================================== */

/* =====================================================
   THREE.JS HERO CANVAS
===================================================== */
(function initThreeJS() {
    const canvas = document.getElementById('hero-canvas');
    if (!canvas || !window.THREE) return;

    const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(window.innerWidth, window.innerHeight);

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 1000);
    camera.position.set(0, 0, 5);

    // === Particle Field ===
    const PARTICLE_COUNT = 2500;
    const positions = new Float32Array(PARTICLE_COUNT * 3);
    const colors = new Float32Array(PARTICLE_COUNT * 3);
    const sizes = new Float32Array(PARTICLE_COUNT);

    const colorOptions = [
        new THREE.Color('#e63946'),
        new THREE.Color('#4361ee'),
        new THREE.Color('#ff6b6b'),
    ];

    for (let i = 0; i < PARTICLE_COUNT; i++) {
        // Sphere distribution
        const theta = Math.random() * Math.PI * 2;
        const phi = Math.acos(2 * Math.random() - 1);
        const r = 2.5 + Math.random() * 4;

        positions[i * 3] = r * Math.sin(phi) * Math.cos(theta);
        positions[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta);
        positions[i * 3 + 2] = r * Math.cos(phi);

        const c = colorOptions[Math.floor(Math.random() * colorOptions.length)];
        colors[i * 3] = c.r;
        colors[i * 3 + 1] = c.g;
        colors[i * 3 + 2] = c.b;

        sizes[i] = 0.5 + Math.random() * 1.5;
    }

    const particleGeo = new THREE.BufferGeometry();
    particleGeo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    particleGeo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    particleGeo.setAttribute('size', new THREE.BufferAttribute(sizes, 1));

    const particleMat = new THREE.PointsMaterial({
        size: 0.035,
        vertexColors: true,
        transparent: true,
        opacity: 0.6,
        sizeAttenuation: true,
    });

    const particles = new THREE.Points(particleGeo, particleMat);
    scene.add(particles);

    // === Central Glowing Sphere ===
    const sphereGeo = new THREE.SphereGeometry(1.0, 64, 64);
    const sphereMat = new THREE.MeshStandardMaterial({
        color: new THREE.Color('#e63946'),
        emissive: new THREE.Color('#e63946'),
        emissiveIntensity: 0.3,
        roughness: 0.3,
        metalness: 0.8,
        wireframe: false,
    });
    const sphere = new THREE.Mesh(sphereGeo, sphereMat);
    scene.add(sphere);

    // Wireframe overlay
    const wireGeo = new THREE.SphereGeometry(1.02, 20, 20);
    const wireMat = new THREE.MeshBasicMaterial({
        color: new THREE.Color('#e63946'),
        wireframe: true,
        transparent: true,
        opacity: 0.1,
    });
    const wire = new THREE.Mesh(wireGeo, wireMat);
    scene.add(wire);

    // === Rings ===
    function createRing(radius, color, opacity) {
        const geo = new THREE.TorusGeometry(radius, 0.004, 8, 100);
        const mat = new THREE.MeshBasicMaterial({ color, transparent: true, opacity });
        return new THREE.Mesh(geo, mat);
    }

    const ring1 = createRing(1.6, '#e63946', 0.3);
    ring1.rotation.x = Math.PI / 3;
    scene.add(ring1);

    const ring2 = createRing(2.2, '#4361ee', 0.15);
    ring2.rotation.x = Math.PI / 5;
    ring2.rotation.y = Math.PI / 4;
    scene.add(ring2);

    // === Lighting ===
    const ambientLight = new THREE.AmbientLight('#ffffff', 0.4);
    scene.add(ambientLight);

    const pointLight = new THREE.PointLight('#e63946', 2, 10);
    pointLight.position.set(2, 2, 2);
    scene.add(pointLight);

    const pointLight2 = new THREE.PointLight('#4361ee', 1, 10);
    pointLight2.position.set(-2, -2, 2);
    scene.add(pointLight2);

    // Move group to right side
    const group = new THREE.Group();
    group.add(sphere, wire, ring1, ring2, particles);
    group.position.set(3, -0.2, 0);
    scene.add(group);

    // === Mouse tracking ===
    let mouseX = 0, mouseY = 0;
    document.addEventListener('mousemove', (e) => {
        mouseX = (e.clientX / window.innerWidth - 0.5) * 2;
        mouseY = -(e.clientY / window.innerHeight - 0.5) * 2;
    });

    // === Animation Loop ===
    let t = 0;
    function animate() {
        requestAnimationFrame(animate);
        t += 0.005;

        sphere.rotation.y = t * 0.4;
        sphere.rotation.x = t * 0.1;
        wire.rotation.y = -t * 0.25;
        particles.rotation.y = t * 0.03;
        ring1.rotation.z = t * 0.3;
        ring2.rotation.z = -t * 0.2;

        // Subtle mouse parallax
        group.rotation.y += (mouseX * 0.3 - group.rotation.y) * 0.04;
        group.rotation.x += (mouseY * 0.15 - group.rotation.x) * 0.04;

        // Pulsing sphere scale
        const pulse = 1 + 0.04 * Math.sin(t * 1.5);
        sphere.scale.setScalar(pulse);
        wire.scale.setScalar(pulse);

        renderer.render(scene, camera);
    }

    animate();

    window.addEventListener('resize', () => {
        camera.aspect = window.innerWidth / window.innerHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(window.innerWidth, window.innerHeight);
    });
})();

/* =====================================================
   NAVBAR SCROLL EFFECT
===================================================== */
const nav = document.getElementById('nav');
window.addEventListener('scroll', () => {
    if (window.scrollY > 60) {
        nav.classList.add('scrolled');
    } else {
        nav.classList.remove('scrolled');
    }
});

/* =====================================================
   AOS - ANIMATE ON SCROLL
===================================================== */
function initAOS() {
    const elements = document.querySelectorAll('[data-aos]');
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const delay = entry.target.getAttribute('data-aos-delay') || 0;
                setTimeout(() => {
                    entry.target.classList.add('aos-in');
                }, parseInt(delay));
                observer.unobserve(entry.target);
            }
        });
    }, { threshold: 0.1 });

    elements.forEach(el => observer.observe(el));
}

initAOS();

/* =====================================================
   PREDICTION FORM LOGIC
===================================================== */
document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('prediction-form');
    const predictBtn = document.getElementById('predict-btn');
    const btnText = document.getElementById('btn-text');
    const btnLoader = document.getElementById('btn-loader');

    const resultIdle = document.getElementById('result-idle');
    const resultData = document.getElementById('result-data');
    const riskBadge = document.getElementById('risk-badge');
    const gaugeFill = document.getElementById('gauge-fill');
    const gaugePct = document.getElementById('gauge-pct');
    const resultVerdict = document.getElementById('result-verdict');
    const rPrediction = document.getElementById('r-prediction');
    const rConfidence = document.getElementById('r-confidence');
    const resetBtn = document.getElementById('reset-btn');

    const API_URL = '/predict';
    const GAUGE_LENGTH = 157; // Approx arc length of the half-circle

    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        // Show loader
        btnText.textContent = 'Analyzing...';
        btnLoader.classList.remove('hidden');

        const formData = new FormData(form);
        const patientData = {};
        for (let [key, val] of formData.entries()) {
            patientData[key] = parseFloat(val);
        }

        try {
            const response = await fetch(API_URL, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(patientData),
            });

            if (!response.ok) {
                const err = await response.json();
                throw new Error(err.detail || 'Server error');
            }

            const result = await response.json();
            displayResults(result);

            // Smooth scroll to results on mobile
            if (window.innerWidth < 1024) {
                document.getElementById('results-panel').scrollIntoView({ behavior: 'smooth', block: 'start' });
            }

        } catch (error) {
            alert(`Error: ${error.message}`);
        } finally {
            btnText.textContent = 'Analyze Patient';
            btnLoader.classList.add('hidden');
        }
    });

    function displayResults(data) {
        resultIdle.classList.add('hidden');
        resultData.classList.remove('hidden');

        // Risk Badge
        const riskText = data.Risk_Level;
        riskBadge.textContent = riskText;
        riskBadge.className = 'result-risk-badge';
        if (riskText.includes('HIGH')) riskBadge.classList.add('high');
        else if (riskText.includes('MODERATE')) riskBadge.classList.add('mod');
        else riskBadge.classList.add('low');

        // Gauge
        const prob = data.Probability;
        const pct = (prob * 100).toFixed(1);

        setTimeout(() => {
            const offset = GAUGE_LENGTH - (prob * GAUGE_LENGTH);
            gaugeFill.style.strokeDashoffset = offset;
            gaugePct.textContent = `${pct}%`;

            // Color the gauge
            if (prob >= 0.6) {
                gaugeFill.style.stroke = '#e63946';
                gaugeFill.style.filter = 'drop-shadow(0 0 6px #e63946)';
            } else if (prob >= 0.4) {
                gaugeFill.style.stroke = '#ffbf00';
                gaugeFill.style.filter = 'drop-shadow(0 0 6px #ffbf00)';
            } else {
                gaugeFill.style.stroke = '#06d6a0';
                gaugeFill.style.filter = 'drop-shadow(0 0 6px #06d6a0)';
            }
        }, 100);

        // Verdict text
        if (data.Prediction === 'Heart Disease') {
            resultVerdict.innerHTML = `<strong style="color:#ff6b6b">Positive indication</strong> for cardiovascular disease detected. Clinical follow-up is strongly recommended.`;
        } else {
            resultVerdict.innerHTML = `<strong style="color:#06d6a0">No significant indicators</strong> of heart disease detected. Continue routine screening.`;
        }

        rPrediction.textContent = data.Prediction;
        rConfidence.textContent = `${pct}%`;
    }

    resetBtn.addEventListener('click', () => {
        resultData.classList.add('hidden');
        resultIdle.classList.remove('hidden');
        form.reset();
        // Reset gauge
        gaugeFill.style.strokeDashoffset = GAUGE_LENGTH;
        gaugePct.textContent = '0%';
    });
});
