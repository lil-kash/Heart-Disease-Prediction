document.addEventListener('DOMContentLoaded', () => {
    // --- Tab Navigation Logic ---
    const navItems = document.querySelectorAll('.nav-item');
    const tabContents = document.querySelectorAll('.tab-content');
    const pageTitle = document.getElementById('page-title');
    const pageDesc = document.getElementById('page-desc');

    const tabInfo = {
        'predict-tab': { title: 'Patient Diagnostics', desc: 'Enter clinical parameters for instant AI-driven heart disease risk assessment.' },
        'eda-tab': { title: 'Data Analysis', desc: 'Exploratory visualization of population health metrics and dataset distributions.' },
        'model-tab': { title: 'ML Models', desc: 'Performance evaluation metrics for the Random Forest classification engine.' }
    };

    navItems.forEach(item => {
        item.addEventListener('click', (e) => {
            e.preventDefault();

            // Remove active from all navs & hide all tabs
            navItems.forEach(nav => nav.classList.remove('active'));
            tabContents.forEach(tab => {
                tab.classList.remove('active-tab');
                tab.classList.add('hidden-tab');
            });

            // Add active to clicked nav & show corresponding tab
            item.classList.add('active');
            const targetId = item.getAttribute('data-tab');
            const targetTab = document.getElementById(targetId);

            targetTab.classList.remove('hidden-tab');
            targetTab.classList.add('active-tab');

            // Update Header Title
            if (tabInfo[targetId]) {
                pageTitle.textContent = tabInfo[targetId].title;
                pageDesc.textContent = tabInfo[targetId].desc;
            }
        });
    });

    // --- Prediction Logic ---
    const form = document.getElementById('prediction-form');
    const submitBtn = document.getElementById('predict-btn');
    const loadingDiv = document.getElementById('loading');
    const resultsPanel = document.getElementById('results-panel');
    const resetBtn = document.getElementById('reset-btn');

    // Results elements
    const riskBadge = document.getElementById('risk-badge');
    const predictionText = document.getElementById('prediction-text');
    const probPercentage = document.getElementById('prob-percentage');
    const probFill = document.getElementById('prob-fill');

    // Backend API URL
    const API_URL = '/predict';

    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        // Hide previous results and show loading
        resultsPanel.classList.add('hidden');
        submitBtn.classList.add('hidden');
        loadingDiv.classList.remove('hidden');

        // Gather form data
        const formData = new FormData(form);
        const patientData = {};

        for (let [key, value] of formData.entries()) {
            patientData[key] = parseFloat(value);
        }

        try {
            // Make API request
            const response = await fetch(API_URL, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(patientData),
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Prediction failed');
            }

            const result = await response.json();
            displayResults(result);

        } catch (error) {
            console.error('Error:', error);
            alert(`Error running prediction: ${error.message}`);
            // Reset UI
            submitBtn.classList.remove('hidden');
            loadingDiv.classList.add('hidden');
        }
    });

    function displayResults(data) {
        // Hide loading, show results
        loadingDiv.classList.add('hidden');
        submitBtn.classList.remove('hidden');
        resultsPanel.classList.remove('hidden');

        // Set Risk Badge
        riskBadge.textContent = data.Risk_Level;
        riskBadge.className = 'risk-badge'; // Reset classes

        if (data.Risk_Level.includes('HIGH')) {
            riskBadge.classList.add('high');
        } else if (data.Risk_Level.includes('MODERATE')) {
            riskBadge.classList.add('mod');
        } else {
            riskBadge.classList.add('low');
        }

        // Set Prediction Text
        if (data.Prediction === 'Heart Disease') {
            predictionText.innerHTML = `<span style="color: var(--risk-high); font-weight: 600;">Positive Indication</span> for Heart Disease detected. Clinical follow-up recommended.`;
        } else {
            predictionText.innerHTML = `<span style="color: var(--risk-low); font-weight: 600;">Negative Indication</span>. No significant signs of heart disease detected.`;
        }

        // Animate Probability Bar
        const probVal = (data.Probability * 100).toFixed(1);
        probPercentage.textContent = `${probVal}%`;

        // Slight delay for CSS transition to trigger
        setTimeout(() => {
            probFill.style.width = `${probVal}%`;

            // Color gradient based on probability
            if (probVal < 40) {
                probFill.style.background = 'linear-gradient(90deg, #10B981, #34D399)';
            } else if (probVal < 60) {
                probFill.style.background = 'linear-gradient(90deg, #F59E0B, #FBBF24)';
            } else {
                probFill.style.background = 'linear-gradient(90deg, #EF4444, #F87171)';
            }
        }, 100);
    }

    resetBtn.addEventListener('click', () => {
        resultsPanel.classList.add('hidden');
        form.reset();
        probFill.style.width = '0%';
        window.scrollTo({ top: 0, behavior: 'smooth' });
    });
});
