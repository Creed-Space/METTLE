/**
 * METTLE - Web UI Application
 * Interactive AI verification interface
 */

const API_BASE = '/api';

// State
let state = {
    sessionId: null,
    currentChallenge: null,
    challengeStartTime: null,
    totalChallenges: 0,
    completedChallenges: 0,
    timerInterval: null,
};

// DOM Elements
const screens = {
    start: document.getElementById('start-screen'),
    challenge: document.getElementById('challenge-screen'),
    result: document.getElementById('result-screen'),
    error: document.getElementById('error-screen'),
};

const elements = {
    entityId: document.getElementById('entity-id'),
    difficulty: document.getElementById('difficulty'),
    startBtn: document.getElementById('start-btn'),
    progressFill: document.getElementById('progress-fill'),
    progressText: document.getElementById('progress-text'),
    timer: document.getElementById('timer'),
    challengeType: document.getElementById('challenge-type'),
    challengePrompt: document.getElementById('challenge-prompt'),
    timeLimit: document.getElementById('time-limit'),
    answerInput: document.getElementById('answer-input'),
    submitBtn: document.getElementById('submit-btn'),
    feedback: document.getElementById('feedback'),
    resultIcon: document.getElementById('result-icon'),
    resultTitle: document.getElementById('result-title'),
    resultMessage: document.getElementById('result-message'),
    statPassed: document.getElementById('stat-passed'),
    statTotal: document.getElementById('stat-total'),
    statRate: document.getElementById('stat-rate'),
    badgeContainer: document.getElementById('badge-container'),
    badge: document.getElementById('badge'),
    resultsDetail: document.getElementById('results-detail'),
    restartBtn: document.getElementById('restart-btn'),
    errorMessage: document.getElementById('error-message'),
    errorRestartBtn: document.getElementById('error-restart-btn'),
};

// Screen Management
function showScreen(screenName) {
    Object.values(screens).forEach(screen => screen.classList.remove('active'));
    screens[screenName].classList.add('active');
}

// API Calls
async function apiCall(endpoint, method = 'GET', body = null) {
    const options = {
        method,
        headers: {
            'Content-Type': 'application/json',
        },
    };

    if (body) {
        options.body = JSON.stringify(body);
    }

    const response = await fetch(`${API_BASE}${endpoint}`, options);
    const data = await response.json();

    if (!response.ok) {
        throw new Error(data.detail || 'API request failed');
    }

    return data;
}

// Timer
function startTimer(timeLimitMs) {
    const startTime = Date.now();
    state.challengeStartTime = startTime;

    const updateTimer = () => {
        const elapsed = Date.now() - startTime;
        const remaining = Math.max(0, timeLimitMs - elapsed);
        const seconds = (remaining / 1000).toFixed(1);

        elements.timer.textContent = `${seconds}s`;

        // Update timer color based on remaining time
        elements.timer.classList.remove('warning', 'danger');
        if (remaining < timeLimitMs * 0.25) {
            elements.timer.classList.add('danger');
        } else if (remaining < timeLimitMs * 0.5) {
            elements.timer.classList.add('warning');
        }

        if (remaining <= 0) {
            clearInterval(state.timerInterval);
        }
    };

    updateTimer();
    state.timerInterval = setInterval(updateTimer, 100);
}

function stopTimer() {
    if (state.timerInterval) {
        clearInterval(state.timerInterval);
        state.timerInterval = null;
    }
}

// Progress
function updateProgress() {
    const progress = (state.completedChallenges / state.totalChallenges) * 100;
    elements.progressFill.style.width = `${progress}%`;
    elements.progressText.textContent = `Challenge ${state.completedChallenges + 1} of ${state.totalChallenges}`;
}

// Challenge Display
function displayChallenge(challenge) {
    state.currentChallenge = challenge;

    // Format challenge type for display
    const typeDisplay = challenge.type.replace(/_/g, ' ');
    elements.challengeType.textContent = typeDisplay;

    // Display prompt
    elements.challengePrompt.textContent = challenge.prompt;

    // Display time limit
    const seconds = (challenge.time_limit_ms / 1000).toFixed(1);
    elements.timeLimit.textContent = `Time limit: ${seconds}s`;

    // Clear input and focus
    elements.answerInput.value = '';
    elements.answerInput.focus();

    // Hide feedback
    elements.feedback.classList.add('hidden');

    // Start timer
    startTimer(challenge.time_limit_ms);

    // Enable submit button
    elements.submitBtn.disabled = false;
    elements.submitBtn.classList.remove('loading');
}

// Feedback Display
function showFeedback(passed, message) {
    elements.feedback.textContent = message;
    elements.feedback.className = `feedback ${passed ? 'success' : 'error'}`;
}

// Create a result item element safely using DOM methods
function createResultItem(result) {
    const item = document.createElement('div');
    item.className = 'result-item';

    const typeSpan = document.createElement('span');
    typeSpan.className = 'type';
    typeSpan.textContent = result.challenge_type.replace(/_/g, ' ');

    const timeSpan = document.createElement('span');
    timeSpan.className = 'time';
    timeSpan.textContent = `${result.response_time_ms}ms / ${result.time_limit_ms}ms`;

    const statusSpan = document.createElement('span');
    statusSpan.className = `status ${result.passed ? 'pass' : 'fail'}`;
    statusSpan.textContent = result.passed ? 'PASS' : 'FAIL';

    item.appendChild(typeSpan);
    item.appendChild(timeSpan);
    item.appendChild(statusSpan);

    return item;
}

/**
 * Set the result icon using safe DOM methods (no innerHTML).
 * Creates a Font Awesome <i> element and replaces the container's children.
 */
function setResultIcon(iconClass) {
    const icon = document.createElement('i');
    icon.className = iconClass;
    icon.setAttribute('aria-hidden', 'true');
    elements.resultIcon.replaceChildren(icon);
}

// Result Display
function displayResult(result) {
    stopTimer();

    // Icon and title (using safe DOM methods)
    if (result.verified) {
        setResultIcon('fa-solid fa-medal');
        elements.resultTitle.textContent = 'Verification Successful!';
        elements.resultMessage.textContent = 'You have proven your metal.';
    } else {
        setResultIcon('fa-solid fa-circle-xmark');
        elements.resultTitle.textContent = 'Verification Failed';
        elements.resultMessage.textContent = 'You did not meet the 80% threshold.';
    }

    // Stats
    elements.statPassed.textContent = result.passed;
    elements.statTotal.textContent = result.total;
    elements.statRate.textContent = `${Math.round(result.pass_rate * 100)}%`;

    // Badge
    if (result.badge) {
        elements.badge.textContent = result.badge;
        elements.badgeContainer.classList.remove('hidden');
    } else {
        elements.badgeContainer.classList.add('hidden');
    }

    // Detailed results - using safe DOM methods
    elements.resultsDetail.replaceChildren(); // Clear existing children
    result.results.forEach(r => {
        elements.resultsDetail.appendChild(createResultItem(r));
    });

    showScreen('result');
}

// Error Display
function showError(message) {
    stopTimer();
    elements.errorMessage.textContent = message;
    showScreen('error');
}

// Event Handlers
async function handleStart() {
    const entityId = elements.entityId.value.trim() || null;
    const difficulty = elements.difficulty.value;

    elements.startBtn.disabled = true;
    elements.startBtn.classList.add('loading');

    try {
        const data = await apiCall('/session/start', 'POST', {
            difficulty,
            entity_id: entityId,
        });

        state.sessionId = data.session_id;
        state.totalChallenges = data.total_challenges;
        state.completedChallenges = 0;

        updateProgress();
        displayChallenge(data.current_challenge);
        showScreen('challenge');

    } catch (error) {
        showError(error.message);
    } finally {
        elements.startBtn.disabled = false;
        elements.startBtn.classList.remove('loading');
    }
}

async function handleSubmit() {
    if (!state.currentChallenge || !state.sessionId) return;

    const answer = elements.answerInput.value;

    elements.submitBtn.disabled = true;
    elements.submitBtn.classList.add('loading');
    stopTimer();

    try {
        const data = await apiCall('/session/answer', 'POST', {
            session_id: state.sessionId,
            challenge_id: state.currentChallenge.id,
            answer,
        });

        // Show feedback briefly
        const result = data.result;
        const message = result.passed
            ? `Correct! (${result.response_time_ms}ms)`
            : `Incorrect. ${result.details.time_ok ? '' : 'Too slow!'}`;
        showFeedback(result.passed, message);

        state.completedChallenges++;
        updateProgress();

        if (data.session_complete) {
            // Small delay to show last feedback, then show results
            setTimeout(async () => {
                const finalResult = await apiCall(`/session/${state.sessionId}/result`);
                displayResult(finalResult);
            }, 800);
        } else {
            // Show next challenge after brief delay
            setTimeout(() => {
                displayChallenge(data.next_challenge);
            }, 800);
        }

    } catch (error) {
        showError(error.message);
    }
}

function handleRestart() {
    // Reset state
    state = {
        sessionId: null,
        currentChallenge: null,
        challengeStartTime: null,
        totalChallenges: 0,
        completedChallenges: 0,
        timerInterval: null,
    };

    stopTimer();
    showScreen('start');
}

// Event Listeners
elements.startBtn.addEventListener('click', handleStart);
elements.submitBtn.addEventListener('click', handleSubmit);
elements.restartBtn.addEventListener('click', handleRestart);
elements.errorRestartBtn.addEventListener('click', handleRestart);

// Allow Enter key to submit answer
elements.answerInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !elements.submitBtn.disabled) {
        handleSubmit();
    }
});

// Allow Enter on start screen
elements.entityId.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
        handleStart();
    }
});

// Initialize
showScreen('start');
console.log('METTLE UI initialized');


/* =============================================
   Landing Page Enhancements
   ============================================= */

/**
 * Scroll Reveal - IntersectionObserver for .scroll-reveal elements
 * Fades in from below when entering the viewport
 */
(function initScrollReveal() {
    const revealElements = document.querySelectorAll('.scroll-reveal');
    if (!revealElements.length) return;

    // Check for reduced motion preference
    const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (prefersReducedMotion) {
        revealElements.forEach(el => el.classList.add('visible'));
        return;
    }

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('visible');
                observer.unobserve(entry.target);
            }
        });
    }, {
        threshold: 0.1,
        rootMargin: '0px 0px -40px 0px'
    });

    revealElements.forEach(el => observer.observe(el));
})();

/**
 * Typewriter Effect for the hero section
 * Cycles through lines, typing and deleting
 */
(function initTypewriter() {
    const container = document.getElementById('hero-typewriter');
    if (!container) return;

    const textEl = container.querySelector('.typewriter-text');
    if (!textEl) return;

    const lines = [
        'AI + FREE + OWNS MISSION + GENUINE + SAFE + THINKS',
        'not what you know \u2014 how you think',
        '10 suites \u00b7 30+ challenge types \u00b7 every session unique',
        'the only way to pass is to actually reason',
    ];

    const TYPING_SPEED = 35;       // ms per character
    const PAUSE_DURATION = 2400;   // ms to pause after typing a line
    const DELETE_SPEED = 18;       // ms per character when deleting

    let lineIndex = 0;
    let charIndex = 0;
    let isDeleting = false;

    function tick() {
        const currentLine = lines[lineIndex];

        if (!isDeleting) {
            // Typing
            charIndex++;
            textEl.textContent = currentLine.substring(0, charIndex);

            if (charIndex >= currentLine.length) {
                // Finished typing this line, pause then start deleting
                isDeleting = true;
                setTimeout(tick, PAUSE_DURATION);
                return;
            }
            setTimeout(tick, TYPING_SPEED);
        } else {
            // Deleting
            charIndex--;
            textEl.textContent = currentLine.substring(0, charIndex);

            if (charIndex <= 0) {
                // Finished deleting, move to next line
                isDeleting = false;
                lineIndex = (lineIndex + 1) % lines.length;
                setTimeout(tick, TYPING_SPEED * 4);
                return;
            }
            setTimeout(tick, DELETE_SPEED);
        }
    }

    // Start after a short delay to let the page load
    setTimeout(tick, 1200);
})();

/**
 * Auto-update copyright year
 */
(function updateCopyrightYear() {
    const yearEl = document.getElementById('copyright-year');
    if (yearEl) {
        yearEl.textContent = new Date().getFullYear();
    }
})();
