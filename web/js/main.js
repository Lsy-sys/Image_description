// Main JavaScript for Cyberpunk UI

const API_BASE = 'http://localhost:8000/api';

// DOM Elements
const uploadArea = document.getElementById('uploadArea');
const imageInput = document.getElementById('imageInput');
const imagePreview = document.getElementById('imagePreview');
const previewImg = document.getElementById('previewImg');
const generateBtn = document.getElementById('generateBtn');
const captionDisplay = document.getElementById('captionDisplay');
const modelButtons = document.querySelectorAll('.model-card');
const strategySlider = document.getElementById('strategySlider');
const samplingParams = document.getElementById('samplingParams');
const topKInput = document.getElementById('topKInput');
const tempInput = document.getElementById('tempInput');
const ciderScore = document.getElementById('ciderScore');
const bleuScore = document.getElementById('bleuScore');
const rougeScore = document.getElementById('rougeScore');
const ciderRing = document.getElementById('ciderRing');
const xraySection = document.getElementById('xraySection');
const attentionCanvas = document.getElementById('attentionCanvas');
const monitorSection = document.getElementById('monitorSection');
const mainSection = document.getElementById('mainSection');
const navItems = document.querySelectorAll('.nav-item');

let currentModel = 'cnn_gru';
let currentStrategy = 'greedy';
let currentImage = null;
let currentCaption = '';
let trainingChart = null;

// Initialize Lottie Animation
function initLottie() {
    // 使用简单的网络节点动画（如果没有Lottie文件，使用CSS动画替代）
    const lottieBg = document.getElementById('lottie-bg');
    // 这里可以加载Lottie JSON文件
    // lottie.loadAnimation({...});
}

// Navigation between sections
navItems.forEach(btn => {
    btn.addEventListener('click', () => {
        navItems.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');

        const target = btn.dataset.target;
        if (target === 'main') {
            if (mainSection) {
                mainSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
        } else if (target === 'xray') {
            // 显示并滚动到 X-Ray Vision 区域
            if (xraySection) {
                xraySection.classList.remove('hidden');
                xraySection.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
        } else if (target === 'monitor') {
            // 显示并初始化训练监控图表
            if (monitorSection) {
                monitorSection.classList.remove('hidden');
                monitorSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
                initTrainingMonitor();
            }
        }
    });
});

// Upload Area Events
uploadArea.addEventListener('click', () => imageInput.click());

uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
        handleImage(file);
    }
});

imageInput.addEventListener('change', (e) => {
    if (e.target.files[0]) {
        handleImage(e.target.files[0]);
    }
});

// Handle Image
function handleImage(file) {
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImg.src = e.target.result;
        imagePreview.classList.remove('hidden');
        currentImage = file;
        generateBtn.disabled = false;
        
        // Show scanning effect
        const scanningOverlay = document.getElementById('scanningOverlay');
        scanningOverlay.style.display = 'block';
        setTimeout(() => {
            scanningOverlay.style.display = 'none';
        }, 2000);
    };
    reader.readAsDataURL(file);
}

// Model Switcher
modelButtons.forEach(btn => {
    btn.addEventListener('click', () => {
        modelButtons.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        currentModel = btn.dataset.model;
    });
});

// Strategy Slider
strategySlider.addEventListener('input', (e) => {
    const strategies = ['greedy', 'beam_search', 'sampling'];
    currentStrategy = strategies[parseInt(e.target.value)];
    if (currentStrategy === 'sampling') {
        samplingParams.classList.remove('hidden');
    } else {
        samplingParams.classList.add('hidden');
    }
});

// Generate Caption
generateBtn.addEventListener('click', async () => {
    if (!currentImage) return;
    
    generateBtn.disabled = true;
    captionDisplay.innerHTML = '<p class="placeholder-text">Generating caption...</p>';
    
    const formData = new FormData();
    formData.append('file', currentImage);
    formData.append('model_type', currentModel);
    formData.append('strategy', currentStrategy);
    formData.append('max_length', '50');
    formData.append('temperature', tempInput ? tempInput.value || '1.0' : '1.0');
    if (currentStrategy === 'sampling') {
        formData.append('top_k', topKInput ? topKInput.value || '5' : '5');
    }
    
    try {
        const response = await fetch(`${API_BASE}/predict`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error('Generation failed');
        }
        
        const data = await response.json();
        currentCaption = data.caption;
        
        // Typewriter effect
        typewriterEffect(data.caption, data.estimated_cider);
        
        // Show X-Ray section
        xraySection.classList.remove('hidden');
        setupXRayVision(data.caption);
        
    } catch (error) {
        console.error('Error:', error);
        captionDisplay.innerHTML = '<p class="placeholder-text" style="color: #ff006e;">Error generating caption. Please try again.</p>';
    } finally {
        generateBtn.disabled = false;
    }
});

// Typewriter Effect
function typewriterEffect(text, ciderScore) {
    captionDisplay.innerHTML = '';
    const captionText = document.createElement('div');
    captionText.className = 'caption-text';
    captionDisplay.appendChild(captionText);
    
    // Highlight keywords
    const keywords = ['red', 'blue', 'green', 'white', 'black', 'sleeveless', 'long', 'short', 
                     'cotton', 'denim', 'floral', 'solid', 'striped', 'round', 'v-neck'];
    
    let index = 0;
    const words = text.split(' ');
    
    function typeNext() {
        if (index < words.length) {
            const word = words[index];
            const span = document.createElement('span');
            
            if (keywords.some(kw => word.toLowerCase().includes(kw))) {
                span.className = 'keyword';
                span.textContent = word;
            } else {
                span.textContent = word;
            }
            
            captionText.appendChild(span);
            if (index < words.length - 1) {
                captionText.appendChild(document.createTextNode(' '));
            }
            
            index++;
            setTimeout(typeNext, 100);
        } else {
            // Update metrics
            updateMetrics(ciderScore);
        }
    }
    
    typeNext();
}

// Update Metrics
function updateMetrics(ciderVal, bleuVal = null, rougeVal = null) {
    const ciderTarget = ciderVal || 0;
    const bleuTarget = bleuVal || ciderTarget * 0.7;
    const rougeTarget = rougeVal || ciderTarget * 0.6;

    gsap.to({value: 0}, {
        value: ciderTarget,
        duration: 1,
        ease: 'power2.out',
        onUpdate: function() {
            const v = this.targets()[0].value;
            ciderScore.textContent = v.toFixed(4);
            const percent = Math.min(100, Math.max(0, v * 100));
            if (ciderRing) {
                ciderRing.setAttribute('stroke-dasharray', `${percent},100`);
                ciderRing.style.stroke = 'url(#gradRing)';
            }
        }
    });

    gsap.to({value: 0}, {
        value: bleuTarget,
        duration: 1,
        ease: 'power2.out',
        onUpdate: function() {
            bleuScore.textContent = this.targets()[0].value.toFixed(4);
        }
    });

    gsap.to({value: 0}, {
        value: rougeTarget,
        duration: 1,
        ease: 'power2.out',
        onUpdate: function() {
            rougeScore.textContent = this.targets()[0].value.toFixed(4);
        }
    });
}

// Setup X-Ray Vision
function setupXRayVision(caption) {
    const captionWithAttention = document.getElementById('captionWithAttention');
    const words = caption.split(' ');
    
    captionWithAttention.innerHTML = words.map((word, index) => {
        return `<span class="word-clickable" data-word="${word}" data-index="${index}">${word}</span>`;
    }).join(' ');
    
    // Add click handlers
    document.querySelectorAll('.word-clickable').forEach(span => {
        span.addEventListener('click', () => {
            const word = span.dataset.word;
            showAttention(word);
            
            // Update active state
            document.querySelectorAll('.word-clickable').forEach(s => s.classList.remove('active'));
            span.classList.add('active');
        });
    });
}

// Show Attention
function showAttention(word) {
    // Request attention map from API
    if (!currentImage) return;
    
    const formData = new FormData();
    formData.append('file', currentImage);
    formData.append('model_type', currentModel);
    formData.append('word', word);
    
    fetch(`${API_BASE}/attention`, {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        // Draw attention map on canvas
        drawAttentionMap(data.attention_map);
    })
    .catch(error => {
        console.error('Error fetching attention:', error);
    });
}

// Draw Attention Map
function drawAttentionMap(attentionBase64) {
    const canvas = attentionCanvas;
    const ctx = canvas.getContext('2d');
    const img = previewImg;
    
    canvas.width = img.width;
    canvas.height = img.height;
    
    const attentionImg = new Image();
    attentionImg.onload = () => {
        ctx.drawImage(attentionImg, 0, 0, canvas.width, canvas.height);
        canvas.classList.add('active');
    };
    attentionImg.src = `data:image/png;base64,${attentionBase64}`;
}

// Initialize Training Monitor chart (Experiment 5: RL dynamics)
function initTrainingMonitor() {
    if (trainingChart || !window.Chart) return;

    const canvas = document.getElementById('trainingChart');
    if (!canvas) return;

    // 默认查看主力模型 Region-Trans 的训练日志
    const url = `${API_BASE}/training_log?model_type=region_trans`;

    fetch(url)
        .then(res => res.json())
        .then(data => {
            const epochs = data.epochs || [];
            const trainLoss = data.train_loss || [];
            const valLoss = data.val_loss || [];

            if (!epochs.length || !trainLoss.length) {
                console.warn('No training log data found for Training Monitor.');
                return;
            }

            const ctx = canvas.getContext('2d');
            const datasets = [
                {
                    label: 'Train Loss',
                    data: trainLoss,
                    borderColor: '#00f0ff',
                    backgroundColor: 'rgba(0, 240, 255, 0.2)',
                    tension: 0.25,
                },
            ];
            if (valLoss.length === trainLoss.length) {
                datasets.push({
                    label: 'Val Loss',
                    data: valLoss,
                    borderColor: '#ff006e',
                    backgroundColor: 'rgba(255, 0, 110, 0.2)',
                    tension: 0.25,
                });
            }

            trainingChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: epochs,
                    datasets,
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    interaction: {
                        mode: 'index',
                        intersect: false,
                    },
                    plugins: {
                        legend: {
                            labels: { color: '#e5f4ff' },
                        },
                        tooltip: {
                            callbacks: {
                                label: c => ` ${c.dataset.label}: ${c.parsed.y.toFixed(4)}`,
                            },
                        },
                    },
                    scales: {
                        x: {
                            title: { display: true, text: 'Epoch', color: '#9ca3af' },
                            ticks: { color: '#9ca3af' },
                            grid: { color: 'rgba(148, 163, 184, 0.15)' },
                        },
                        y: {
                            title: { display: true, text: 'Loss', color: '#9ca3af' },
                            ticks: { color: '#9ca3af' },
                            grid: { color: 'rgba(148, 163, 184, 0.15)' },
                        },
                    },
                },
            });
        })
        .catch(err => {
            console.error('Failed to load training log:', err);
        });
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    initLottie();
    console.log('DeepFashion AI Workbench initialized');
});


