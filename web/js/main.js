// Main JavaScript for Cyberpunk UI

const API_BASE = 'http://localhost:8000/api';

// DOM Elements
const uploadArea = document.getElementById('uploadArea');
const imageInput = document.getElementById('imageInput');
const imagePreview = document.getElementById('imagePreview');
const previewImg = document.getElementById('previewImg');
const generateBtn = document.getElementById('generateBtn');
const captionDisplay = document.getElementById('captionDisplay');
const modelButtons = document.querySelectorAll('.model-btn');
const strategySlider = document.getElementById('strategySlider');
const ciderScore = document.getElementById('ciderScore');
const ciderProgress = document.getElementById('ciderProgress');
const xraySection = document.getElementById('xraySection');
const attentionCanvas = document.getElementById('attentionCanvas');

let currentModel = 'cnn_gru';
let currentStrategy = 'greedy';
let currentImage = null;
let currentCaption = '';

// Initialize Lottie Animation
function initLottie() {
    // 使用简单的网络节点动画（如果没有Lottie文件，使用CSS动画替代）
    const lottieBg = document.getElementById('lottie-bg');
    // 这里可以加载Lottie JSON文件
    // lottie.loadAnimation({...});
}

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
    formData.append('temperature', '1.0');
    
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
function updateMetrics(score) {
    // Animate score
    gsap.to({value: 0}, {
        value: score,
        duration: 1,
        ease: 'power2.out',
        onUpdate: function() {
            ciderScore.textContent = this.targets()[0].value.toFixed(4);
            ciderProgress.style.width = `${(this.targets()[0].value * 100)}%`;
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

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    initLottie();
    console.log('DeepFashion AI Workbench initialized');
});

