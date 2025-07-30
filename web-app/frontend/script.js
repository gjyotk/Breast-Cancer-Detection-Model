// DOM Elements
const fileInput = document.getElementById('file-input');
const uploadArea = document.getElementById('upload-area');
const uploadPlaceholder = document.getElementById('upload-placeholder');
const imagePreview = document.getElementById('image-preview');
const previewImg = document.getElementById('preview-img');
const removeBtn = document.getElementById('remove-btn');
const analyzeBtn = document.getElementById('analyze-btn');
const resultsSection = document.getElementById('results-section');
const originalImage = document.getElementById('original-image');
const segmentationImage = document.getElementById('segmentation-image');
const diagnosisValue = document.getElementById('diagnosis-value');
const confidenceBar = document.getElementById('confidence-bar');
const confidenceValue = document.getElementById('confidence-value');
const loadingOverlay = document.getElementById('loading-overlay');

// API Endpoints
const API_URL = 'http://localhost:8000';
const ANALYZE_ENDPOINT = `${API_URL}/analyze`;

// Event Listeners
document.addEventListener('DOMContentLoaded', initializeApp);

function initializeApp() {
    // File upload event handlers
    fileInput.addEventListener('change', handleFileSelect);
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('dragleave', handleDragLeave);
    uploadArea.addEventListener('drop', handleDrop);
    uploadArea.addEventListener('click', () => fileInput.click());
    removeBtn.addEventListener('click', removeImage);
    analyzeBtn.addEventListener('click', analyzeImage);
}

// File Upload Functions
function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file && isValidImageFile(file)) {
        displayImagePreview(file);
    }
}

function handleDragOver(e) {
    e.preventDefault();
    uploadArea.classList.add('active');
}

function handleDragLeave(e) {
    e.preventDefault();
    uploadArea.classList.remove('active');
}

function handleDrop(e) {
    e.preventDefault();
    uploadArea.classList.remove('active');
    
    const file = e.dataTransfer.files[0];
    if (file && isValidImageFile(file)) {
        fileInput.files = e.dataTransfer.files;
        displayImagePreview(file);
    }
}

function isValidImageFile(file) {
    const validTypes = ['image/jpeg', 'image/png', 'image/bmp', 'image/tiff'];
    if (!validTypes.includes(file.type)) {
        alert('Please upload a valid image file (JPEG, PNG, BMP, TIFF)');
        return false;
    }
    return true;
}

function displayImagePreview(file) {
    const reader = new FileReader();
    reader.onload = function(e) {
        previewImg.src = e.target.result;
        uploadPlaceholder.style.display = 'none';
        imagePreview.style.display = 'block';
        analyzeBtn.disabled = false;
    };
    reader.readAsDataURL(file);
}

function removeImage() {
    fileInput.value = '';
    previewImg.src = '';
    uploadPlaceholder.style.display = 'flex';
    imagePreview.style.display = 'none';
    analyzeBtn.disabled = true;
    resultsSection.style.display = 'none';
}

async function analyzeImage() {
    if (!fileInput.files[0]) return;

    // Show loading overlay
    loadingOverlay.style.display = 'flex';
    
    const formData = new FormData();
    const imageFile = fileInput.files[0];
    formData.append('image', imageFile);
    
    // Add the image name to the form data
    formData.append('image_name', imageFile.name);

    try {
        const response = await fetch(ANALYZE_ENDPOINT, {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            throw new Error('Analysis failed');
        }

        const data = await response.json();
        displayResults(data, imageFile);
    } catch (error) {
        console.error('Error analyzing image:', error);
        alert('Error analyzing image. Please try again.');
    } finally {
        // Hide loading overlay
        loadingOverlay.style.display = 'none';
    }
}

function displayResults(data, originalFile) {
    // Display original image
    originalImage.src = URL.createObjectURL(originalFile);
    
    // Display segmentation image
    segmentationImage.src = `data:image/png;base64,${data.segmentation_image}`;
    
    // Display classification results
    diagnosisValue.textContent = data.diagnosis;
    diagnosisValue.style.color = data.diagnosis === 'Malignant' ? '#e74c3c' : '#2ecc71';
    
    // Update confidence bar
    const confidencePercentage = data.confidence * 100;
    confidenceBar.style.width = `${confidencePercentage}%`;
    confidenceValue.textContent = `${confidencePercentage}%`;
    
    // Show results section
    resultsSection.style.display = 'block';
    
    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth' });
}