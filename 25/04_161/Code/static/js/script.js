

document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const fileInput = document.getElementById('file-input');
    const uploadArea = document.getElementById('upload-area');
    const originalImg = document.getElementById('original-img');
    const maskImg = document.getElementById('mask-img');
    const statusDiv = document.getElementById('status');
    const loadingDiv = document.getElementById('loading');
    const resultsContainer = document.getElementById('results-container');
    const processingTimeEl = document.getElementById('processing-time');
    const uploadContent = document.querySelector('.upload-content');
    const uploadIcon = document.querySelector('.upload-icon svg');

    // Initial state
    resultsContainer.style.display = 'none';
    loadingDiv.style.display = 'none';

    // File selection handler
    fileInput.addEventListener('change', function() {
        if (this.files && this.files[0]) {
            processImage(this.files[0]);
        }
    });

    // Drag and drop handlers
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
        uploadIcon.style.transform = 'translateY(-5px)';
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
        uploadIcon.style.transform = 'translateY(0)';
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        uploadIcon.style.transform = 'translateY(0)';
        
        if (e.dataTransfer.files.length) {
            processImage(e.dataTransfer.files[0]);
        }
    });

    // Main processing function
    async function processImage(file) {
        // Validate file
        if (!file.type.match('image.*')) {
            showStatus('Please upload a valid image file (JPEG or PNG)', 'error');
            return;
        }

        if (file.size > 5 * 1024 * 1024) {
            showStatus('File size too large (max 5MB)', 'error');
            return;
        }

        // Show loading state
        loadingDiv.style.display = 'flex';
        resultsContainer.style.display = 'none';
        uploadContent.style.opacity = '0.5';
        showStatus('Processing your image...', 'info');

        const formData = new FormData();
        formData.append('file', file);

        try {
            const startTime = Date.now();
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || 'Server error occurred');
            }

            const data = await response.json();
            const processingTime = ((Date.now() - startTime) / 1000).toFixed(2);

            // Display results
            originalImg.src = data.original + '?t=' + Date.now();
            maskImg.src = data.mask + '?t=' + Date.now();
            
            originalImg.onload = () => {
                resultsContainer.style.display = 'block';
                processingTimeEl.textContent = `${processingTime}s`;
                showStatus('Segmentation completed successfully!', 'success');
                loadingDiv.style.display = 'none';
                uploadContent.style.opacity = '1';
                
                // Smooth scroll to results
                resultsContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            };
            
        } catch (error) {
            console.error('Processing error:', error);
            showStatus(error.message || 'Failed to process image', 'error');
            loadingDiv.style.display = 'none';
            uploadContent.style.opacity = '1';
            resultsContainer.style.display = 'none';
        }
    }

    // Helper function to show status messages
    function showStatus(message, type) {
        statusDiv.textContent = message;
        statusDiv.className = 'status-message ' + type;
    }
});