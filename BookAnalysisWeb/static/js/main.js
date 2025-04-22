document.addEventListener('DOMContentLoaded', function() {
    // Elements
    const dropArea = document.getElementById('drop-area');
    const fileInput = document.getElementById('file-input');
    const uploadBtn = document.getElementById('upload-btn');
    const uploadSection = document.getElementById('upload-section');
    const progressSection = document.getElementById('progress-section');
    const progressBar = document.getElementById('progress-bar');
    const progressMessage = document.getElementById('progress-message');
    const resultsSection = document.getElementById('results-section');
    const bookTitle = document.getElementById('book-title');
    const newAnalysisBtn = document.getElementById('new-analysis-btn');
    
    // Visualization elements
    const polarityImg = document.getElementById('polarity-img');
    const emotionImg = document.getElementById('emotion-img');
    const characterImg = document.getElementById('character-img');
    const topicImg = document.getElementById('topic-img');
    const topicsList = document.getElementById('topics-list');
    
    // Download links
    const polarityDownload = document.getElementById('polarity-download');
    const emotionDownload = document.getElementById('emotion-download');
    const characterDownload = document.getElementById('character-download');
    const topicDownload = document.getElementById('topic-download');
    
    // Current analysis state
    let currentFile = null;
    let currentAnalysisId = null;
    let progressInterval = null;

    // Event listeners for drag and drop
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, preventDefaults, false);
    });
    
    ['dragenter', 'dragover'].forEach(eventName => {
        dropArea.addEventListener(eventName, highlight, false);
    });
    
    ['dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, unhighlight, false);
    });
    
    dropArea.addEventListener('drop', handleDrop, false);
    fileInput.addEventListener('change', handleFileSelect, false);
    uploadBtn.addEventListener('click', uploadFile);
    newAnalysisBtn.addEventListener('click', resetUI);
    
    // Basic event handlers
    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }
    
    function highlight() {
        dropArea.classList.add('highlight');
    }
    
    function unhighlight() {
        dropArea.classList.remove('highlight');
    }
    
    // Handle file selection
    function handleFileSelect(e) {
        const file = e.target.files[0];
        if (file) {
            if (validateFile(file)) {
                currentFile = file;
                uploadBtn.disabled = false;
                dropArea.innerHTML = `<p>Selected: <strong>${file.name}</strong> (${formatFileSize(file.size)})</p>`;
            } else {
                alert('Please select a valid text file (.txt)');
                fileInput.value = '';
                uploadBtn.disabled = true;
            }
        }
    }
    
    // Handle file drop
    function handleDrop(e) {
        const file = e.dataTransfer.files[0];
        if (file) {
            if (validateFile(file)) {
                currentFile = file;
                uploadBtn.disabled = false;
                dropArea.innerHTML = `<p>Selected: <strong>${file.name}</strong> (${formatFileSize(file.size)})</p>`;
            } else {
                alert('Please select a valid text file (.txt)');
                uploadBtn.disabled = true;
            }
        }
    }
    
    // Validate file
    function validateFile(file) {
        return file.type === 'text/plain' || file.name.endsWith('.txt');
    }
    
    // Format file size
    function formatFileSize(bytes) {
        if (bytes < 1024) return bytes + ' bytes';
        else if (bytes < 1048576) return (bytes / 1024).toFixed(1) + ' KB';
        else return (bytes / 1048576).toFixed(1) + ' MB';
    }
    
    // Upload and analyze file
    function uploadFile() {
        if (!currentFile) return;
        
        const formData = new FormData();
        formData.append('file', currentFile);
        
        // Show progress section
        uploadSection.style.display = 'none';
        progressSection.style.display = 'block';
        
        // Reset progress
        progressBar.style.width = '0%';
        progressMessage.textContent = 'Starting analysis...';
        
        // Upload the file
        fetch('/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                currentAnalysisId = data.analysis_id;
                startProgressMonitoring();
            } else {
                showError(data.error || 'An error occurred during upload');
            }
        })
        .catch(error => {
            showError('Upload failed: ' + error.message);
        });
    }
    
    // Monitor analysis progress
    function startProgressMonitoring() {
        progressInterval = setInterval(checkProgress, 1000);
    }
    
    function checkProgress() {
        if (!currentAnalysisId) return;
        
        fetch(`/progress/${currentAnalysisId}`)
            .then(response => response.json())
            .then(data => {
                // Update progress
                progressBar.style.width = `${data.progress}%`;
                progressMessage.textContent = data.message;
                
                // Check if complete or failed
                if (data.status === 'completed') {
                    clearInterval(progressInterval);
                    displayResults();
                } else if (data.status === 'failed') {
                    clearInterval(progressInterval);
                    showError('Analysis failed: ' + data.message);
                }
            })
            .catch(error => {
                clearInterval(progressInterval);
                showError('Failed to check progress: ' + error.message);
            });
    }
    
    // Display analysis results
    function displayResults() {
        if (!currentAnalysisId) return;
        
        fetch(`/results/${currentAnalysisId}`)
            .then(response => response.json())
            .then(data => {
                if (data.status === 'success') {
                    // Hide progress section
                    progressSection.style.display = 'none';
                    
                    // Update book title
                    bookTitle.textContent = data.profile.title;
                    
                    // Set visualization images
                    setVisualizationImages(data.visualizations);
                    
                    // Display topic information
                    displayTopics(data.profile.topics);
                    
                    // Show results section
                    resultsSection.style.display = 'block';
                } else {
                    showError(data.error || 'Failed to get analysis results');
                }
            })
            .catch(error => {
                showError('Error displaying results: ' + error.message);
            });
    }
    
    // Set visualization images and download links
    function setVisualizationImages(visualizations) {
        for (const file of visualizations) {
            const url = `/results/${currentAnalysisId}/visualization/${file}`;
            
            if (file.includes('polarity')) {
                polarityImg.src = url;
                polarityDownload.href = url;
                polarityDownload.download = file;
            } else if (file.includes('emotion')) {
                emotionImg.src = url;
                emotionDownload.href = url;
                emotionDownload.download = file;
            } else if (file.includes('character')) {
                characterImg.src = url;
                characterDownload.href = url;
                characterDownload.download = file;
            } else if (file.includes('topic')) {
                topicImg.src = url;
                topicDownload.href = url;
                topicDownload.download = file;
            }
        }
    }
    
    // Display topics
    function displayTopics(topics) {
        topicsList.innerHTML = '';
        
        for (const [topicId, wordList] of Object.entries(topics)) {
            const topicNumber = topicId.split('_')[1];
            const topicItem = document.createElement('div');
            topicItem.className = 'topic-item';
            topicItem.innerHTML = `
                <h5>Topic ${topicNumber}</h5>
                <p class="topic-words">${wordList.join(', ')}</p>
            `;
            topicsList.appendChild(topicItem);
        }
    }
    
    // Show error message
    function showError(message) {
        progressSection.style.display = 'block';
        progressBar.style.width = '100%';
        progressBar.classList.remove('bg-primary');
        progressBar.classList.add('bg-danger');
        progressMessage.textContent = message;
        
        clearInterval(progressInterval);
    }
    
    // Reset UI for new analysis
    function resetUI() {
        // Clear variables
        currentFile = null;
        currentAnalysisId = null;
        
        // Reset file input
        fileInput.value = '';
        uploadBtn.disabled = true;
        
        // Reset drop area
        dropArea.innerHTML = `
            <p>Drag & drop your book file or click to select</p>
            <input type="file" id="file-input" accept=".txt" class="file-input">
            <label for="file-input" class="file-label">Choose a File</label>
        `;
        
        // Reassign event listener to new file input
        document.getElementById('file-input').addEventListener('change', handleFileSelect, false);
        
        // Reset progress
        progressBar.style.width = '0%';
        progressBar.classList.remove('bg-danger');
        progressBar.classList.add('bg-primary');
        
        // Show upload section, hide others
        uploadSection.style.display = 'block';
        progressSection.style.display = 'none';
        resultsSection.style.display = 'none';
    }
}); 