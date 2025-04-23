// Configuration
const API_URL = 'http://localhost:8000';
let selectedFile = null;
let currentJobId = null;
let statusCheckInterval = null;

// DOM elements
const dropArea = document.getElementById('drop-area');
const fileInput = document.getElementById('file-input');
const fileDetails = document.getElementById('file-details');
const fileName = document.getElementById('file-name');
const fileSize = document.getElementById('file-size');
const removeFileBtn = document.getElementById('remove-file');
const analyzeBtn = document.getElementById('analyze-button');
const uploadContainer = document.getElementById('upload-container');
const processingContainer = document.getElementById('processing-container');
const resultsContainer = document.getElementById('results-container');
const progressBar = document.getElementById('progress-bar');
const progressMessage = document.getElementById('progress-message');
const bookTitle = document.getElementById('book-title');
const bookGenre = document.getElementById('book-genre');
const errorModal = new bootstrap.Modal(document.getElementById('error-modal'));
const errorMessage = document.getElementById('error-message');
const downloadAllBtn = document.getElementById('download-all');
const analyzeAnotherBtn = document.getElementById('analyze-another');

// Event Listeners
document.addEventListener('DOMContentLoaded', () => {
    // Drag and drop handlers
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
    removeFileBtn.addEventListener('click', removeFile, false);
    analyzeBtn.addEventListener('click', analyzeBook, false);
    
    // Download handlers
    document.querySelectorAll('.download-vis').forEach(btn => {
        btn.addEventListener('click', downloadVisualization);
    });
    
    downloadAllBtn.addEventListener('click', downloadAllVisualizations);
    
    // Analyze another book handler
    analyzeAnotherBtn.addEventListener('click', resetAndAnalyzeAnother);
});

// Helper Functions
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

function handleDrop(e) {
    const dt = e.dataTransfer;
    const files = dt.files;
    
    if (files.length > 0) {
        handleFile(files[0]);
    }
}

function handleFileSelect(e) {
    const files = e.target.files;
    
    if (files.length > 0) {
        handleFile(files[0]);
    }
}

function handleFile(file) {
    // Check file type
    const fileExt = file.name.split('.').pop().toLowerCase();
    if (!['txt', 'pdf'].includes(fileExt)) {
        showError('Unsupported file format. Please upload a .txt or .pdf file.');
        return;
    }
    
    // Check file size (max 20MB)
    if (file.size > 20 * 1024 * 1024) {
        showError('File is too large. Maximum file size is 20MB.');
        return;
    }
    
    // Update UI
    selectedFile = file;
    fileName.textContent = file.name;
    fileSize.textContent = `Size: ${formatFileSize(file.size)}`;
    fileDetails.classList.remove('d-none');
    dropArea.classList.add('d-none');
}

function removeFile() {
    selectedFile = null;
    fileInput.value = '';
    fileDetails.classList.add('d-none');
    dropArea.classList.remove('d-none');
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

async function analyzeBook() {
    if (!selectedFile) {
        showError('Please select a file to analyze.');
        return;
    }
    
    try {
        // Prepare form data
        const formData = new FormData();
        formData.append('file', selectedFile);
        
        // Update UI to processing state
        uploadContainer.parentElement.classList.add('d-none');
        processingContainer.classList.remove('d-none');
        progressBar.style.width = '0%';
        progressMessage.textContent = 'Starting analysis...';
        
        // Send the file for analysis
        const response = await fetch(`${API_URL}/api/analyze`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Failed to start analysis');
        }
        
        const data = await response.json();
        currentJobId = data.job_id;
        
        // Start checking status
        startStatusChecking();
        
    } catch (error) {
        showError(`Error: ${error.message}`);
        resetUI();
    }
}

function startStatusChecking() {
    // Clear any existing interval
    if (statusCheckInterval) {
        clearInterval(statusCheckInterval);
    }
    
    // Check status every 2 seconds
    statusCheckInterval = setInterval(checkJobStatus, 2000);
}

async function checkJobStatus() {
    if (!currentJobId) return;
    
    try {
        const response = await fetch(`${API_URL}/api/jobs/${currentJobId}`);
        
        if (!response.ok) {
            throw new Error('Failed to check job status');
        }
        
        const status = await response.json();
        
        // Update progress
        updateProgress(status);
        
        // Check if job is complete or failed
        if (status.status === 'completed') {
            clearInterval(statusCheckInterval);
            await fetchResults();
        } else if (status.status === 'failed') {
            clearInterval(statusCheckInterval);
            showError(`Analysis failed: ${status.message}`);
            resetUI();
        }
        
    } catch (error) {
        console.error('Error checking job status:', error);
    }
}

function updateProgress(status) {
    const progress = status.progress || 0;
    progressBar.style.width = `${progress}%`;
    progressBar.setAttribute('aria-valuenow', progress);
    
    if (status.message) {
        progressMessage.textContent = status.message;
    }
}

async function fetchResults() {
    try {
        // Get analysis results
        const resultsResponse = await fetch(`${API_URL}/api/results/${currentJobId}`);
        
        if (!resultsResponse.ok) {
            throw new Error('Failed to fetch results');
        }
        
        const results = await resultsResponse.json();
        
        // Get visualization data
        const visualizationsResponse = await fetch(`${API_URL}/api/visualizations-data/${currentJobId}`, {
            method: 'POST'
        });
        
        if (!visualizationsResponse.ok) {
            throw new Error('Failed to fetch visualizations');
        }
        
        const visualizationsData = await visualizationsResponse.json();
        
        // Display results
        displayResults(results, visualizationsData.visualizations);
        
        // Save results to recommendations database
        try {
            await fetch(`${API_URL}/api/results/${currentJobId}`, {
                method: 'POST'
            });
            
            // Fetch recommendations
            fetchRecommendations(results.title);
        } catch (recError) {
            console.error('Error with recommendations:', recError);
            showRecommendationsError('Unable to generate recommendations. Not enough books analyzed yet.');
        }
        
    } catch (error) {
        showError(`Error fetching results: ${error.message}`);
        resetUI();
    }
}

function displayResults(results, visualizations) {
    // Update book info
    bookTitle.textContent = results.title;
    bookGenre.textContent = results.genre || 'Unknown';
    
    // Display emotion summary
    displayEmotionSummary(results.overall_emotions);
    
    // Display character emotions summary
    displayCharacterSummary(results.character_emotions);
    
    // Display topic summary
    displayTopicSummary(results.topics);
    
    // Set visualization images
    if (visualizations) {
        if (visualizations.polarity) {
            document.getElementById('polarity-vis').src = `data:image/png;base64,${visualizations.polarity}`;
        }
        
        if (visualizations.emotions) {
            document.getElementById('emotions-vis').src = `data:image/png;base64,${visualizations.emotions}`;
        }
        
        if (visualizations.emotion_composition) {
            document.getElementById('emotion-composition-vis').src = `data:image/png;base64,${visualizations.emotion_composition}`;
        }
        
        if (visualizations.character_emotions) {
            document.getElementById('character-emotions-vis').src = `data:image/png;base64,${visualizations.character_emotions}`;
        }
        
        if (visualizations.topic_heatmap) {
            document.getElementById('topic-heatmap-vis').src = `data:image/png;base64,${visualizations.topic_heatmap}`;
        }
    }
    
    // Show results container
    processingContainer.classList.add('d-none');
    resultsContainer.classList.remove('d-none');
    resultsContainer.classList.add('fade-in');
}

function displayEmotionSummary(emotions) {
    const emotionsList = document.getElementById('emotion-summary-list');
    emotionsList.innerHTML = '';
    
    // Get top 5 emotions
    const topEmotions = Object.entries(emotions)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 5);
    
    // Create list items for each emotion
    topEmotions.forEach(([emotion, value]) => {
        const li = document.createElement('li');
        li.className = 'list-group-item p-1 border-0';
        
        // Only show the emotion name, not the value
        li.innerHTML = `
            <span>- ${capitalizeFirstLetter(emotion)}</span>
        `;
        
        emotionsList.appendChild(li);
    });
}

function displayCharacterSummary(characterEmotions) {
    const characterSummary = document.getElementById('character-summary');
    characterSummary.innerHTML = '';
    
    // Get up to 10 characters
    const characters = Object.keys(characterEmotions).slice(0, 10);
    
    characters.forEach(character => {
        // Get top 3 emotions for this character
        const emotions = characterEmotions[character];
        const topEmotions = Object.entries(emotions)
            .sort((a, b) => b[1] - a[1])
            .slice(0, 3);
        
        const emotionText = topEmotions
            .map(([emotion, value]) => emotion)
            .join(', ');
        
        const characterElem = document.createElement('p');
        characterElem.className = 'mb-1 ps-2';
        characterElem.style.textIndent = '-1em';
        characterElem.style.marginLeft = '1em';
        characterElem.innerHTML = `${character}: ${emotionText}`;
        
        characterSummary.appendChild(characterElem);
    });
}

function displayTopicSummary(topics) {
    const topicSummary = document.getElementById('topic-summary');
    topicSummary.innerHTML = '';
    
    // Get up to 10 topics
    const topicKeys = Object.keys(topics).slice(0, 10);
    
    topicKeys.forEach(topicKey => {
        const words = topics[topicKey].join(', ');
        
        const topicElem = document.createElement('p');
        topicElem.className = 'mb-1 ps-2';
        topicElem.style.textIndent = '-1em';
        topicElem.style.marginLeft = '1em';
        topicElem.innerHTML = `${topicKey}: ${words}`;
        
        topicSummary.appendChild(topicElem);
    });
}

function capitalizeFirstLetter(string) {
    return string.charAt(0).toUpperCase() + string.slice(1);
}

function downloadVisualization(e) {
    const imgId = e.currentTarget.getAttribute('data-img');
    const img = document.getElementById(imgId);
    
    if (img && img.src) {
        // Create a link element
        const link = document.createElement('a');
        link.href = img.src;
        link.download = `${imgId.replace('-vis', '')}.png`;
        
        // Add to document, trigger click, and remove
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    }
}

async function downloadAllVisualizations() {
    try {
        const zip = new JSZip();
        const imgs = document.querySelectorAll('.vis-container img');
        
        // Add each visualization to the zip
        imgs.forEach(img => {
            if (img.src && img.src.startsWith('data:image/png;base64,')) {
                const base64 = img.src.replace('data:image/png;base64,', '');
                const filename = `${img.id.replace('-vis', '')}.png`;
                zip.file(filename, base64, {base64: true});
            }
        });
        
        // Generate and download the zip
        const content = await zip.generateAsync({type: 'blob'});
        saveAs(content, `${bookTitle.textContent.replace(/\s+/g, '_')}_visualizations.zip`);
        
    } catch (error) {
        showError(`Error creating download: ${error.message}`);
    }
}

function showError(message) {
    errorMessage.textContent = message;
    errorModal.show();
}

function resetAndAnalyzeAnother() {
    // Reset the file input
    selectedFile = null;
    fileInput.value = '';
    
    // Hide results and show upload section
    resultsContainer.classList.add('d-none');
    uploadContainer.parentElement.classList.remove('d-none');
    
    // Reset file details
    fileDetails.classList.add('d-none');
    dropArea.classList.remove('d-none');
    
    // Clear any status checking
    if (statusCheckInterval) {
        clearInterval(statusCheckInterval);
        statusCheckInterval = null;
    }
    
    // Reset progress indicators
    progressBar.style.width = '0%';
    progressBar.setAttribute('aria-valuenow', 0);
    progressMessage.textContent = 'Starting analysis...';
    
    // Scroll to top
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

function resetUI() {
    processingContainer.classList.add('d-none');
    uploadContainer.parentElement.classList.remove('d-none');
    
    // Clear any status checking
    if (statusCheckInterval) {
        clearInterval(statusCheckInterval);
        statusCheckInterval = null;
    }
}

async function fetchRecommendations(bookTitle) {
    try {
        // Show loading state
        document.getElementById('recommendations-loading').classList.remove('d-none');
        document.getElementById('recommendations-content').classList.add('d-none');
        document.getElementById('recommendations-error').classList.add('d-none');
        
        // Fetch recommendations from API
        const response = await fetch(`${API_URL}/api/recommendations/${encodeURIComponent(bookTitle)}`);
        
        // Handle different response status codes
        if (!response.ok) {
            // Try to parse error as JSON
            try {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Failed to get recommendations');
            } catch (jsonError) {
                // If not JSON or other parsing error
                const textError = await response.text();
                if (textError.includes("Internal Server Error")) {
                    throw new Error("Server error processing recommendations. Please try again later.");
                } else {
                    throw new Error('Not enough books analyzed yet to generate recommendations.');
                }
            }
        }
        
        // Try to parse the response as JSON
        let recommendations;
        try {
            recommendations = await response.json();
        } catch (jsonError) {
            console.error("Failed to parse recommendations JSON:", jsonError);
            throw new Error("Invalid recommendation data received from server.");
        }
        
        // Display recommendations
        displayRecommendations(recommendations);
        
        // Hide loading, show content
        document.getElementById('recommendations-loading').classList.add('d-none');
        document.getElementById('recommendations-content').classList.remove('d-none');
        
    } catch (error) {
        console.error('Error fetching recommendations:', error);
        showRecommendationsError(error.message);
    }
}

function displayRecommendations(recommendations) {
    // Display similar books
    const similarBooksContainer = document.getElementById('similar-books');
    similarBooksContainer.innerHTML = '';
    
    if (recommendations.similar_books && recommendations.similar_books.length > 0) {
        recommendations.similar_books.forEach(([title, score]) => {
            const bookElem = document.createElement('p');
            bookElem.className = 'mb-1 ps-2';
            bookElem.style.textIndent = '-1em';
            bookElem.style.marginLeft = '1em';
            
            const percentage = Math.round(score * 100);
            bookElem.innerHTML = `📘 ${title} (${percentage}% match)`;
            
            similarBooksContainer.appendChild(bookElem);
        });
    } else {
        similarBooksContainer.innerHTML = '<p class="text-muted">No similar books found.</p>';
    }
    
    // Display emotional matches
    const emotionalMatchesContainer = document.getElementById('emotional-matches');
    emotionalMatchesContainer.innerHTML = '';
    
    if (recommendations.emotional_matches && recommendations.emotional_matches.length > 0) {
        recommendations.emotional_matches.forEach(emotionGroup => {
            const emotionTitle = document.createElement('p');
            emotionTitle.className = 'fw-bold mb-1 mt-2';
            emotionTitle.textContent = `${capitalizeFirstLetter(emotionGroup.emotion)}:`;
            emotionalMatchesContainer.appendChild(emotionTitle);
            
            emotionGroup.matches.forEach(([title, score]) => {
                const matchElem = document.createElement('p');
                matchElem.className = 'mb-1 ps-2';
                matchElem.style.textIndent = '-1em';
                matchElem.style.marginLeft = '1em';
                
                // Choose emoji based on emotion
                let emoji = '😊';
                if (emotionGroup.emotion === 'sadness') emoji = '😢';
                if (emotionGroup.emotion === 'fear') emoji = '😨';
                if (emotionGroup.emotion === 'anger') emoji = '😠';
                if (emotionGroup.emotion === 'love') emoji = '❤️';
                if (emotionGroup.emotion === 'surprise') emoji = '😲';
                
                matchElem.innerHTML = `${emoji} ${title} (${score.toFixed(2)})`;
                emotionalMatchesContainer.appendChild(matchElem);
            });
        });
    } else {
        emotionalMatchesContainer.innerHTML = '<p class="text-muted">No emotional matches found.</p>';
    }
    
    // Display genre clusters
    const genreClustersContainer = document.getElementById('genre-clusters');
    genreClustersContainer.innerHTML = '';
    
    const clusters = recommendations.genre_clusters;
    if (clusters && Object.keys(clusters).length > 0) {
        for (const [clusterId, books] of Object.entries(clusters)) {
            const clusterTitle = document.createElement('p');
            clusterTitle.className = 'fw-bold mb-1 mt-2';
            clusterTitle.textContent = `Cluster ${clusterId}:`;
            genreClustersContainer.appendChild(clusterTitle);
            
            books.forEach(book => {
                const bookElem = document.createElement('p');
                bookElem.className = 'mb-1 ps-2';
                bookElem.style.textIndent = '-1em';
                bookElem.style.marginLeft = '1em';
                bookElem.innerHTML = `- ${book.title}`;
                genreClustersContainer.appendChild(bookElem);
            });
        }
    } else {
        genreClustersContainer.innerHTML = '<p class="text-muted">No genre clusters found.</p>';
    }
}

function showRecommendationsError(message) {
    // Hide loading and content, show error
    document.getElementById('recommendations-loading').classList.add('d-none');
    document.getElementById('recommendations-content').classList.add('d-none');
    document.getElementById('recommendations-error').classList.remove('d-none');
    
    // Set error message
    document.getElementById('recommendations-error-message').textContent = message;
} 