// Configuration file for the Office Ordering System
// Replace 'YOUR_GEMINI_API_KEY' with your actual Gemini API key

const CONFIG = {
    // Gemini AI API Configuration
    GEMINI_API_KEY: 'AIzaSyDYZa6wjHyKIMJRZTk96ta-bDVm4oXdkIA', // Get from: https://makersuite.google.com/app/apikey
    GEMINI_API_URL: 'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent',
    
    // Admin Configuration
    ADMIN_NAME: 'Kamran',
    
    // Server Configuration (if needed)
    SERVER_PORT: 8000
};

// Export for use in other files
if (typeof module !== 'undefined' && module.exports) {
    module.exports = CONFIG;
}
