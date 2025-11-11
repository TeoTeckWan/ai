// Configuration
const API_BASE_URL = 'http://localhost:8111';
const STORAGE_KEY = 'agentic_chat_history';
const THREAD_ID_KEY = 'agentic_chat_thread_id';

// Command definitions
const COMMANDS = [
    { command: '/calc', description: 'Calculate arithmetic expressions', example: '/calc 15 * 23 + 100' },
    { command: '/products', description: 'Search for products', example: '/products show me tumblers' },
    { command: '/outlets', description: 'Find outlet locations', example: '/outlets where is SS 2?' },
    { command: '/reset', description: 'Reset conversation and clear history', example: '/reset' }
];

// State management
let conversationHistory = [];
let threadId = null;
let isProcessing = false;

// DOM elements
const welcomeScreen = document.getElementById('welcomeScreen');
const messagesContainer = document.getElementById('messagesContainer');
const messageInput = document.getElementById('messageInput');
const sendBtn = document.getElementById('sendBtn');
const resetBtn = document.getElementById('resetBtn');
const autocompleteSuggestions = document.getElementById('autocompleteSuggestions');

// Initialize app
function init() {
    loadFromLocalStorage();
    attachEventListeners();
    adjustTextareaHeight();
    
    if (conversationHistory.length > 0) {
        showChatInterface();
        renderConversationHistory();
    }
}

// Event listeners
function attachEventListeners() {
    sendBtn.addEventListener('click', handleSendMessage);
    resetBtn.addEventListener('click', handleReset);
    
    messageInput.addEventListener('keydown', handleKeyDown);
    messageInput.addEventListener('input', handleInput);
    
    // Click outside to close autocomplete
    document.addEventListener('click', (e) => {
        if (!messageInput.contains(e.target) && !autocompleteSuggestions.contains(e.target)) {
            hideAutocomplete();
        }
    });
    
    // Quick action buttons
    document.querySelectorAll('.quick-action-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const action = btn.dataset.action;
            messageInput.value = action;
            handleSendMessage();
        });
    });
}

// Handle keyboard events
function handleKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleSendMessage();
    }
}

// Handle input changes
function handleInput(e) {
    adjustTextareaHeight();
    handleAutocomplete();
}

// Adjust textarea height dynamically
function adjustTextareaHeight() {
    messageInput.style.height = 'auto';
    messageInput.style.height = Math.min(messageInput.scrollHeight, 120) + 'px';
}

// Autocomplete for commands
function handleAutocomplete() {
    const value = messageInput.value;
    const cursorPos = messageInput.selectionStart;
    const textBeforeCursor = value.substring(0, cursorPos);
    const lastLine = textBeforeCursor.split('\n').pop();
    
    if (lastLine.startsWith('/')) {
        if (lastLine.length === 1) {
            showAutocomplete(COMMANDS);
        } else {
            const query = lastLine.substring(1).toLowerCase();
            const matches = COMMANDS.filter(cmd => 
                cmd.command.substring(1).toLowerCase().startsWith(query)
            );
            
            if (matches.length > 0) {
                showAutocomplete(matches);
            } else {
                hideAutocomplete();
            }
        }
    } else {
        hideAutocomplete();
    }
}

// Show autocomplete suggestions
function showAutocomplete(matches) {
    autocompleteSuggestions.innerHTML = matches.map(match => `
        <div class="autocomplete-item" data-command="${match.command}">
            <div class="autocomplete-main">
                <span class="autocomplete-command">${match.command}</span>
                <span class="autocomplete-description">${match.description}</span>
            </div>
            <div class="autocomplete-example">${match.example}</div>
        </div>
    `).join('');
    
    autocompleteSuggestions.classList.add('active');
    
    // Add click handlers to autocomplete items
    document.querySelectorAll('.autocomplete-item').forEach(item => {
        item.addEventListener('click', () => {
            const value = messageInput.value;
            const cursorPos = messageInput.selectionStart;
            const lines = value.substring(0, cursorPos).split('\n');
            const currentLineStart = value.substring(0, cursorPos).lastIndexOf('\n') + 1;
            
            const newValue = value.substring(0, currentLineStart) + 
                           item.dataset.command + ' ' + 
                           value.substring(cursorPos);
            
            messageInput.value = newValue;
            messageInput.focus();
            
            const newCursorPos = currentLineStart + item.dataset.command.length + 1;
            messageInput.setSelectionRange(newCursorPos, newCursorPos);
            
            hideAutocomplete();
        });
    });
}

// Hide autocomplete suggestions
function hideAutocomplete() {
    autocompleteSuggestions.classList.remove('active');
}

// Parse multi-line commands (supports both newline and comma-separated)
function parseMultiLineCommands(message) {
    // First split by newlines, then split each line by commas
    const lines = message.split('\n');
    const commands = [];

    for (const line of lines) {
        // Check if line contains comma-separated commands
        if (line.includes(',')) {
            // Split by comma and process each command
            const commaSeparated = line.split(',').map(cmd => cmd.trim()).filter(cmd => cmd.length > 0);
            commands.push(...commaSeparated);
        } else {
            // Single command on this line
            const trimmed = line.trim();
            if (trimmed.length > 0) {
                commands.push(trimmed);
            }
        }
    }

    return commands;
}

// Handle send message
async function handleSendMessage() {
    const message = messageInput.value.trim();
    
    if (!message || isProcessing) return;
    
    // Clear input
    messageInput.value = '';
    adjustTextareaHeight();
    hideAutocomplete();
    
    // Show chat interface if first message
    if (conversationHistory.length === 0) {
        showChatInterface();
    }
    
    // Check for reset command
    if (message === '/reset') {
        handleReset();
        return;
    }
    
    // Parse multi-line commands
    const commands = parseMultiLineCommands(message);
    
    // Add user message (preserve line breaks for display)
    addMessage('user', message);
    
    // Disable input during processing
    isProcessing = true;
    sendBtn.disabled = true;
    
    try {
        // Process each command
        for (let i = 0; i < commands.length; i++) {
            const cmd = commands[i];
            
            // Show typing indicator for each command
            const typingId = showTypingIndicator();
            
            try {
                // Send to backend
                const response = await sendMessageToAPI(cmd);
                
                // Remove typing indicator
                removeTypingIndicator(typingId);
                
                // Add bot response
                addMessage('bot', response.message, response.metadata);
                
                // Small delay between multiple commands for better UX
                if (i < commands.length - 1) {
                    await new Promise(resolve => setTimeout(resolve, 500));
                }
                
            } catch (error) {
                console.error(`Error processing command "${cmd}":`, error);
                
                // Remove typing indicator
                removeTypingIndicator(typingId);
                
                // Show error message
                addMessage('bot', `Sorry, I encountered an error processing: "${cmd}". ${error.message}`, {
                    error: true,
                    errorMessage: error.message
                });
            }
        }
        
    } catch (error) {
        console.error('Error in message handling:', error);
    } finally {
        isProcessing = false;
        sendBtn.disabled = false;
        messageInput.focus();
    }
}

// Send message to API
async function sendMessageToAPI(message) {
    const response = await fetch(`${API_BASE_URL}/chat`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            message: message,
            thread_id: threadId
        })
    });
    
    if (!response.ok) {
        throw new Error(`API error: ${response.status}`);
    }
    
    const data = await response.json();
    
    // Update thread ID if provided
    if (data.thread_id) {
        threadId = data.thread_id;
        localStorage.setItem(THREAD_ID_KEY, threadId);
    }
    
    return data;
}

// Add message to chat
function addMessage(sender, text, metadata = {}) {
    const timestamp = new Date().toLocaleTimeString('en-US', { 
        hour: 'numeric', 
        minute: '2-digit',
        hour12: true 
    });
    
    const message = {
        sender,
        text,
        timestamp,
        metadata
    };
    
    conversationHistory.push(message);
    saveToLocalStorage();
    
    renderMessage(message);
    scrollToBottom();
}

// Render a single message
function renderMessage(message) {
    const messageEl = document.createElement('div');
    messageEl.className = `message ${message.sender}`;
    
    const avatarText = message.sender === 'user' ? 'U' : '🤖';
    
    let metadataHTML = '';
    
    // Add thinking process if available
    if (message.metadata && message.metadata.thinking) {
        metadataHTML += renderThinkingProcess(message.metadata.thinking);
    }
    
    // Add error message if available
    if (message.metadata && message.metadata.error) {
        metadataHTML += `<div class="error-message">Error: ${message.metadata.errorMessage || 'Unknown error occurred'}</div>`;
    }
    
    // Format text with line breaks
    const formattedText = formatMessageText(message.text);
    
    messageEl.innerHTML = `
        <div class="message-avatar">${avatarText}</div>
        <div class="message-content">
            <div class="message-bubble">${formattedText}</div>
            ${metadataHTML}
            <div class="message-time">${message.timestamp}</div>
        </div>
    `;
    
    messagesContainer.appendChild(messageEl);
}

// Format message text to preserve line breaks
function formatMessageText(text) {
    // Escape HTML first
    const div = document.createElement('div');
    div.textContent = text;
    let escaped = div.innerHTML;
    
    // Convert **text** to <strong>text</strong> (markdown bold)
    // Improved regex to handle multiple bold sections and avoid greedy matching
    escaped = escaped.replace(/\*\*([^\*\n]+?)\*\*/g, '<strong>$1</strong>');
    
    // Convert newlines to <br> tags
    return escaped.replace(/\n/g, '<br>');
}

// Render thinking process
function renderThinkingProcess(thinking) {
    const items = [
        { label: 'I notice this requires', value: thinking.notice || 'Processing...' },
        { label: 'Computing', value: thinking.computing || 'Analyzing...' },
        { label: 'Preparing response', value: thinking.preparing || 'Generating...' }
    ];
    
    const itemsHTML = items.map(item => `
        <div class="thinking-item">
            <span class="thinking-label">${item.label}:</span>
            <span class="thinking-value">${item.value}</span>
        </div>
    `).join('');
    
    return `
        <div class="thinking-process">
            <div class="thinking-header">
                Decision Process (Debug Info, may remove this)
            </div>
            ${itemsHTML}
        </div>
    `;
}

// Show typing indicator
function showTypingIndicator() {
    const id = 'typing-' + Date.now();
    const typingEl = document.createElement('div');
    typingEl.id = id;
    typingEl.className = 'message bot';
    typingEl.innerHTML = `
        <div class="message-avatar">🤖</div>
        <div class="message-content">
            <div class="typing-indicator">
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
            </div>
        </div>
    `;
    
    messagesContainer.appendChild(typingEl);
    scrollToBottom();
    
    return id;
}

// Remove typing indicator
function removeTypingIndicator(id) {
    const el = document.getElementById(id);
    if (el) {
        el.remove();
    }
}

// Render entire conversation history
function renderConversationHistory() {
    messagesContainer.innerHTML = '';
    conversationHistory.forEach(message => renderMessage(message));
    scrollToBottom();
}

// Show chat interface
function showChatInterface() {
    welcomeScreen.classList.add('hidden');
    messagesContainer.classList.add('active');
}

// Show welcome screen
function showWelcomeScreen() {
    welcomeScreen.classList.remove('hidden');
    messagesContainer.classList.remove('active');
}

// Handle reset
function handleReset() {
    if (conversationHistory.length === 0) return;
    
    if (confirm('Are you sure you want to reset the conversation? This will clear all messages.')) {
        conversationHistory = [];
        threadId = generateThreadId();
        messagesContainer.innerHTML = '';
        saveToLocalStorage();
        showWelcomeScreen();
    }
}

// Scroll to bottom
function scrollToBottom() {
    setTimeout(() => {
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }, 100);
}

// Generate thread ID
function generateThreadId() {
    return 'thread_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
}

// Local storage operations
function saveToLocalStorage() {
    try {
        localStorage.setItem(STORAGE_KEY, JSON.stringify(conversationHistory));
        if (threadId) {
            localStorage.setItem(THREAD_ID_KEY, threadId);
        }
    } catch (error) {
        console.error('Error saving to localStorage:', error);
    }
}

function loadFromLocalStorage() {
    try {
        const saved = localStorage.getItem(STORAGE_KEY);
        const savedThreadId = localStorage.getItem(THREAD_ID_KEY);
        
        if (saved) {
            conversationHistory = JSON.parse(saved);
        }
        
        if (savedThreadId) {
            threadId = savedThreadId;
        } else {
            threadId = generateThreadId();
        }
    } catch (error) {
        console.error('Error loading from localStorage:', error);
        conversationHistory = [];
        threadId = generateThreadId();
    }
}

// Utility: Escape HTML
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}