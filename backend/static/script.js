const chatMessages = document.getElementById('chat-messages');
const userInput = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');

function addMessage(text, role) {
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('message', role);
    messageDiv.innerHTML = `<div class="bubble">${text}</div>`;
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

async function sendMessage() {
    const text = userInput.value.trim();
    if (!text) return;

    userInput.value = '';
    addMessage(text, 'user');

    // Indicador de carga
    const loadingDiv = document.createElement('div');
    loadingDiv.classList.add('message', 'bot', 'loading');
    loadingDiv.innerHTML = `<div class="bubble">Escribiendo...</div>`;
    chatMessages.appendChild(loadingDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;

    try {
        const response = await fetch('/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: text })
        });

        const data = await response.json();
        chatMessages.removeChild(loadingDiv);
        addMessage(data.response, 'bot');
    } catch (error) {
        chatMessages.removeChild(loadingDiv);
        addMessage('Lo siento, ha ocurrido un error al conectar con el asistente.', 'bot');
        console.error('Error:', error);
    }
}

sendBtn.addEventListener('click', sendMessage);
userInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') sendMessage();
});
