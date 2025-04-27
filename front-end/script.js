// Gestion du chat RAG moderne
const chatContainer = document.getElementById('chat-container');
const userInput = document.getElementById('user-input');
const goalSelect = document.getElementById('goal-select');

// Ajoute un message dans le chat
function addMessage(role, content) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}-message`;
    messageDiv.innerHTML = content;
    chatContainer.appendChild(messageDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

// Envoie la question à l'API pour générer une réponse
async function sendMessage() {
    const query = userInput.value.trim();
    if (!query) return;

    addMessage('user', query);
    userInput.value = '';

    const selectedGoalId = goalSelect.value;

    try {
        const response = await fetch('http://localhost:8000/generate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                goal_id: selectedGoalId,
                query_text: query,
                top_k: 10,
                similarity_threshold: 1
            })
        });
        if (!response.ok) {
            const errorData = await response.json();
            addMessage('bot', `<span style="color:red">Erreur : ${errorData.detail || response.statusText}</span>`);
            return;
        }
        const data = await response.json();
        addMessage('bot', data.response);
    } catch (error) {
        console.error("Erreur:", error);
        addMessage('bot', `Erreur : ${error.message}`);
    }
}

// Récupère uniquement le contexte extrait
async function fetchContext() {
    const query = userInput.value.trim();
    if (!query) return;

    addMessage('user', query + ' (contexte demandé)');
    userInput.value = '';

    const selectedGoalId = goalSelect.value;

    try {
        const response = await fetch('http://localhost:8000/query', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                goal_id: selectedGoalId,
                query_text: query,
                top_k: 10,
                similarity_threshold: 1
            })
        });
        if (!response.ok) {
            const errorData = await response.json();
            addMessage('bot', `<span style="color:red">Erreur : ${errorData.detail || response.statusText}</span>`);
            return;
        }
        const data = await response.json();
        addMessage('bot', `<b>Contexte extrait :</b><br>${data.context}`);
    } catch (error) {
        console.error("Erreur:", error);
        addMessage('bot', `Erreur : ${error.message}`);
    }
}

// Envoi avec la touche Entrée
userInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        sendMessage();
    }
});
