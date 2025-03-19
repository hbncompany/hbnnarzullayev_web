document.addEventListener('DOMContentLoaded', () => {
    const messageContainer = document.getElementById('message-container');
    const messageForm = document.getElementById('send-message-form');

    const receiverInput = messageForm.elements['receiver'];
    const messageInput = messageForm.elements['message'];

    const sender = '{{ session["username"] }}';
    let receiver = '';

    messageForm.addEventListener('submit', (e) => {
        e.preventDefault();

        const message = messageInput.value;

        if (message.trim() !== '' && receiver.trim() !== '') {
            const xhr = new XMLHttpRequest();
            xhr.open('POST', '{{ url_for('send_message') }}', true);
            xhr.setRequestHeader('Content-Type', 'application/x-www-form-urlencoded');
            xhr.onload = () => {
                if (xhr.status === 200) {
                    messageInput.value = '';
                    fetchMessages();
                }
            };
            xhr.send(`receiver=${receiver}&message=${encodeURIComponent(message)}`);
        }
    });

    receiverInput.addEventListener('input', () => {
        receiver = receiverInput.value;
        fetchMessages();
    });

    function fetchMessages() {
        const xhr = new XMLHttpRequest();
        xhr.open('GET', `{{ url_for('get_messages') }}?receiver=${receiver}`, true);
        xhr.onload = () => {
            if (xhr.status === 200) {
                const data = JSON.parse(xhr.responseText);
                const messages = data.messages;
                let html = '';

                for (let i = 0; i < messages.length; i++) {
                    const message = messages[i];
                    const isSent = message.sender === sender;

                    if ((isSent && message.receiver === receiver) || (!isSent && message.sender === receiver)) {
                        html += `
                            <div class="message ${isSent ? 'sent' : 'received'}">
                                <p>${message.message}</p>
                                <span>${message.created_at}</span>
                            </div>
                        `;
                    }
                }

                messageContainer.innerHTML = html;
            }
        };
        xhr.send();
    }
});
