document.addEventListener('DOMContentLoaded', function(){
	const messageContainer = document.querySelector("#message_container");
	const messageInput = document.querySelector('[name=message_input]')
	const messageInputBtn = document.querySelector('[name=send_message_button]')
	const fileInput = document.querySelector('[name=file_input]')

	let websocketClient = new WebSocket("ws://localhost:8765");
	websocketClient.onopen = () =>{
		messageInputBtn.onclick = () =>{
			const reader = new FileReader();
			const file = fileInput.files[0];
			const text = messageInput.value.trim()
			// websocketClient.send(messageInput.value);
			 reader.onloadend = () => {
	            const base64Data = reader.result.replace(/^data:image\/\w+;base64,/, '');
	            const message = { text, image: base64Data };
	            websocketClient.send(JSON.stringify(message)); // Отправляем JSON с текстом и изображением
	        };
	        reader.readAsDataURL(file);
			const newUserMessage = document.createElement('div');
			newUserMessage.innerHTML = text
			newUserMessage.className += 'user_message'
			messageContainer.appendChild(newUserMessage)
		};
		websocketClient.onmessage = (message)=>{
			const newMessage = document.createElement('div');
			newMessage.innerHTML = message.data;
			newMessage.className += 'anton_message'
			messageContainer.appendChild(newMessage)
		}
	};

},false)
