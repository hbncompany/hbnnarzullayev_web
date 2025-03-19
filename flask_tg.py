     from flask import Flask, request

     app = Flask(__name__)


     BOT_TOKEN = '6804698522:AAG-BTwZafn-fIVOwXYt14BgE_oDzUvSakQ'

     # Example of sending a message
     def send_message(chat_id, text):
         url = f'https://api.telegram.org/bot{BOT_TOKEN}/sendMessage'
         data = {'chat_id': chat_id, 'text': text}
         response = requests.post(url, data=data)
         return response.json()

     @app.route('/')
     def index():
         return 'Hello, Telegram!'
         
     @app.route('/your-bot-token', methods=['POST'])
     def webhook():
         data = request.json
         # Implement your logic to handle incoming messages
         return 'Hello from Flaskjon+!'

     if __name__ == '__main__':
         app.run()
     
