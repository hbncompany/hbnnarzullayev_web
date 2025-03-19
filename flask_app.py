import pandas as pd
from flask_cors import *
import numpy as np
# import random
import re
import io
import os
import hashlib
import time
import schedule
import threading
from flask_mysqldb import MySQL
#from app import app
#import urllib.request
import requests
import threading
import time
import MySQLdb as msd
import MySQLdb.cursors
#import mysql
from flask_login import login_required#, SQLAlchemyAdapter, UserManager, UserMixin
import mysql.connector
from datetime import datetime, timedelta, timezone, date
#from mysql import connector
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user
from flask import Flask, redirect, render_template, request, session, url_for, send_from_directory,flash, send_file, g, jsonify, Response
from functools import wraps
#from flask import Flask,render_template, request
#from flask_mysqldb import MySQL
from flask_mail import *
from flask_sqlalchemy import SQLAlchemy
#from pydal import DAL, Field
#from flask import send_from_directory
from flask_socketio import SocketIO, emit
from time import sleep
from itsdangerous import URLSafeTimedSerializer,SignatureExpired
#from flask.ext.mobility.decorators import mobile_template, mobilized
from werkzeug.utils import secure_filename
from flask_login import logout_user
from functools import wraps
from flask import abort
from passlib.hash import sha256_crypt
import pandas.io.sql as sql
from celery import Celery
from flask import current_app
from random import *
from pdf2docx import Converter
import redis
from flask_oauthlib.client import OAuth
import json
import yfinance as yf
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update, ReplyKeyboardMarkup, InputFile, KeyboardButton, Location
from telegram.ext import Updater, CommandHandler, CallbackContext, CallbackQueryHandler, Filters, MessageHandler, ConversationHandler
import pandas_ta as ta
from forex_python.converter import CurrencyRates
from forex_python.bitcoin import BtcConverter
import ta
import matplotlib.pyplot as plt
from twelvedata import TDClient
import random
import string
import pandas_ta as ta
from flask_cors import CORS
import re
from urllib.parse import urlparse, parse_qs

subscribers = ['466437144']

FOREX_API_URL = 'https://www.alphavantage.co/query'
FOREX_API_KEY = 'C9FPYCBSJFGRS531'

# Redis connection setup
redis_host = 'redis-11369.c1.asia-northeast1-1.gce.cloud.redislabs.com'
redis_port = 11369
redis_password = 'g2Td26z6j4UbQKLBe6gZ75eh7zya3tSf'
# bot_token: '8103687651:AAE_sKddznoU3a2CDCeb3OlvgRK4Km1pCS4'
# Create a Redis connection
# r = redis.Redis(host=redis_host, port=redis_port, password=redis_password)

#print(dir(mysql))
UPLOAD_FOLDER = 'home/hbnnarzullayev/mysite/static/Image'
datetime.now(tz=timezone(timedelta(hours=5)))

app = Flask(__name__)
CORS(app)
mail=Mail(app)
app.config["MAIL_SERVER"]='smtp.gmail.com'
app.config["MAIL_PORT"]=997
app.config["MAIL_USERNAME"]='hbncompanyofficials@gmail.com'
app.config['MAIL_PASSWORD']='Sersarson7$'
app.config['MAIL_USE_TLS']=False
app.config['MAIL_USE_SSL']=True
app.config['CELERY_BROKER_URL'] = f'redis://default:{redis_password}@{redis_host}:{redis_port}/0'
# app.config['CELERY_BROKER_URL'] = 'redis://default:' + r.connection_pool.connection_password.decode('utf-8') + '@' + redis_host + ':' + str(redis_port) + '/0'
# app.config['CELERY_BROKER_URL'] = 'r://default:g2Td26z6j4UbQKLBe6gZ75eh7zya3tSf@redis-11369.c1.asia-northeast1-1.gce.cloud.redislabs.com:11369'
celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])
app.config["DEBUG"] = True
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

app.config["SESSION_PERMANENT"] = False
app.config['SECRET_KEY'] = 'your_secret_key_here123'
app.config["SESSION_TYPE"] = "filesystem"
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(minutes=55)
app.config['GOOGLE_ID'] = '1007902059331-shcf8lm18qmhq3jildr1fjj4g56j6c46.apps.googleusercontent.com'
app.config['GOOGLE_SECRET'] = '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'
socketio = SocketIO(app)

# app.config['MAIL_SERVER'] = 'smtp.gmail.com'
# app.config['MAIL_PORT'] = 997
# app.config['MAIL_USE_TLS'] = True
# app.config['MAIL_USERNAME'] = 'hbncompanyofficials@gmail.com'  # Your Gmail address
# app.config['MAIL_PASSWORD'] = 'Sersarson7$'  # Your Gmail password or app-specific password
BOT_TOKEN = '6804698522:AAG-BTwZafn-fIVOwXYt14BgE_oDzUvSakQ'
app.secret_key = 'super secret key'
oauth = OAuth(app)

google = oauth.remote_app(
 'google',
 consumer_key=app.config.get('GOOGLE_ID'),
 consumer_secret=app.config.get('GOOGLE_SECRET'),
 request_token_params={
     'scope': 'email'
 },
 base_url='https://www.googleapis.com/oauth2/v1/',
 request_token_url=None,
 access_token_method='POST',
 access_token_url='https://accounts.google.com/o/oauth2/token',
 authorize_url='https://accounts.google.com/o/oauth2/auth',
)



username = 'hbnnarzullayev'
password = ''
hostname = 'hbnnarzullayev.mysql.pythonanywhere-services.com'
mysql = 'hbnnarzullayev$flask3'


app.config['MYSQL_HOST'] = 'hbnnarzullayev.mysql.pythonanywhere-services.com'  # MySQL host
app.config['MYSQL_USER'] = 'hbnnarzullayev'  # MySQL username
app.config['MYSQL_PASSWORD'] = ''  # MySQL password
app.config['MYSQL_DB'] = 'hbnnarzullayev$flask3'  # MySQL database name
mail=Mail(app)
otp=randint(000000,999999)
mysql = MySQL(app)
login_manager = LoginManager(app)

s=URLSafeTimedSerializer('secret123')
#cursor = conn.cursor()
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

TELEGRAM_BOT_TOKEN = '8103687651:AAE_sKddznoU3a2CDCeb3OlvgRK4Km1pCS4' #forex
updater = Updater(token=TELEGRAM_BOT_TOKEN)
dispatcher = updater.dispatcher

td = TDClient(apikey="43090367388c427aaff6712bbdcb6f1b")

API_URL_HISTORICAL = "https://api.api-ninjas.com/v1/goldpricehistorical?period=5m"
API_URL = "https://api.api-ninjas.com/v1/goldprice"
API_KEY = "7ZjD5kkNCnx4zsj9S8Qh5Q==i35MixTgjSq8Bjwo"


# API_KEY = os.getenv('ALPHA_VANTAGE_KEY', 'A7QA865LMBF6JVEJ')

def role_required(required_role):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # Check if the user is logged in and has a role in the session
            if 'user_role' not in session:
                abort(401)  # Unauthorized (Not logged in)

            # Retrieve the user's role from the session
            user_role = session['user_role']

            # Check if the user has the required role
            if user_role != required_role:
                abort(403)  # Forbidden

            return f(*args, **kwargs)
        return decorated_function
    return decorator

def execute_query_to_dataframe(query):
    app.secret_key = "super secret key"
    user = 'hbnnarzullayev'
    password = 'Sersarson7'
    host = 'hbnnarzullayev.mysql.pythonanywhere-services.com'
    mysql = 'hbnnarzullayev$flask3'
    dbs = msd.connect(host,user,password, mysql)
    df = pd.read_sql_query(query,dbs)
    return df

def fetch_forex_data(symbol, interval):
    # symbol = symbol.replace("/", "")  # e.g., 'EUR/USD' becomes 'EURUSD'

    try:
        # Fetch historical data using the Twelve Data API
        data = td.time_series(symbol=symbol, interval=interval, outputsize=500).as_pandas()
        print("Raw API response:", data.head())  # Inspect the first few rows of the DataFrame

        if data.empty:
            raise ValueError(f"No data available for {symbol}.")

        # Ensure that 'datetime' is in the correct format and set it as the index
        data['datetime'] = pd.to_datetime(data.index)
        data.set_index('datetime', inplace=True)

        # Return only the 'close' column for analysis
        return data[['close']]

    except Exception as e:
        raise ValueError(f"Error fetching data for {symbol}: {str(e)}")

def analyze_data(df):
    # Ensure the input is a pandas DataFrame
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame, but got {}".format(type(df)))

    # Add technical indicators (e.g., SMA and RSI)
    df['SMA_20'] = df['close'].rolling(window=20).mean()
    df['RSI'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()

    # Identify Buy and Sell signals
    df['buy_signal'] = (df['RSI'] < 30) & (df['close'] > df['SMA_20'])
    df['sell_signal'] = (df['RSI'] > 70) & (df['close'] < df['SMA_20'])

    return df

def generate_advice(df):
    # Check the latest buy/sell signals
    latest_signal = df.iloc[-1]

    advice = {
        "buy": latest_signal['buy_signal'],
        "sell": latest_signal['sell_signal'],
        "take_profit": latest_signal['close'] * 1.02 if latest_signal['buy_signal'] else None,
        "stop_loss": latest_signal['close'] * 0.98 if latest_signal['buy_signal'] else None
    }
    return advice

@app.route('/analyze', methods=['GET', 'POST'])
def analyze():
    if request.method == 'POST':
        try:
            # Parse JSON data
            data = request.get_json()
            if not data:
                return jsonify({'error': 'No data provided'}), 400

            symbol = data.get('symbol')
            interval = data.get('interval')

            if not symbol or not interval:
                return jsonify({'error': 'Missing symbol or interval'}), 400

            # Fetch and process data
            raw_data = fetch_forex_data(symbol, interval)
            processed_data = analyze_data(raw_data)
            print(processed_data.head())
            chart_data_json = processed_data.to_json(orient="records")
            advice = generate_advice(processed_data)

            # Ensure advice is JSON serializable
            advice = make_serializable(advice)

            return jsonify({
                'chart_data': chart_data_json,
                'advice': advice
            })

        except ValueError as e:
            return jsonify({'error': str(e)}), 400

    return render_template('analyze.html')

def make_serializable(obj):
    """Recursively converts numpy types to Python native types."""
    if isinstance(obj, np.generic):  # If the object is a numpy type
        return obj.item()  # Convert it to its native Python equivalent
    elif isinstance(obj, dict):  # If the object is a dictionary
        return {key: make_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):  # If the object is a list
        return [make_serializable(item) for item in obj]
    return obj  # Return the object if no conversion is needed

def start(update: Update, context: CallbackContext) -> int:
    user = update.effective_user
    username1 = user.username
    icon_data = emojize(":bar_chart:")
    icon_announces = emojize(":speaker_high_volume:")
    icon_announce = emojize(":loudspeaker:")
    icon_register = emojize(":bust_in_silhouette:")
    icon_login = emojize(":key:")
    icon_logout = emojize(":door:")
    icon_others = emojize(":right_arrow:")
    icon_info = emojize(":information:")
    icon_home = emojize(":house:")

    # Define buttons
    button1 = InlineKeyboardButton(f"{icon_announces} E'lonlar", callback_data='send_photo')
    button2 = InlineKeyboardButton(f"{icon_announce} E'lon berish", callback_data='announce')
    button3 = InlineKeyboardButton(f"{icon_register} Register", callback_data='register')
    button4 = InlineKeyboardButton(f"{icon_login} Kirish", callback_data='login')
    button5 = InlineKeyboardButton(f"{icon_logout} Chiqish", callback_data='logout')
    button7 = InlineKeyboardButton(f"{icon_register} Kabinet", callback_data='profile')
    button9 = InlineKeyboardButton(f"{icon_others} Boshqa xizmatlar", callback_data='other_services')
    button6 = InlineKeyboardButton(f"{icon_home} Bosh sahifa", callback_data='start')
    button8 = InlineKeyboardButton(f"{icon_info} Biz haqimizda", callback_data='about')

    # Create button rows
    button_row1 = [button1, button2, button3]
    button_row2 = [button4, button5, button7]
    button_row4 = [button9, button8]
    button_row3 = [button6]

    # Combine button rows into a single list
    button_rows = [button_row1, button_row2, button_row4, button_row3]

    # Create InlineKeyboardMarkup with the button rows
    reply_markup = InlineKeyboardMarkup(button_rows)
    reply_markup1 = ReplyKeyboardMarkup([[KeyboardButton('Bekor qilish', request_location=True)]], resize_keyboard=True, one_time_keyboard=True)

    # Send message with inline buttons
    if update.message:
        update.message.reply_text(
            "Hush kelibsiz! Bu O'zbekistondagi eng birinchi ommaviy-savdo telegram boti",
            reply_markup=reply_markup
        )
        return ConversationHandler.END
        context.bot.send_message(chat_id=update.effective_chat.id,text="Bosh sahifa")
    else:
        context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="Bosh sahifa, {}".format(username1),
            reply_markup=reply_markup
        )
        return ConversationHandler.END
        context.bot.send_message(chat_id=update.effective_chat.id,text="Bosh sahifa")

    return ConversationHandler.END
    context.bot.send_message(chat_id=update.effective_chat.id,text="Bosh sahifa")

# Add the handler to the dispatcher
dispatcher.add_handler(CommandHandler('start', start))
dispatcher.add_handler(CallbackQueryHandler(start, pattern='^start$'))
dispatcher.add_handler(MessageHandler(Filters.regex('^Bosh sahifa$'), start))

def get_gold_price():
    """Fetch the current price of Gold (Futures: GC=F) from Yahoo Finance."""
    gold = yf.Ticker('XAUUSD=X')
    data = gold.history(period='1d')  # Get the latest 1 day of data
    return data['Close'][0]  # Return the most recent close price

def get_thresholds():
    # """Calculate dynamic buy and sell thresholds based on recent market data."""
    # api_url = "https://data-asg.goldprice.org/dbXRates/USD"

    # headers = {
    #     "User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:120.0) Gecko/20100101 Firefox/120.0"
    # }

    # data = requests.get(api_url, headers=headers).json()
    gold = yf.Ticker('XAUUSD=X')
    data = gold.history(period='1d', interval='5m')  # Fetch 5 days of 15-minute interval data
    prices = data['Close'].tail(10)  # Use the last 10 data points for calculation

    recent_high = prices.max()  # High from recent 10 candles
    recent_low = prices.min()   # Low from recent 10 candles

    # Adjust thresholds based on the range
    buy_threshold = recent_low + (recent_high - recent_low) * 0.25
    sell_threshold = recent_high - (recent_high - recent_low) * 0.25

    return buy_threshold, sell_threshold

def send_message_fx(chat_id, text):
    """Send a message via Telegram Bot API."""
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {'chat_id': chat_id, 'text': text}
    response = requests.post(url, json=payload)

    # Check if message was sent successfully
    if response.status_code == 200:
        print(f"Message sent to {chat_id}: {text}")
    else:
        print(f"Failed to send message to {chat_id}")

@app.route('/check_gold', methods=['GET'])
def check_gold():
    """Fetch gold price and send alerts dynamically."""
    buy_threshold, sell_threshold = get_thresholds()
    if not buy_threshold or not sell_threshold:
        return "Failed to fetch thresholds."

    # Fetch the latest gold price
    gold_price = get_gold_price()

    for chat_id in subscribers:
        if gold_price < buy_threshold:
            send_message_fx(chat_id, f"Gold price is ${gold_price}. It's time to BUY! (Threshold: ${buy_threshold:.2f})")
        elif gold_price > sell_threshold:
            send_message_fx(chat_id, f"Gold price is ${gold_price}. It's time to SELL! (Threshold: ${sell_threshold:.2f})")

    return 'Checked and notified.'

def fetch_gold_history():
    """Fetch 1-month historical gold prices (XAU/USD) at 5-minute intervals."""
    try:
        # Fetch 1-month data with a 5-minute interval
        gold = yf.Ticker('GC=F')  # Gold futures as XAU/USD is more reliable
        data = gold.history(period='1mo', interval='5m')

        if data.empty:
            raise ValueError("No data found for the specified ticker.")

        return data
    except Exception as e:
        print(f"Error fetching gold history: {e}")
        return None

def analyze_gold_data(data, tp_percent=0.5, sl_percent=0.3):
    """Analyze historical gold prices, provide buy/sell points, and set TP/SL."""
    # Calculate moving averages
    data['SMA_10'] = data['Close'].rolling(window=10).mean()  # 10 periods moving average
    data['SMA_50'] = data['Close'].rolling(window=50).mean()  # 50 periods moving average

    # Identify buy and sell signals
    data['Signal'] = 0
    data.loc[data['SMA_10'] > data['SMA_50'], 'Signal'] = 1  # Buy signal
    data.loc[data['SMA_10'] < data['SMA_50'], 'Signal'] = -1  # Sell signal

    # Identify buy and sell points with TP/SL
    buy_sell_points = []
    for i in range(1, len(data)):
        if data['Signal'].iloc[i] == 1 and data['Signal'].iloc[i - 1] != 1:
            close_price = data['Close'].iloc[i]
            tp = close_price * (1 + tp_percent / 100)
            sl = close_price * (1 - sl_percent / 100)
            buy_sell_points.append((data.index[i], "BUY", close_price, tp, sl))
        elif data['Signal'].iloc[i] == -1 and data['Signal'].iloc[i - 1] != -1:
            close_price = data['Close'].iloc[i]
            tp = close_price * (1 - tp_percent / 100)
            sl = close_price * (1 + sl_percent / 100)
            buy_sell_points.append((data.index[i], "SELL", close_price, tp, sl))

    return buy_sell_points, data

@app.route('/gold_advice', methods=['GET','POST'])
def gold_advice(tp_percent=0.5, sl_percent=0.3):
    """Fetch historical data, analyze, and provide advice with TP/SL."""
    data = fetch_gold_history()
    if data is None:
        return "Unable to fetch gold price data."

    buy_sell_points, data = analyze_gold_data(data, tp_percent, sl_percent)

    # Display Buy/Sell Points
    print("Buy/Sell Points with TP/SL:")
    for point in buy_sell_points:
        action, price, tp, sl = point[1], point[2], point[3], point[4]
        print(f"{action} at {point[0]} -> Price: ${price:.2f}, TP: ${tp:.2f}, SL: ${sl:.2f}")

    # Get the most recent advice
    latest_signal = data['Signal'].iloc[-1]
    latest_price = data['Close'].iloc[-1]
    advice = "HOLD"
    if latest_signal == 1:
        advice = "BUY"
    elif latest_signal == -1:
        advice = "SELL"
    for chat_id in subscribers:
        send_message_fx(chat_id, f"Current Gold Price: ${latest_price:.2f}\nAdvice: {advice}")

    return f"Current Gold Price: ${latest_price:.2f}\nAdvice: {advice}"

@app.route('/webhook', methods=['POST'])
def telegram_webhook():
    update = request.get_json()
    print(json.dumps(update, indent=4))  # Debugging incoming updates

    if 'message' in update and 'text' in update['message']:
        chat_id = update['message']['chat']['id']
        text = update['message']['text']

        # Handle /gold_advice command
        if text == '/gold_advice':
            # Send a response to the user
            send_message(chat_id, "Analyzing gold prices, please wait...")
            advice = gold_advice()
            send_message(chat_id, advice)

    return '', 200

def send_message_forex(chat_id, text):
    import requests
    url = f"https://api.telegram.org/bot<YOUR_BOT_TOKEN>/sendMessage"
    payload = {'chat_id': chat_id, 'text': text}
    requests.post(url, json=payload)

def generate_confirmation_token(email):
    # Create a unique token based on the user's email and current timestamp
    timestamp = str(int(time.time()))
    data = email + timestamp
    token = hashlib.sha256(data.encode()).hexdigest()
    return token
# User class representing a user record from the MySQL table
class User(UserMixin):
    def __init__(self, user_id):
        self.id = user_id

def send_message(chat_id, text):
         url = f'https://api.telegram.org/bot{BOT_TOKEN}/sendMessage'
         data = {'chat_id': chat_id, 'text': text}
         response = requests.post(url, data=data)
         return response.json()

@app.route('/8103687651:AAE_sKddznoU3a2CDCeb3OlvgRK4Km1pCS4', methods=['POST'])
def webhook():
 data = request.json
 # Implement your logic to handle incoming messages
 return 'Hello from Flaskjon+!'

@login_manager.user_loader
def load_user(user_id):
    return User(user_id)


def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def home():
    # thread = threading.Thread(target=saldo)
    # thread.start()
    #city = "London"  # Replace with your desired city or make it dynamic
    #temperature, description = get_weather(city)
    id:home
    if request.method == 'GET':
        pass

    if request.method == 'POST':
        pass

    return render_template("index.html", content="True")

@app.route("/tax")
# @role_required('admin')
def tax():
    return render_template("tax.html", content="True")

@app.route('/chats')
def chats():
    return render_template('chats.html')

@app.route('/tax_details')
def detailed():
    return render_template('tax_details.html')

@app.route('/execute_procedure')
def execute_procedure():
    with open('/home/hbnnarzullayev/mysite/SQL/Procedure.sql', 'r') as file:
        sql_script = file.read()

    db = MySQLdb.connect("hbnnarzullayev.mysql.pythonanywhere-services.com","hbnnarzullayev","Sersarson7","hbnnarzullayev$flask3" )
    cursor = db.cursor()

    # Split the SQL script into separate queries
    queries = sql_script.split(';')

    for query in queries:
        if query.strip():
            try:
                cursor.execute(query)
                db.commit()
            except Exception as e:
                db.rollback()
                print(f"Error executing query: {query}")
                print(str(e))

    db.close()
    return render_template('index.html')
@socketio.on('message')
def handle_message(message):
    print('received message: ' + message)
    socketio.emit('message', message)

def pdf_to_word(pdf_path, word_path):
    cv = Converter(pdf_path)
    cv.convert(word_path, start=0, end=None)
    cv.close()


@app.route("/pdftoword", methods=["GET", "POST"])
def pdftoword():
    if request.method == "POST":
        if "pdf_file" not in request.files:
            return "No file part"

        pdf_file = request.files["pdf_file"]

        if pdf_file.filename == "":
            return "No selected file"

        if pdf_file:
            pdf_filename = pdf_file.filename
            pdf_path = f"/home/hbnnarzullayev/mysite/uploads/"+pdf_filename
            word_filename = pdf_filename.replace(".pdf", ".docx")
            word_path = f"/home/hbnnarzullayev/mysite/uploads/"+word_filename

            pdf_file.save(pdf_path)
            pdf_to_word(pdf_path, word_path)

            return send_file(word_path, as_attachment=True)

    return render_template("pdftoword.html")


@app.route('/', methods=['POST'])
def upload_image():
	if 'file' not in request.files:
		flash('No file part')
		return redirect(request.url)
	file = request.files['file']
	if file.filename == '':
		flash('No image selected for uploading')
		return redirect(request.url)
	if file and allowed_file(file.filename):
		filename = secure_filename(file.filename)
		file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
		#print('upload_image filename: ' + filename)
		flash('Image successfully uploaded and displayed below')
		return render_template('upload.html', filename=filename)
	else:
		flash('Allowed image types are -> png, jpg, jpeg, gif')
		return redirect(request.url)

@app.route('/display/<filename>')
def display_image(filename):
	#print('display_image filename: ' + filename)
	return redirect(url_for('static', filename='uploads/' + filename), code=301)

@app.route("/signup", methods=["POST", "GET"])
def signup():
    msg = ""
    app.config["DEBUG"] = True
    app.secret_key = "super secret key"
    user = 'hbnnarzullayev'
    password = 'Sersarson7'
    host = 'hbnnarzullayev.mysql.pythonanywhere-services.com'
    mysql = 'hbnnarzullayev$flask3'
    dbs = msd.connect(host,user,password, mysql)
    if request.method == 'POST':
        now = datetime.now(tz=timezone(timedelta(hours=5)))
        username = request.form['username']
        email = request.form['email']
        password = request.form['pwd']
        passwordc = request.form['pwdc']
        # hashed_password = sha256_crypt.hash(password)
        cursor = dbs.cursor()
        cursor.execute('SELECT * FROM users2 WHERE username = %s', (username,))
        #result2=cursor.execute("SELECT * FROM users WHERE email=%s",request.form[email])
        user = cursor.fetchone()
        # token = generate_confirmation_token(user.email)  # Replace with your token generation logic
        # confirm_url = url_for('confirm_email', token=token, _external=True)
        # send_confirmation_email(user.email, confirm_url)  # Send confirmation email

        flash('A confirmation email has been sent. Please check your inbox.', 'success')
        print(user)
        if user:
            msg = "Nik avvaldan mavjud!"
        elif user:
            (session['username'] == user[0]) or (session['email'] == email)
            msg = "Akkount avvaldan mavjud!"
        elif not re.match(r'[^@]+@[^@]+\.[^@]+', email):
            msg = "E-mail xato!"
        elif not re.match(r'[A-Za-z0-9]+', username):
            msg = "Username(nik) faqat xarf va raqamlardan iborat bolishi lozim!"
        elif not username or not password or not email:
            msg = "Formani to'ldiring!"
        elif password != passwordc:
            msg = "Tasdiqlash paroli xato!"
        else:
            session['loggedin'] = True
            # Account doesnt exists and the form data is valid, now insert new account into accounts table
            cursor.execute(''' INSERT INTO users2 VALUES(%s,%s,%s,%s,%s,%s,%s)''',[username,password,email,'',now.strftime('%Y-%m-%d %H:%M:%S'),'',''])
            flash('A confirmation link has been sent to your email','success')
            dbs.commit()
            msg = "Muvaffaqqiyatli ro'yxatdan o'tdingiz!"
    return render_template('signup.html', msg=msg)

@app.route("/register", methods=["POST", "GET"])
def register():
    data = request.get_json()
    msg = ""
    app.config["DEBUG"] = True
    app.secret_key = "super secret key"
    user = 'hbnnarzullayev'
    password = 'Sersarson7'
    host = 'hbnnarzullayev.mysql.pythonanywhere-services.com'
    mysql = 'hbnnarzullayev$flask3'
    dbs = msd.connect(host,user,password, mysql)
    if request.method == 'POST':
        now = datetime.now(tz=timezone(timedelta(hours=5)))
        username = data.get('username')
        email = data.get('email')
        password = data.get('pwd')
        passwordc = data.get('pwdc')
        # hashed_password = sha256_crypt.hash(password)
        cursor = dbs.cursor()
        cursor.execute('SELECT * FROM users2 WHERE username = %s', (username,))
        #result2=cursor.execute("SELECT * FROM users WHERE email=%s",request.form[email])
        user = cursor.fetchone()
        # token = generate_confirmation_token(user.email)  # Replace with your token generation logic
        # confirm_url = url_for('confirm_email', token=token, _external=True)
        # send_confirmation_email(user.email, confirm_url)  # Send confirmation email

        flash('A confirmation email has been sent. Please check your inbox.', 'success')
        print(user)
        if user:
            msg = "Nik avvaldan mavjud!"
        elif user:
            (session['username'] == user[0]) or (session['email'] == email)
            msg = "Akkount avvaldan mavjud!"
        elif not re.match(r'[^@]+@[^@]+\.[^@]+', email):
            msg = "E-mail xato!"
        elif not re.match(r'[A-Za-z0-9]+', username):
            msg = "Username(nik) faqat xarf va raqamlardan iborat bolishi lozim!"
        elif not username or not password or not email:
            msg = "Formani to'ldiring!"
        elif password != passwordc:
            msg = "Tasdiqlash paroli xato!"
        else:
            session['loggedin'] = True
            # Account doesnt exists and the form data is valid, now insert new account into accounts table
            cursor.execute(''' INSERT INTO users2 VALUES(%s,%s,%s,%s,%s,%s,%s)''',[username,password,email,'',now.strftime('%Y-%m-%d %H:%M:%S'),'',''])
            flash('A confirmation link has been sent to your email','success')
            dbs.commit()
            msg = "Muvaffaqqiyatli ro'yxatdan o'tdingiz!"
    return render_template('register.html', msg=msg)

@app.route("/register_app", methods=["POST", "GET"])
def register_app():
    data = request.get_json()
    print(data)
    msg = ""
    app.config["DEBUG"] = True
    app.secret_key = "super secret key"
    user = 'hbnnarzullayev'
    password = 'Sersarson7'
    host = 'hbnnarzullayev.mysql.pythonanywhere-services.com'
    mysql = 'hbnnarzullayev$flask3'
    dbs = msd.connect(host,user,password, mysql)
    if request.method == 'POST':
        now = datetime.now(tz=timezone(timedelta(hours=5)))
        username = data.get('username')
        email = data.get('email')
        password = data.get('pwd')
        passwordc = data.get('pwdc')
        app_id = data.get('app_id')
        # hashed_password = sha256_crypt.hash(password)
        cursor = dbs.cursor()
        cursor.execute('SELECT * FROM users2 WHERE username = %s', (username,))
        #result2=cursor.execute("SELECT * FROM users WHERE email=%s",request.form[email])
        user = cursor.fetchone()
        # token = generate_confirmation_token(user.email)  # Replace with your token generation logic
        # confirm_url = url_for('confirm_email', token=token, _external=True)
        # send_confirmation_email(user.email, confirm_url)  # Send confirmation email

        flash('A confirmation email has been sent. Please check your inbox.', 'success')
        print(user)
        if user:
            msg = "Nik avvaldan mavjud!"
        elif user:
            (session['username'] == user[0]) or (session['email'] == email)
            msg = "Akkount avvaldan mavjud!"
        elif not re.match(r'[^@]+@[^@]+\.[^@]+', email):
            msg = "E-mail xato!"
        elif not re.match(r'[A-Za-z0-9]+', username):
            msg = "Username(nik) faqat xarf va raqamlardan iborat bolishi lozim!"
        elif not username or not password or not email:
            msg = "Formani to'ldiring!"
        elif password != passwordc:
            msg = "Tasdiqlash paroli xato!"
        else:
            session['loggedin'] = True
            # Account doesnt exists and the form data is valid, now insert new account into accounts table
            cursor.execute('''INSERT INTO users2 (username, password, email, rowid, app_id, new_date2, new_date1, new_date, user_role) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) ''', (username, password, email, '', app_id, now.strftime('%Y-%m-%d %H:%M:%S'), '', '','2'))
            # cursor.execute(''' INSERT INTO users2 VALUES(%s,%s,%s,%s,%s,%s,%s,%s)''',[username,password,email,'',app_id, now.strftime('%Y-%m-%d %H:%M:%S'),'',''])
            flash('A confirmation link has been sent to your email','success')
            dbs.commit()
            msg = "Muvaffaqqiyatli ro'yxatdan o'tdingiz!"
        print(msg)
    return render_template('register.html', msg=msg)

@app.route("/fire_register_app", methods=["POST", "GET"])
def fire_register_app():
    data = request.get_json()
    print(data)
    msg = ""
    app.config["DEBUG"] = True
    app.secret_key = "super secret key"
    user = 'hbnnarzullayev'
    password = 'Sersarson7'
    host = 'hbnnarzullayev.mysql.pythonanywhere-services.com'
    mysql = 'hbnnarzullayev$flask3'
    dbs = msd.connect(host,user,password, mysql)
    if request.method == 'POST':
        now = datetime.now(tz=timezone(timedelta(hours=5)))
        fire_user_id = data.get('user_id')
        username = data.get('username')
        email = data.get('email')
        phone = data.get('phone')
        password = data.get('pwd')
        passwordc = data.get('pwdc')
        app_id = data.get('app_id')
        # hashed_password = sha256_crypt.hash(password)
        cursor = dbs.cursor()
        cursor.execute('SELECT * FROM fire_users WHERE phone = %s', (phone,))
        #result2=cursor.execute("SELECT * FROM users WHERE email=%s",request.form[email])
        user = cursor.fetchone()
        # token = generate_confirmation_token(user.email)  # Replace with your token generation logic
        # confirm_url = url_for('confirm_email', token=token, _external=True)
        # send_confirmation_email(user.email, confirm_url)  # Send confirmation email

        flash('A confirmation email has been sent. Please check your inbox.', 'success')
        print(user)
        if user:
            msg = "Akkount avvaldan mavjud!"
            return Response(response="Akkount avvaldan mavjud!", status=500)
        elif user:
            (session['username'] == user[0]) or (session['email'] == email)
            msg = "Akkount avvaldan mavjud!"
            return Response(response=msg, status=500)
        elif not re.match(r'[A-Za-z0-9]+', username):
            msg = "Username(nik) faqat xarf va raqamlardan iborat bolishi lozim!"
            return Response(response=msg, status=500)
        elif not username or not password or not email:
            msg = "Formani to'ldiring!"
            return Response(response=msg, status=500)
        elif password != passwordc:
            msg = "Tasdiqlash paroli xato!"
            return Response(response=msg, status=500)
        else:
            session['loggedin'] = True
            # Account doesnt exists and the form data is valid, now insert new account into accounts table
            cursor.execute('''INSERT INTO fire_users (fire_user_id, username, password, email, phone, rowid, app_id, new_date2, new_date, user_role) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s) ''', (fire_user_id, username, password, email, phone, '', app_id, now.strftime('%Y-%m-%d %H:%M:%S'), '','2'))
            # cursor.execute(''' INSERT INTO users2 VALUES(%s,%s,%s,%s,%s,%s,%s,%s)''',[username,password,email,'',app_id, now.strftime('%Y-%m-%d %H:%M:%S'),'',''])
            flash('A confirmation link has been sent to your email','success')
            dbs.commit()
            msg = "Muvaffaqqiyatli ro'yxatdan o'tdingiz!"
            return Response(response=msg, status=200)
        print(msg)
    return render_template('register.html', msg=msg)


@app.route('/verify', methods=["POST"])
def verify():
    email=request.form["email"]
    msg=Message('OTP', sender='hbncompanyofficials@gmail.com', recipients=[email])
    msg.body=str(otp)
    mail.send(msg)
    return render_template("verify.html")

@app.route('/get_quiz', methods=['GET'])
def get_quiz():
    app.config["DEBUG"] = True
    app.secret_key = "super secret key"
    user = 'hbnnarzullayev'
    password = 'Sersarson7'
    host = 'hbnnarzullayev.mysql.pythonanywhere-services.com'
    mysql = 'hbnnarzullayev$flask3'
    dbs = msd.connect(host,user,password, mysql)
    cursor = dbs.cursor()
    # Fetch 10 random questions
    cursor.execute("SELECT Id, quiz, option1, option2, option3, option4, answer FROM quizes where Teacher='Uchlik' ORDER BY RAND() LIMIT 10")
    rows = cursor.fetchall()

    # Get column names for dictionary conversion
    column_names = [desc[0] for desc in cursor.description]

    # Convert each row to a dictionary
    questions = [dict(zip(column_names, row)) for row in rows]

    # Prepare the result with randomized options
    result = []
    for question in questions:
        options = [
            {"option": question['option1'], "index": 1},
            {"option": question['option2'], "index": 2},
            {"option": question['option3'], "index": 3},
            {"option": question['option4'], "index": 4},
        ]
        shuffle(options)  # Shuffle options in-place

        # Find the new index of the correct answer after shuffling
        correct_option_index = next(i + 1 for i, opt in enumerate(options) if opt["index"] == question['answer'])

        # Add the question and randomized options to the result
        result.append({
            "id": question['Id'],
            "quiz": question['quiz'],
            "options": [opt["option"] for opt in options],  # Shuffled options
            "answer": correct_option_index,  # New index of the correct option
        })

    return jsonify(result)

@app.route('/get_quiz_type', methods=['GET','POST'])
def get_quiz_type():
    print("get_quiz_type:")
    app.config["DEBUG"] = True
    app.secret_key = "super secret key"
    data = request.get_json()
    print(data)
    app_id = data.get('app_id')
    # username = data.get('username')
    user = 'hbnnarzullayev'
    password = 'Sersarson7'
    host = 'hbnnarzullayev.mysql.pythonanywhere-services.com'
    mysql = 'hbnnarzullayev$flask3'
    dbs = msd.connect(host,user,password, mysql)
    cursor = dbs.cursor()
    # Fetch 10 random questions
    try:
        cursor.execute("SELECT DISTINCT CONCAT(science, '-test') AS concatenated_science FROM quizes WHERE Teacher = %s", (app_id,))
        rows = cursor.fetchall()

        # Extract the concatenated science values from rows and return as a list
        quiz_types = [row[0] for row in rows]  # Each row[0] will give the concatenated value
        print(quiz_types)
        return jsonify(quiz_types)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get_quiz_types', methods=['GET','POST'])
def get_quiz_types():
    print("get_quiz_types:")
    app.config["DEBUG"] = True
    app.secret_key = "super secret key"
    data = request.get_json()
    print(data)
    app_id = data.get('app_id')
    username = data.get('username')
    user = 'hbnnarzullayev'
    password = 'Sersarson7'
    host = 'hbnnarzullayev.mysql.pythonanywhere-services.com'
    mysql = 'hbnnarzullayev$flask3'
    dbs = msd.connect(host,user,password, mysql)
    cursor = dbs.cursor()
    # Fetch 10 random questions
    try:
        cursor.execute("SELECT user_role FROM users2 WHERE username = %s and app_id=%s", (username, app_id,))
        user_data = cursor.fetchone()
        print(user_data)

        if user_data:
            user_role = user_data[0]
            if user_role == 1:
                is_admin = 'true'
            else:
                is_admin = False
        else:
            is_admin = False
        print(is_admin)
        cursor.execute("SELECT DISTINCT CONCAT(science, '-test') AS concatenated_science FROM quizes WHERE Teacher = %s", (app_id,))
        rows = cursor.fetchall()

        # Extract the concatenated science values from rows and return as a list
        quiz_types = [row[0] for row in rows]  # Each row[0] will give the concatenated value
        # return jsonify(quiz_types)
        # video_list = [{'url': video[0], 'name': video[1], 'app': video[2]} for video in videos]
        print(quiz_types)
        return jsonify({'quiz_types': quiz_types, 'is_admin': is_admin})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get_quiz_type_test', methods=['GET','POST'])
def get_quiz_type_test():
    print("get_quiz_type_test:")
    app.config["DEBUG"] = True
    app.secret_key = "super secret key"
    data = request.get_json()
    print(data)
    app_id = data.get('app_id')
    username = data.get('username')
    user = 'hbnnarzullayev'
    password = 'Sersarson7'
    host = 'hbnnarzullayev.mysql.pythonanywhere-services.com'
    mysql = 'hbnnarzullayev$flask3'
    dbs = msd.connect(host,user,password, mysql)
    cursor = dbs.cursor()
    # Fetch 10 random questions
    try:
        cursor.execute("SELECT user_role FROM users2 WHERE username = %s and app_id=%s", (username, app_id,))
        user_data = cursor.fetchone()
        print(user_data)

        if user_data:
            user_role = user_data[0]
            if user_role == 1:
                is_admin = 'true'
            else:
                is_admin = False
        else:
            is_admin = False
        print(is_admin)
        cursor.execute("SELECT DISTINCT CONCAT(science, '-test') AS concatenated_science, ifnull(Name,'Nomlanmagan') as Name FROM quizes WHERE Teacher = %s", (app_id,))
        rows = cursor.fetchall()

        # Extract the concatenated science values from rows and return as a list
        quiz_types = [row[0] for row in rows]  # Each row[0] will give the concatenated value
        quiz_type_names = [row[1] for row in rows]  # Each row[0] will give the concatenated value
        # return jsonify(quiz_types)
        # video_list = [{'url': video[0], 'name': video[1], 'app': video[2]} for video in videos]
        print(quiz_types)
        return jsonify({'quiz_types': quiz_types, 'quiz_type_names': quiz_type_names, 'is_admin': is_admin})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/app_get_quiz', methods=['GET','POST'])
def app_get_quiz():
    print("app_get_quiz______________")
    data = request.get_json()
    print(data)
    app_id = data.get('app_id')
    science = data.get('science')
    science_code = science[:1]
    print(science_code)
    app.config["DEBUG"] = True
    app.secret_key = "super secret key"
    user = 'hbnnarzullayev'
    password = 'Sersarson7'
    host = 'hbnnarzullayev.mysql.pythonanywhere-services.com'
    mysql = 'hbnnarzullayev$flask3'
    dbs = msd.connect(host,user,password, mysql)
    cursor = dbs.cursor()
    # Fetch 10 random questions
    cursor.execute("SELECT Id, quiz, option1, option2, option3, option4, answer FROM quizes where Teacher=%s and science=%s ORDER BY RAND() LIMIT 10", (app_id, science_code, ))
    rows = cursor.fetchall()

    # Get column names for dictionary conversion
    column_names = [desc[0] for desc in cursor.description]

    # Convert each row to a dictionary
    questions = [dict(zip(column_names, row)) for row in rows]

    # Prepare the result with randomized options
    result = []
    for question in questions:
        options = [
            {"option": question['option1'], "index": 1},
            {"option": question['option2'], "index": 2},
            {"option": question['option3'], "index": 3},
            {"option": question['option4'], "index": 4},
        ]
        shuffle(options)  # Shuffle options in-place

        # Find the new index of the correct answer after shuffling
        correct_option_index = next(i + 1 for i, opt in enumerate(options) if opt["index"] == question['answer'])

        # Add the question and randomized options to the result
        result.append({
            "id": question['Id'],
            "quiz": question['quiz'],
            "options": [opt["option"] for opt in options],  # Shuffled options
            "answer": correct_option_index,  # New index of the correct option
        })
    print(result)
    return jsonify(result)

@app.route("/api/userscores", methods=["POST", "GET"])
def userscore():
    data = request.get_json()
    msg = ""
    app.config["DEBUG"] = True
    app.secret_key = "super secret key"
    user = 'hbnnarzullayev'
    password = 'Sersarson7'
    host = 'hbnnarzullayev.mysql.pythonanywhere-services.com'
    mysql = 'hbnnarzullayev$flask3'
    dbs = msd.connect(host, user, password, mysql)

    if request.method == 'POST':
        now = datetime.now(tz=timezone(timedelta(hours=5)))
        score = data.get('score')
        user = data.get('user')
        app_data = data.get('app')  # Renamed to avoid overwriting the `app` variable

        # Ensure all required data is present
        if not score or not user or not app_data:
            return jsonify({"error": "Missing required parameters"}), 400

        try:
            cursor = dbs.cursor()
            cursor.execute('''INSERT INTO user_scores (user, score, app, datetime) VALUES (%s, %s, %s, %s)''',
                           [user, score, app_data, now.strftime('%Y-%m-%d %H:%M:%S')])
            dbs.commit()
            return jsonify({"message": "Score saved successfully!"}), 200
        except Exception as e:
            print(f"Error: {e}")
            return jsonify({"error": "Failed to save score"}), 500

    return jsonify({"error": "Invalid request method"}), 405

@app.route('/get_data_as_json')
def get_data_as_json():
    app.config["DEBUG"] = True
    app.secret_key = "super secret key"
    user = 'hbnnarzullayev'
    password = 'Sersarson7'
    host = 'hbnnarzullayev.mysql.pythonanywhere-services.com'
    mysql = 'hbnnarzullayev$flask3'
    dbs = msd.connect(host,user,password, mysql)
    cursor = dbs.cursor()
    cursor.execute("SELECT COUNT(*) FROM quiz")
    total_rows = cursor.fetchone()[0]

    # Generate a list of all row indices and shuffle them
    all_rows = list(range(1, total_rows + 1))
    shuffle(all_rows)

    # Select the first 10 shuffled row indices
    random_rows = all_rows[:10]

    # Fetch the 10 random rows
    data_list = []
    for row_id in random_rows:
        cursor.execute("SELECT * FROM quiz LIMIT 1 OFFSET %s", (row_id - 1,))
        row_data = cursor.fetchone()
        if row_data:
            data_dict = {
                'question': row_data[1],
                'choices':[row_data[2],row_data[3],row_data[4],row_data[5]],
                'correct_answer':row_data[2]

            # Add more columns as needed
            }
            data_list.append(data_dict)

    # Create a JSON response
    response = jsonify(data_list)

    # Save the JSON data to a file
    with open('random_data.json', 'w') as json_file:
        json_file.write(response.get_data(as_text=True))

    return response

    # # Convert the data to a list of dictionaries
    # data_list = []
    # for row in data:
        # data_dict = {
        #     'question': row[0],
        #     'choices':[row[1],row[2],row[3],row[4]],
        #     'correct_answer':row[1]

        #     # Add more columns as needed
        # }
    #     data_list.append(data_dict)

    # Create a JSON response
    # response = jsonify(data_list)

    # # Optionally, save the JSON data to a file
    # with open('data.json', 'w') as json_file:
    #     json_file.write(response.get_data(as_text=True))

    # return response

@app.route('/api/testscores', methods=['GET', 'POST'])
def get_test_results():
    app.config["DEBUG"] = True
    app.secret_key = "789456asd"
    user = 'hbnnarzullayev'
    password = 'Sersarson7'
    host = 'hbnnarzullayev.mysql.pythonanywhere-services.com'
    mysql = 'hbnnarzullayev$flask3'
    dbs = msd.connect(host, user, password, mysql)
    cursor = dbs.cursor()

    # Handle the case when method is POST, and request contains JSON body
    if request.method == 'POST':
        data = request.get_json()  # This is valid for POST, not GET

        # Fetch the password for the 'Teacher' from the database
        cursor.execute("SELECT password FROM users2 WHERE username='Teacher'")
        passwords = cursor.fetchall()

        # Check if the result is not empty
        if passwords:
            stored_password = passwords[0][0]  # Extract the password from the tuple
        else:
            stored_password = ''

        print(data)  # Debugging: print the incoming data
        print(stored_password)  # Debugging: print the fetched password

        # Validate the incoming password with the fetched one
        if data.get('password') == stored_password:
            print('Correct password')

            # Query to fetch all user scores where app=1
            cursor.execute("SELECT * FROM user_scores WHERE app='1'")
            scores = cursor.fetchall()

            # Convert results into a JSON-compatible format
            results = [
                dict(zip([desc[0] for desc in cursor.description], row)) for row in scores
            ]

            return jsonify(results)  # Return the results as JSON

        else:
            print('Invalid password')
            return jsonify({"error": "Invalid password"}), 403

    # Handle the GET request without password validation
    elif request.method == 'GET':
        cursor.execute("SELECT * FROM user_scores WHERE app='1'")
        scores = cursor.fetchall()

        results = [
            dict(zip([desc[0] for desc in cursor.description], row)) for row in scores
        ]
        return jsonify(results)  # Return the results as JSON

@app.route('/api/app_testscores', methods=['GET', 'POST'])
def app_testscores():
    app.config["DEBUG"] = True
    app.secret_key = "789456asd"
    user = 'hbnnarzullayev'
    password = 'Sersarson7'
    host = 'hbnnarzullayev.mysql.pythonanywhere-services.com'
    mysql = 'hbnnarzullayev$flask3'
    dbs = msd.connect(host, user, password, mysql)
    cursor = dbs.cursor()

    # Handle the case when method is POST, and request contains JSON body
    if request.method == 'POST':
        data = request.get_json()  # This is valid for POST, not GET

        # Fetch the password for the 'Teacher' from the database
        cursor.execute("SELECT password FROM users2 WHERE user_role=1")
        passwords = cursor.fetchall()  # Fetch all passwords for users with user_role = 1
        print(passwords)

        # Extract passwords into a list
        stored_passwords = [password[0] for password in passwords]  # Extracting passwords from tuples

        print(data)  # Debugging: print the incoming data
        print(stored_passwords)

        # Validate the incoming password with the fetched one
        if data.get('password') in stored_passwords:
            print('Correct password')
            app_id=data.get('app_id')
            # Query to fetch all user scores where app=1
            cursor.execute("SELECT * FROM user_scores WHERE app=%s", (app_id,))
            scores = cursor.fetchall()

            # Convert results into a JSON-compatible format
            results = [
                dict(zip([desc[0] for desc in cursor.description], row)) for row in scores
            ]

            return jsonify(results)  # Return the results as JSON

        else:
            print('Invalid password')
            return jsonify({"error": "Invalid password"}), 403

    # Handle the GET request without password validation
    elif request.method == 'GET':
        cursor.execute("SELECT * FROM user_scores WHERE app='1'")
        scores = cursor.fetchall()

        results = [
            dict(zip([desc[0] for desc in cursor.description], row)) for row in scores
        ]
        return jsonify(results)  # Return the results as JSON

@app.route('/api/testscore', methods=['GET', 'POST'])
def get_test_result():
    app.config["DEBUG"] = True
    app.secret_key = "789456asd"
    user = 'hbnnarzullayev'
    password = 'Sersarson7'
    host = 'hbnnarzullayev.mysql.pythonanywhere-services.com'
    mysql = 'hbnnarzullayev$flask3'
    dbs = msd.connect(host, user, password, mysql)
    cursor = dbs.cursor()

    # Handle the case when method is POST, and request contains JSON body
    if request.method == 'POST':
        data = request.get_json()  # This is valid for POST, not GET
        app_id=data.get('app_id')
        # Fetch the password for the 'Teacher' from the database
        cursor.execute("SELECT password FROM users2 WHERE username='Teacher'")
        passwords = cursor.fetchall()

        # Check if the result is not empty
        if passwords:
            stored_password = passwords[0][0]  # Extract the password from the tuple
        else:
            stored_password = ''

        print(data)  # Debugging: print the incoming data
        print(stored_password)  # Debugging: print the fetched password

        # Validate the incoming password with the fetched one
        if data.get('password') == stored_password:
            print('Correct password')

            # Query to fetch all user scores where app=1
            cursor.execute("SELECT * FROM user_scores WHERE app=%s", (app_id,))
            scores = cursor.fetchall()

            # Convert results into a JSON-compatible format
            results = [
                dict(zip([desc[0] for desc in cursor.description], row)) for row in scores
            ]

            return jsonify(results)  # Return the results as JSON

        else:
            print('Invalid password')
            return jsonify({"error": "Invalid password"}), 403

    # Handle the GET request without password validation
    elif request.method == 'GET':
        cursor.execute("SELECT * FROM user_scores WHERE app=%s", (app_id,))
        scores = cursor.fetchall()

        results = [
            dict(zip([desc[0] for desc in cursor.description], row)) for row in scores
        ]
        return jsonify(results)  # Return the results as JSON


@app.route('/api/mine_results', methods=['POST'])
def get_mine_results():
    app.config["DEBUG"] = True
    app.secret_key = "789456asd"
    user = 'hbnnarzullayev'
    password = 'Sersarson7'
    host = 'hbnnarzullayev.mysql.pythonanywhere-services.com'
    mysql = 'hbnnarzullayev$flask3'
    print('Mine')
    try:
        # Ensure request has JSON data
        if not request.is_json:
            return jsonify({"error": "Request must be in JSON format"}), 400

        data = request.get_json()
        print(data)
        username = data.get('user')

        # Validate input
        if not username:
            return jsonify({"error": "Username is required"}), 400

        # Connect to the database
        dbs = msd.connect(host, user, password, mysql)
        cursor = dbs.cursor()

        # Query to fetch user-specific results
        query = """
        SELECT *
        FROM user_scores
        WHERE user = %s and app='1'
        """
        cursor.execute(query, (username,))
        results = cursor.fetchall()

        # Check if results exist
        if not results:
            print('MYSQL xato')
            return jsonify({"message": "No results found for the user"}), 404

        # Convert results into a JSON-compatible format
        response_data = [
            dict(zip([desc[0] for desc in cursor.description], row)) for row in results
        ]

        # Close the connection
        cursor.close()
        dbs.close()
        print('json sent')
        return jsonify(response_data), 200

    except Exception as e:
        # Handle errors
        return jsonify({"error": str(e)}), 500

@app.route('/api/mine_result', methods=['POST'])
def mine_result():
    app.config["DEBUG"] = True
    app.secret_key = "789456asd"
    user = 'hbnnarzullayev'
    password = 'Sersarson7'
    host = 'hbnnarzullayev.mysql.pythonanywhere-services.com'
    mysql = 'hbnnarzullayev$flask3'
    print('Mine')
    try:
        # Ensure request has JSON data
        if not request.is_json:
            return jsonify({"error": "Request must be in JSON format"}), 400

        data = request.get_json()
        print(data)
        username = data.get('user')
        app_id = data.get('app_id')

        # Validate input
        if not username:
            return jsonify({"error": "Username is required"}), 400

        # Connect to the database
        dbs = msd.connect(host, user, password, mysql)
        cursor = dbs.cursor()

        # Query to fetch user-specific results
        query = """
        SELECT *
        FROM user_scores
        WHERE user = %s and app=%s
        """
        cursor.execute(query, (username,app_id, ))
        results = cursor.fetchall()

        # Check if results exist
        if not results:
            print('MYSQL xato')
            return jsonify({"message": "No results found for the user"}), 404

        # Convert results into a JSON-compatible format
        response_data = [
            dict(zip([desc[0] for desc in cursor.description], row)) for row in results
        ]

        # Close the connection
        cursor.close()
        dbs.close()
        print('json sent')
        return jsonify(response_data), 200

    except Exception as e:
        # Handle errors
        return jsonify({"error": str(e)}), 500

@app.route('/api/questions', methods=['GET'])
def get_questions():
    app.config["DEBUG"] = True
    app.secret_key = "789456asd"
    user = 'hbnnarzullayev'
    password = 'Sersarson7'
    host = 'hbnnarzullayev.mysql.pythonanywhere-services.com'
    mysql = 'hbnnarzullayev$flask3'
    dbs = msd.connect(host,user,password, mysql)
    questions = [
        {
            'question': '1  O’rta asrlarda (10 - 11) O’rta Osiyoda parazit hayvonlar haqida ilmiy  ma’lumotlarni yaratgan mutafakkirlar?',
            'choices': ['Firdavsiy, Beruniy, Abu Ali Ibn Sino', 'Umar Hayyom, Beruniy, Firdavsiy', 'Firdavsiy, K.A. Rudolf, Abu Ali Ibn Sino', 'Umar Hayyom, Karl Linney, Fransesko Redi'],
            'correct_answer': 'Umar Hayyom, Karl Linney, Fransesko Redi'
        },
        {
            'question': '2. Tirik organizmlarning o’zaro munosabat shakllari:',
            'choices': ['Mutualizm, kommensalizm, forez, yirtqichlik, parazitizm', 'Simbioz, sinoykiya, ektoparazitizm, paraoykiya, forez', 'Simbioz, antibioz,  kvartirantlik, epioykiya, parazitlik', 'Parazitizm, forez, simbioz, mutualizm, entoykiya.'],
            'correct_answer': 'Parazitizm, forez, simbioz, mutualizm, entoykiya.'
        },
        {
            'question': '3. 20 asrning 2-yarmida O’zbekistonda parazitologiya  fanini rivojlanishida o’z hissalar ini qo’shgan o’zbek olimlari:',
            'choices': ['Sultonov M.A. Azimov Sh.A. , To’laganov A. T. , Ergashev E.X. , R.R. Magdiyev., Abdiyev T.A.', 'To’laganov A.T., J.A.Azimov, Zufarov T. Z., S. O. Osmonov, T.Z.Zokirov.', 'Xamroyev A.B., Xaitov M.X., Muxammadiyev A.M., Zokirov Q..Z.', 'Olimjanov R. A., Salimov B. S., Shopo’latov J.Sh., N.M.Matchanov'],
            'correct_answer': 'Olimjanov R. A., Salimov B. S., Shopo’latov J.Sh., N.M.Matchanov'
        },
        {
            'question': 'What is the capital of France?',
            'choices': ['London', 'Berlin', 'Paris', 'Madrid'],
            'correct_answer': 'Paris'
        }
        # Add more questions
    ]
    return jsonify(questions)

@app.route("/api/scores", methods=["POST", "GET"])
def scores():
    data = request.get_json()
    msg = ""
    app.config["DEBUG"] = True
    app.secret_key = "super secret key"
    user = 'hbnnarzullayev'
    password = 'Sersarson7'
    host = 'hbnnarzullayev.mysql.pythonanywhere-services.com'
    mysql = 'hbnnarzullayev$flask3'
    dbs = msd.connect(host,user,password, mysql)
    if request.method == 'POST':
        now = datetime.now(tz=timezone(timedelta(hours=5)))
        score=data.get('score')
        user=data.get('user')
        # hashed_password = sha256_crypt.hash(password)
        cursor = dbs.cursor()
        # token = generate_confirmation_token(user.email)  # Replace with your token generation logic
        # confirm_url = url_for('confirm_email', token=token, _external=True)
        # send_confirmation_email(user.email, confirm_url)  # Send confirmation email

        # flash('A confirmation email has been sent. Please check your inbox.', 'success')
        # print(user)
            # Account doesnt exists and the form data is valid, now insert new account into accounts table
        cursor.execute(''' INSERT INTO scores VALUES(%s,%s,%s)''',[user,score,now.strftime('%Y-%m-%d %H:%M:%S')])
        flash('A confirmation link has been sent to your email','success')
        dbs.commit()
        msg = "Muvaffaqqiyatli ro'yxatdan o'tdingiz!"
    return render_template('register.html', msg=msg)

@app.route('/validate', methods=["POST"])
def validate():
    user_otp=request.form['otp']
    if otp==int(user_otp):
        return "Succes"
    else:
        return "Fail"

@app.route('/confirm/<token>')
def confirm_email(token):
    # Verify the token and update the user's account status in the database
    # ...

    flash('Your email has been confirmed. You can now log in.', 'success')
    return redirect(url_for('login'))

def send_confirmation_email(to_email, confirm_url):
    subject = 'Confirm Your Email'
    body = f'Click the following link to confirm your email: {confirm_url}'
    msg = Message(subject, recipients=[to_email], body=body)
    mail.send(msg)

@app.route('/login_google')
def login_google():
    return google.authorize(callback=url_for('login', _external=True))

@app.route("/login", methods=["POST", "GET"])
def login():
    error = ''
    app.config["DEBUG"] = True
    app.secret_key = "789456asd"
    user = 'hbnnarzullayev'
    password = 'Sersarson7'
    host = 'hbnnarzullayev.mysql.pythonanywhere-services.com'
    mysql = 'hbnnarzullayev$flask3'
    dbs = msd.connect(host,user,password, mysql)
    if 'loggedin' in session:
        error = 'Siz akkountga kirgansiz!'
        return render_template('profile.html', error=error)
    elif request.method == 'POST':
        username = request.form['nma']
        password = request.form['pwda']
        session["nma"]=username
        session.permanent = True
        cur = dbs.cursor()

        #cur = mysql.connection.cursor()
        cur.execute("SELECT * FROM users2 WHERE username = %s and password = %s", (username,password,))
        user = cur.fetchall()
        #cur.close()
        print(user)
        if user:
            session['loggedin'] = True
            session['nma'] = username
            session['pwda'] = password
            user_obj = User(user[0])
            login_user(user_obj)
            return render_template('profile.html', value=user)
        else:
            flash("Agar yangi foydalanuvchi bo'lsangiz")
            error = 'Username(nik) yoki parol xato!'
            return render_template('login.html', error=error)

    return render_template('login.html')

@app.route("/logins", methods=["POST", "GET"])
def logins():
    data = request.get_json()
    error = ''
    app.config["DEBUG"] = True
    app.secret_key = "789456asd"
    user = 'hbnnarzullayev'
    password = 'Sersarson7'
    host = 'hbnnarzullayev.mysql.pythonanywhere-services.com'
    mysql = 'hbnnarzullayev$flask3'
    dbs = msd.connect(host,user,password, mysql)
    if 'loggedin' in session:
        error = 'Siz akkountga kirgansiz!'
        return render_template('profile.html', error=error)
    elif request.method == 'POST':
        username = data.get('nma')
        password = data.get('pwda')
        session["nma"]=username
        session.permanent = True
        cur = dbs.cursor()

        #cur = mysql.connection.cursor()
        cur.execute("SELECT email FROM users2 WHERE username = %s and password = %s", (username,password,))
        emails = cur.fetchone()
        if emails is not None:
            email = emails[0]
        else:
            email = ''
        cur.execute("SELECT user_role FROM users2 WHERE username = %s and password = %s", (username,password,))
        admins = cur.fetchone()
        if admins is not None:
            adminv = admins[0]
        else:
            adminv = ''
        if adminv == 1:
            adminv = 'true'
        else:
            adminv = 'false'
        cur.execute("SELECT * FROM users2 WHERE username = %s and password = %s", (username,password,))
        user = cur.fetchall()
        cur.execute("SELECT sum(score) FROM scores WHERE user = %s", (username,))
        summ_score1 = cur.fetchone()
        if summ_score1 is not None:
            summ_score = summ_score1[0]
        else:
            summ_score = 0
        cur.execute("SELECT count(score) FROM scores WHERE user = %s", (username,))
        count_score1 = cur.fetchone()
        if count_score1 is not None:
            count_score = count_score1[0]
        else:
            count_score = 0
        cur.execute("SELECT score FROM scores WHERE user = %s order by datetime desc limit 1", (username,))
        score1 = cur.fetchone()
        if score1 is not None:
            last_score = score1[0]
        else:
            last_score = 0
        #cur.close()
        print(user)
        if user:
            session['loggedin'] = True
            session['nma'] = username
            session['pwda'] = password
            user_obj = User(user[0])
            login_user(user_obj)
            print("adminv:")
            print(adminv)
            return jsonify({"username": username, "email": email, "summ_score": summ_score,"count_score": count_score,"last_score": last_score, "password": password, 'adminv': adminv})
        else:
            return jsonify({"error": "Invalid credentials"}), 401

    return render_template('logins.html')

@app.route("/app_logins", methods=["POST", "GET"])
def app_logins():
    data = request.get_json()
    error = ''
    app.config["DEBUG"] = True
    app.secret_key = "789456asd"
    user = 'hbnnarzullayev'
    password = 'Sersarson7'
    host = 'hbnnarzullayev.mysql.pythonanywhere-services.com'
    mysql = 'hbnnarzullayev$flask3'
    dbs = msd.connect(host,user,password, mysql)
    if 'loggedin' in session:
        error = 'Siz akkountga kirgansiz!'
        return render_template('profile.html', error=error)
    elif request.method == 'POST':
        username = data.get('nma')
        password = data.get('pwda')
        app_id = data.get('app_id')
        session["nma"]=username
        session.permanent = True
        cur = dbs.cursor()

        #cur = mysql.connection.cursor()
        cur.execute("SELECT email FROM users2 WHERE username = %s and password = %s and app_id=%s", (username,password,app_id, ))
        email = cur.fetchone()[0]
        cur.execute("SELECT * FROM users2 WHERE username = %s and password = %s and app_id=%s", (username,password,app_id, ))
        user = cur.fetchall()
        cur.execute("SELECT sum(score) FROM scores WHERE user = %s", (username,))
        summ_score1 = cur.fetchone()
        if summ_score1 is not None:
            summ_score = summ_score1[0]
        else:
            summ_score = 0
        cur.execute("SELECT count(score) FROM scores WHERE user = %s", (username,))
        count_score1 = cur.fetchone()
        if count_score1 is not None:
            count_score = count_score1[0]
        else:
            count_score = 0
        cur.execute("SELECT score FROM scores WHERE user = %s order by datetime desc limit 1", (username,))
        score1 = cur.fetchone()
        if score1 is not None:
            last_score = score1[0]
        else:
            last_score = 0
        #cur.close()
        print(user)
        if user:
            session['loggedin'] = True
            session['nma'] = username
            session['pwda'] = password
            user_obj = User(user[0])
            login_user(user_obj)
            return jsonify({"username": username, "email": email, "summ_score": summ_score,"count_score": count_score,"last_score": last_score, "password": password})
        else:
            return jsonify({"error": "Invalid credentials"}), 401

    return render_template('logins.html')

@app.route("/users_scores", methods=["POST", "GET"])
def users_scores():
    data = request.get_json()
    error = ''
    app.config["DEBUG"] = True
    app.secret_key = "789456asd"
    user = 'hbnnarzullayev'
    password = 'Sersarson7'
    host = 'hbnnarzullayev.mysql.pythonanywhere-services.com'
    mysql = 'hbnnarzullayev$flask3'
    dbs = msd.connect(host,user,password, mysql)
    if 'loggedin' in session:
        error = 'Siz akkountga kirgansiz!'
        return render_template('profile.html', error=error)
    elif request.method == 'POST':
        username = data.get('nms')
        app_data = data.get('app')
        session["nms"]=username
        session.permanent = True
        cur = dbs.cursor()

        # #cur = mysql.connection.cursor()
        # cur.execute("SELECT email FROM users2 WHERE username = %s and password = %s", (username,))
        # email = cur.fetchone()[0]
        # cur.execute("SELECT * FROM users2 WHERE username = %s and password = %s", (username,))
        # user = cur.fetchall()
        cur.execute("SELECT sum(score) FROM user_scores WHERE user = %s and app = %s", (username, app_data,))
        summ_score = cur.fetchone()[0]
        cur.execute("SELECT count(score) FROM user_scores WHERE user = %s and app = %s", (username, app_data,))
        count_score = cur.fetchone()[0]
        cur.execute("SELECT score FROM user_scores WHERE user = %s and app = %s order by datetime desc limit 1", (username, app_data,))
        last_score = cur.fetchone()
        if last_score is not None:
            last_score = last_score[0]
        else:
            last_score='0'
        #cur.close()
        print(user)
        if summ_score:
            return jsonify({"username": username, "summ_score": summ_score,"count_score": count_score,"last_score": last_score})
        else:
            return jsonify({"error": "Invalid credentials"}), 401

    return render_template('logins.html')

@app.route("/userscores", methods=["POST", "GET"])
def userscores():
    data = request.get_json()
    error = ''
    app.config["DEBUG"] = True
    app.secret_key = "789456asd"
    user = 'hbnnarzullayev'
    password = 'Sersarson7'
    host = 'hbnnarzullayev.mysql.pythonanywhere-services.com'
    mysql = 'hbnnarzullayev$flask3'
    dbs = msd.connect(host,user,password, mysql)
    if 'loggedin' in session:
        error = 'Siz akkountga kirgansiz!'
        return render_template('profile.html', error=error)
    elif request.method == 'POST':
        username = data.get('nms')
        session["nms"]=username
        session.permanent = True
        cur = dbs.cursor()

        # #cur = mysql.connection.cursor()
        # cur.execute("SELECT email FROM users2 WHERE username = %s and password = %s", (username,))
        # email = cur.fetchone()[0]
        # cur.execute("SELECT * FROM users2 WHERE username = %s and password = %s", (username,))
        # user = cur.fetchall()
        cur.execute("SELECT sum(score) FROM scores WHERE user = %s", (username,))
        summ_score = cur.fetchone()[0]
        cur.execute("SELECT count(score) FROM scores WHERE user = %s", (username,))
        count_score = cur.fetchone()[0]
        cur.execute("SELECT score FROM scores WHERE user = %s order by datetime desc limit 1", (username,))
        last_score = cur.fetchone()[0]
        #cur.close()
        print(user)
        if summ_score:
            return jsonify({"username": username, "summ_score": summ_score,"count_score": count_score,"last_score": last_score})
        else:
            return jsonify({"error": "Invalid credentials"}), 401

    return render_template('logins.html')

@app.route('/logout')
def logout():
    app.secret_key = "789456asd"
    session.clear()
    return redirect(url_for('login'))

@app.route('/delete')
def delete():
    msg = ''
    app.config["DEBUG"] = True
    app.secret_key = "789456asd"
    user = 'hbnnarzullayev'
    password = 'Sersarson7'
    host = 'hbnnarzullayev.mysql.pythonanywhere-services.com'
    mysql = 'hbnnarzullayev$flask3'
    dbs = msd.connect(host,user,password, mysql)
    # Check if user is loggedin
    if 'loggedin' in session and session['nma'] not in ('admin','Admin'):
        # We need all the account info for the user so we can display it on the profile page
        cur = dbs.cursor()
        cur.execute("DELETE FROM users2 where username = '%s' and password = '%s'" % (session['nma'],session['pwda']))
        dbs.commit()
        cur.close()
        session['loggedin'] = False
        #flash('%s deleted'(session['nma']), 'success')
        # Show the profile page with account info
        return render_template('index.html', user=user)
    else:
        msg = 'not logged in'
    # User is not loggedin redirect to login page
    return render_template('profile.html', msg=msg)

@app.route('/upload_excels', methods=['POST'])
def upload_excels():
    if 'file' not in request.files:
        return 'No file part'

    file = request.files['file']
    if file.filename == '':
        return 'No selected file'

    if file:
        df = pd.read_excel(file)
        # Check if the text in the Excel file should be converted from Latin to Cyrillic
        if request.form.get('convert_to') == 'cyrillic':
            df = df.applymap(latin_to_cyrillic)
        # Check if the text in the Excel file should be converted from Cyrillic to Latin
        elif request.form.get('convert_to') == 'latin':
            df = df.applymap(cyrillic_to_latin)

        output = io.BytesIO()
        writer = pd.ExcelWriter(output, engine='xlsxwriter')
        df.to_excel(writer, index=False)
        writer.save()
        output.seek(0)
        return send_file(output, attachment_filename='converted_file.xlsx', as_attachment=True)

def cyrillic_to_latin(text):
    if not isinstance(text, str):
        return text
    cyrillic_to_latin_map = {
        'а': 'a',
        'б': 'b',
        'д': 'd',
        'е': 'e',
        'ф': 'f',
        'г': 'g',
        'ҳ': 'h',
        'и': 'i',
        'ж': 'j',
        'к': 'k',
        'л': 'l',
        'м': 'm',
        'н': 'n',
        'о': 'o',
        'п': 'p',
        'қ': 'q',
        'р': 'r',
        'с': 's',
        'т': 't',
        'у': 'u',
        'в': 'v',
        'х': 'x',
        'й': 'y',
        'з': 'z',
        'ч': 'ch',
        'ш': 'sh',
        'ю': 'yu',
        'я': 'ya',
    }
    #
    # Perform conversion
    latin_text = ''
    for char in text:
        latin_text += cyrillic_to_latin_map.get(char, char)
    return latin_text
    # return text

def latin_to_cyrillic(text):
    if not isinstance(text, str):
        return text
    latin_to_cyrillic_map = {
        'a': 'а',
        'b': 'б',
        'd': 'д',
        'e': 'е',
        'f': 'ф',
        'g': 'г',
        'h': 'ҳ',
        'i': 'и',
        'j': 'ж',
        'k': 'к',
        'l': 'л',
        'm': 'м',
        'n': 'н',
        'o': 'о',
        'p': 'п',
        'q': 'қ',
        'r': 'р',
        's': 'с',
        't': 'т',
        'u': 'у',
        'v': 'в',
        'x': 'х',
        'y': 'й',
        'z': 'з',
        'ch': 'ч',
        'sh': 'ш',
        'ye': 'е',
        'yu': 'ю',
        'ya': 'я',
    }
    # Perform conversion
    cyrillic_text = ''
    for char in text:
        cyrillic_text += latin_to_cyrillic_map.get(char, char)
    return cyrillic_text

@app.route('/upload_excel', methods=['GET', 'POST'])
def upload_excel():
    if 'file' not in request.files:
        return render_template('cyrillic.html')

    file = request.files['file']
    if file.filename == '':
        return 'No selected file'

    if file:
        df = pd.read_excel(file)
        # Check if the text in the Excel file should be converted from Latin to Cyrillic
        if request.form.get('convert_to') == 'cyrillic':
            df = df.applymap(latin_to_cyrillic)
        # Check if the text in the Excel file should be converted from Cyrillic to Latin
        elif request.form.get('convert_to') == 'latin':
            df = df.applymap(cyrillic_to_latin)

        output = io.BytesIO()
        writer = pd.ExcelWriter(output, engine='xlsxwriter')
        df.to_excel(writer, index=False)
        writer.save()
        output.seek(0)
        return send_file(output, attachment_filename='converted_file.xlsx', as_attachment=True)
    return render_template('cyrillic.html')

@app.route('/show', methods=['GET', 'POST'])
def show():
    msg = ''
    app.config["DEBUG"] = True
    app.secret_key = "super secret key"
    user = 'hbnnarzullayev'
    password = 'Sersarson7'
    host = 'hbnnarzullayev.mysql.pythonanywhere-services.com'
    mysql = 'hbnnarzullayev$flask3'
    dbs = msd.connect(host,user,password, mysql)
    # Check if user is loggedin#
    if (request.method == 'POST'):
        table_name2 = request.form['table_name2']
        cur = dbs.cursor()
        cur.execute(f"select * from {table_name2}")
        data=cur.fetchall()
        column_names = [desc[0] for desc in cur.description]
        cur.execute("SHOW TABLES WHERE `Tables_in_hbnnarzullayev$flask3` in ('min_zp','na2','penya','questions','ulush');")
        tables = [row[0] for row in cur.fetchall()]
        return render_template('show.html', values=data, column_names=column_names, tables=tables)

    cur = dbs.cursor()

    # Get list of table names from the database
    cur.execute("SHOW TABLES WHERE `Tables_in_hbnnarzullayev$flask3` NOT in ('users2','nl','nla','Images','messages');")
    tables = [row[0] for row in cur.fetchall()]
    return render_template('show.html', tables=tables)

@app.route('/upload', methods=['GET', 'POST'])
#@login_required
def upload():
    msg=''
    app.config["DEBUG"] = True
    app.secret_key = "789456asd"
    user = 'hbnnarzullayev'
    password = 'Sersarson7'
    host = 'hbnnarzullayev.mysql.pythonanywhere-services.com'
    mysql = 'hbnnarzullayev$flask3'
    dbs = msd.connect(host,user,password, mysql)
    if (request.method == 'POST') and (session['nma'] in ('admin','Admin')) and (session['pwda'] in ('123','123')):
        file = request.files['file']
        if 'file' not in request.files:
            return 'No file uploaded'
        if file.filename == '':
            return 'No file selected'

        table_name = request.form['table_name']
        # table_name1 = request.form['table_name1']

        try:
            msg="Yuklandi!"
            cur = dbs.cursor()

            # Get list of table names from the database
            cur.execute("SHOW TABLES")
            tables = [row[0] for row in cur.fetchall()]
            # Read Excel data into a pandas DataFrame
            df = pd.read_excel(file)
            df = df.where(pd.notna(df), None)

            # Establish MySQL connection
            conn = MySQLdb.connect("hbnnarzullayev.mysql.pythonanywhere-services.com","hbnnarzullayev","Sersarson7","hbnnarzullayev$flask3" )
            cursor = conn.cursor()
            create_table_sql = f"CREATE TABLE IF NOT EXISTS {table_name} ({', '.join([f'{col} VARCHAR(255)' for col in df.columns])})"
            cursor.execute(create_table_sql)
            cursor.execute(f"DELETE FROM {table_name}")
            cursor.execute(f"DESCRIBE {table_name}")
            columns = [row[0] for row in cursor.fetchall()]

            # Insert DataFrame records into MySQL table
            for _, row in df.iterrows():
                # columns = ", ".join(row.index)
                values = tuple(row)
                # placeholders = ", ".join(["%s" if value is not None else "NULL" for value in values] * len(row))
                placeholders = ", ".join(["%s"] * len(row))

                sql = f"INSERT INTO {table_name} ({','.join(columns)}) VALUES ({placeholders})"
                cursor.execute(sql, values)

            # Commit changes and close the connection
            conn.commit()
            cursor.close()
            conn.close()

            return render_template('upload.html',tables=tables, msg=msg, columns=columns)

        except Exception as e:
            return 'Error: ' + str(e)
        # except Exception as e:
        #     conn.rollback()
        #     # print(f"Error executing query: {query}")
        #     print(str(e))
        # return render_template('upload.html',tables=tables)

    elif (request.method == 'POST') and (session['nma'] in ('admin','Admin')) and (session['pwda'] in ('123','123')):
        table_name1 = request.form['table_name1']
        cur = dbs.cursor()
        cur.execute(f"select * from {table_name1}")
        data=cur.fetchall()
        return render_template('upload.html', values=data)
    elif (session['nma'] in ('admin','Admin')) and (session['pwda'] in ('123','123')):
        cur = dbs.cursor()

        # Get list of table names from the database
        cur.execute("SHOW TABLES")
        tables = [row[0] for row in cur.fetchall()]
        return render_template('upload.html', tables=tables)
    else:
        return render_template('index.html', tables=tables)
    #return render_template('upload.html', tables=tables)

@app.route('/update_login', methods=['GET', 'POST'])
def update_login():
    app.config["DEBUG"] = True
    app.secret_key = "789456asd"
    user = 'hbnnarzullayev'
    password = 'Sersarson7'
    host = 'hbnnarzullayev.mysql.pythonanywhere-services.com'
    mysql = 'hbnnarzullayev$flask3'
    dbs = msd.connect(host,user,password, mysql)
    if 'loggedin' not in session:
         return redirect(url_for('login'))

    user_name = session['nma']
    pass_word = session['pwda']
    new_username = request.form['new_username']
    new_password = request.form['new_password']

    cursor = dbs.cursor()
    cursor.execute(
        "update users2 set username=%s, password=%s where username = %s and password=%s",
        (new_username, new_password, user_name, pass_word)
    )
    dbs.commit()
    cursor.close()
    return render_template('profile.html')

@app.route('/download', methods=['GET', 'POST'])
def download():
    # filename=request.form['filename']
    # path = '/home/hbnnarzullayev/mysite/uploads/'+filename
    # return send_file(path, as_attachment=True)
    query = request.form.get('filename')  # Get the SQL query from the form
    filename='result.xlsx'
    df = execute_query_to_dataframe("select * from %s" % (query))
    excel_file_path = '/home/hbnnarzullayev/mysite/script/'
    df.to_excel(excel_file_path+filename, index=False)
    return send_file(excel_file_path+filename, as_attachment=True)

@app.route('/downloada', methods=['POST','GET'])
def downloada():
    if request.method == "POST":
        query = request.form.get('query')  # Get the SQL query from the form
        filename='result.xlsx'
        df = execute_query_to_dataframe(query)
        excel_file_path = '/home/hbnnarzullayev/mysite/script/'
        df.to_excel(excel_file_path+filename, index=False)
        return send_file(excel_file_path+filename, as_attachment=True)
    return render_template("downloada.html")

@app.route('/get-column-names', methods=['POST'])
def get_column_names():
    table_name = request.json['table_name']

    # Connect to MySQL database
    cur = mysql.connection.cursor()

    # Get column names of the selected table
    cur.execute(f"DESCRIBE {table_name}")
    columns = [row[0] for row in cur.fetchall()]

    return {'column_names': columns}

@app.route("/test_form", methods=["POST", "GET"])
def test_form():
    msg = ''
    if request.method == "POST" and request.form["nma"] == "admin" and request.form["pwda"] != "":
        #user = request.form["nm"]
        return redirect(url_for("profile"))
    else:
        return render_template("test_form.html")

@app.route('/chat', methods=['GET', 'POST'])
def chat():
    app.config["DEBUG"] = True
    app.secret_key = "789456asd"
    user = 'hbnnarzullayev'
    password = 'Sersarson7'
    host = 'hbnnarzullayev.mysql.pythonanywhere-services.com'
    mysql = 'hbnnarzullayev$flask3'
    dbs = msd.connect(host,user,password, mysql)
    if 'loggedin' not in session:

        return render_template('login.html')

    elif 'loggedin' in session and request.method == 'POST':
        receiver=request.form["table_name2"]
        cur = dbs.cursor()
        cur.execute(f"SELECT * FROM messages WHERE ((sender = %s and receiver = %s) or (sender = %s and receiver = %s))", (session['nma'],receiver,receiver,session['nma']))
        data = cur.fetchall()
        cur.execute("select * from users2")
        tables = [row[0] for row in cur.fetchall()]
        cur.execute("select * from messages where sender=%s and receiver=%s", (session['nma'],receiver))
        message_ids = [row[0] for row in cur.fetchall()]
        return render_template('chat.html', values=data, tables=tables, message_ids=message_ids)
    cur = dbs.cursor()

    # Get list of table names from the database
    cur.execute("select * from users2")
    tables = [row[0] for row in cur.fetchall()]
    cur.execute("select * from messages where (sender=%s or receiver=%s)", (session['nma'],session['nma'],))
    message_ids = [row[0] for row in cur.fetchall()]
    cur.execute("SELECT * FROM messages WHERE sender = %s or receiver = %s", (session['nma'],session['nma'],))
    data=cur.fetchall()

    return render_template('chat.html', tables=tables, values=data, message_ids=message_ids)

@app.route('/delete/<int:message_id>', methods=['GET'])
def delete_messagea(message_id):
    if 'loggedin' not in session:
        return redirect('/login')
    user = 'hbnnarzullayev'
    password = 'Sersarson7'
    host = 'hbnnarzullayev.mysql.pythonanywhere-services.com'
    mysql_db = 'hbnnarzullayev$flask3'
    # Connect to the database
    dbs = msd.connect(host=host, user=user, password=password, database=mysql_db)

    cursor = dbs.cursor()
    UPLOAD_FOLDER = '/home/hbnnarzullayev/mysite/uploads/saved_files/'

    # Fayl nomini bazadan olish
    cursor.execute("SELECT file_name FROM messages WHERE ID = %s", (message_id,))
    file = cursor.fetchone()
    print("file:")
    print(file)
    if file:
        file_name = file[0]
        file_path = os.path.join('/home/hbnnarzullayev/mysite/uploads/saved_files/', file_name)
        print(file_path)
        os.remove(file_path)
        cursor.execute("DELETE FROM messages WHERE id = %s", (message_id,))
        # Serverdan faylni o'chirish
        if os.path.exists(file_path):
            os.remove(file_path)
            cursor.execute("DELETE FROM messages WHERE id = %s", (message_id,))
        dbs.commit()
    dbs.commit()

    return redirect('/chat')


@app.route('/delete_message', methods=['POST'])
def delete_message():
    msg="Ushbu xabar muvaffaqqiyatli o'chirildi"
    app.config["DEBUG"] = True
    app.secret_key = "789456asd"
    user = 'hbnnarzullayev'
    password = 'Sersarson7'
    host = 'hbnnarzullayev.mysql.pythonanywhere-services.com'
    mysql = 'hbnnarzullayev$flask3'
    dbs = msd.connect(host,user,password, mysql)
    if 'loggedin' not in session:
         return redirect(url_for('login'))

    sender = session['nma']
    message = request.form['message_id']

    cursor = dbs.cursor()
    cursor.execute(
        "DELETE FROM messages where sender=%s and id=%s",(sender, message)
    )
    dbs.commit()
    cursor.close()
    cur = dbs.cursor()

    # Get list of table names from the database
    cur.execute("select * from users2")
    tables = [row[0] for row in cur.fetchall()]
    cur.execute("select * from messages where (sender=%s or receiver=%s)", (session['nma'],session['nma'],))
    message_ids = [row[0] for row in cur.fetchall()]
    cur.execute("SELECT * FROM messages WHERE sender = %s or receiver = %s", (session['nma'],session['nma'],))
    data=cur.fetchall()

    return render_template('chat.html', msg=msg, tables=tables, values=data, message_ids=message_ids)

@app.route('/send_message', methods=['POST'])
def send_message():
    # parol = request.form['password']
    app.config["DEBUG"] = True
    app.secret_key = "789456asd"
    user = 'hbnnarzullayev'
    password = 'Sersarson7'
    host = 'hbnnarzullayev.mysql.pythonanywhere-services.com'
    mysql = 'hbnnarzullayev$flask3'
    dbs = msd.connect(host,user,password, mysql)
    if 'loggedin' not in session:
         return redirect(url_for('login'))
    # ALLOWED_EXTENSIONS = {'xls', 'xlsx', 'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}
    if 'file' not in request.files:
        return 'Fayl topilmadi'
    file = request.files['file']

    # Fayl nomi bo'sh emasligini tekshirish
    if file.filename == '':
        return 'Fayl tanlanmadi'
    if file and allowed_file(file.filename):
        sender = session['nma']
        receiver = request.form['receiver']
        message = request.form['message']
        created_at = datetime.now()
        file_pkey = generate_random_pkey()
        now = datetime.now(tz=timezone(timedelta(hours=5)))
        user = 'hbnnarzullayev'
        password = 'Sersarson7'
        host = 'hbnnarzullayev.mysql.pythonanywhere-services.com'
        mysql_db = 'hbnnarzullayev$flask3'
        # Connect to the database
        dbs = msd.connect(host=host, user=user, password=password, database=mysql_db)
        cursor = dbs.cursor()
        s_filename = file_pkey + '-' + file.filename
        # cursor.execute('''INSERT INTO saved_files (file_name, username, date_time, pkey) VALUES (%s, %s, %s, %s)''',
                #   [s_filename, session.get("nma"), now.strftime('%Y-%m-%d %H:%M:%S'), file_pkey])
        cursor.execute(
            "INSERT INTO messages (sender, receiver, message, created_at, file_name) VALUES (%s, %s, %s, %s, %s)",
            (sender, receiver, message, created_at, s_filename)
        )
        dbs.commit()
        cursor.close()
        dbs.close()
        file_path = f"/home/hbnnarzullayev/mysite/uploads/saved_files/"+s_filename
        file.save(file_path)
        return redirect(url_for('chat'))
    return redirect(url_for('chat'))

@app.route('/get_messages', methods=['POST'])
def get_messages():
    app.config["DEBUG"] = True
    app.secret_key = "789456asd"
    user = 'hbnnarzullayev'
    password = 'Sersarson7'
    host = 'hbnnarzullayev.mysql.pythonanywhere-services.com'
    mysql = 'hbnnarzullayev$flask3'
    dbs = msd.connect(host,user,password, mysql)
    if 'loggedin' not in session:
         return redirect(url_for('login'))
    else:
        sender = session['nma']
        receiver = request.form.get('receiver')

        cursor = dbs.cursor()
        cursor.execute(
            "SELECT * FROM messages WHERE (sender = %s AND receiver = %s) OR (sender = %s AND receiver = %s) ORDER BY created_at",
            (sender, receiver, receiver, sender)
        )
        messages = cursor.fetchall()
        cursor.close()

        return render_template('chat.html', values=messages)

@app.route('/music')
#@login_required
def music():
    app.config["DEBUG"] = True
    app.secret_key = "789456asd"
    return render_template('music.html')

@app.route("/profile", methods=['POST','GET'])
def profile():
    app.config["DEBUG"] = True
    app.secret_key = "super secret key"
    username = 'hbnnarzullayev'
    password = 'Sersarson7'
    host = 'hbnnarzullayev.mysql.pythonanywhere-services.com'
    mysql = 'hbnnarzullayev$flask3'
    dbs = msd.connect(host,username,password, mysql)
    # Check if user is loggedin
    if 'loggedin' in session:
        cursor = dbs.cursor()
        cursor.execute('SELECT * FROM users2 WHERE username = %s and password = %s', (session['nma'],session['pwda']))
        data = cursor.fetchall()
        rowid = [row[3] for row in cursor.fetchall()]
        for row in data:
            aa=row[3]
        if request.method=="POST" and request.files['file']:
        # Get the uploaded file
            file = request.files['file']
            #table_name = request.form['table_name']
            filename = secure_filename(file.filename)
            file.save('/home/hbnnarzullayev/mysite/static/Image/' + session['nma']+'.jpg', (rowid))

            cur = dbs.cursor()
            created_at = datetime.now()
            query = "INSERT INTO Images (filename, id, username, Date) VALUES (%s,%s,%s,%s)"
            cur.execute(query, (filename,aa,session['nma'],created_at,))
            #cur.execute("INSERT INTO Images (filename,id) VALUES (%s,%s)", (filename,rowid))


            # Commit the changes and close the cursor
            dbs.commit()
            cur.close()
            return render_template('profile.html', value=data)

        return render_template('profile.html', value=data)
    # User is not loggedin redirect to login page
    else:
        return redirect(url_for('login'))

@app.route('/upload_img', methods=['POST'])
def upload_img():
    app.config["DEBUG"] = True
    app.secret_key = "789456asd"
    user = 'hbnnarzullayev'
    password = 'Sersarson7'
    host = 'hbnnarzullayev.mysql.pythonanywhere-services.com'
    mysql = 'hbnnarzullayev$flask3'
    dbs = msd.connect(host,user,password, mysql)
    cur = dbs.cursor()
    cur.execute("select rowid from users2 where username=%s and password=%s", (session['nma'],session['pwda']))
    data = cur.fetchall()
    rowid = [row[0] for row in cur.fetchall()]
    for row in data:
        aa=row[0]
    if (request.method == 'GET'):
        # Get the uploaded file
        file = request.files['file']
        #table_name = request.form['table_name']
        filename = secure_filename(file.filename)
        # Save the file to a location on the server
        file.save('/home/hbnnarzullayev/mysite/static/image/' + f'{aa}-id '+file.filename, (rowid))

        cur = dbs.cursor()
        query = "INSERT INTO Images (filename, id, username) VALUES (%s,'%s',%s)"
        cur.execute(query, (filename,aa,session['nma'],))
        #cur.execute("INSERT INTO Images (filename,id) VALUES (%s,%s)", (filename,rowid))


        # Commit the changes and close the cursor
        dbs.commit()
        cur.close()

        return render_template('profile.html', value=data) #('Fayl yuklandi va databazaga kiritildi.')

@app.route('/api/upload_img', methods=['POST'])
def upload_img_one():
    print('Keldi')

    # Check if the file and user_id are provided in the request
    if 'photo' not in request.files:
        return jsonify({"error": "No photo uploaded"}), 400

    file = request.files['photo']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Get the user_id from the request
    user_id = request.form.get('user_id')
    if not user_id:
        return jsonify({"error": "User ID not provided"}), 400

    # Generate the filename and save the file
    try:
        filename = f"{user_id}.jpg"
        file_path = os.path.join('/home/hbnnarzullayev/mysite/static/Image/', filename)
        file.save(file_path)

        # Connect to the database
        user = 'hbnnarzullayev'
        password = 'Sersarson7'
        host = 'hbnnarzullayev.mysql.pythonanywhere-services.com'
        mysql = 'hbnnarzullayev$flask3'
        dbs = msd.connect(host, user, password, mysql)
        cur = dbs.cursor()

        # Insert file info into the database
        cur.execute("SELECT rowid FROM users2 WHERE username=%s", (user_id, ))
        datas = cur.fetchall()
        if datas:
            rowid = datas[0][0]
            query = "INSERT INTO Images (filename, id, username) VALUES (%s, %s, %s)"
            cur.execute(query, (filename, rowid, user_id))
            dbs.commit()

        cur.close()
        return jsonify({"message": "Photo uploaded successfully"}), 200
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": "Failed to upload photo"}), 500

@app.route('/api/upload_img_food', methods=['POST'])
def upload_img_food():
    print('Keldi')

    # Check if the file and user_id are provided in the request
    if 'photo' not in request.files:
        return jsonify({"error": "No photo uploaded"}), 400

    file = request.files['photo']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Get the user_id from the request
    group_name = request.form.get('group_name')
    if not user_id:
        return jsonify({"error": "User ID not provided"}), 400

    # Generate the filename and save the file
    try:
        filename = f"{group_name}.jpg"
        file_path = os.path.join('/home/hbnnarzullayev/mysite/static/Image/', filename)
        file.save(file_path)

        # Connect to the database
        user = 'hbnnarzullayev'
        password = 'Sersarson7'
        host = 'hbnnarzullayev.mysql.pythonanywhere-services.com'
        mysql = 'hbnnarzullayev$flask3'
        dbs = msd.connect(host, user, password, mysql)
        cur = dbs.cursor()
        cur.execute("insert into  rowid FROM users2 WHERE username=%s", (user_id, ))

        # Insert file info into the database
        cur.execute("SELECT rowid FROM users2 WHERE username=%s", (user_id, ))
        datas = cur.fetchall()
        if datas:
            rowid = datas[0][0]
            query = "INSERT INTO Images (filename, id, username) VALUES (%s, %s, %s)"
            cur.execute(query, (filename, rowid, user_id))
            dbs.commit()

        cur.close()
        return jsonify({"message": "Photo uploaded successfully"}), 200
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": "Failed to upload photo"}), 500

@app.route('/profile1')
def upload_form():
	return render_template('profile1.html')


@app.route("/yangiliklar1")
def yangiliklar1():
    if request.method == "POST" and request.form["nma"] == "admin":
        #user = request.form["nm"]
        return redirect(url_for("yangiliklar1"))
    else:
        return render_template("yangiliklar1.html")

@app.route("/yangiliklar2")
def yangiliklar2():
    if request.method == "POST" and request.form["nma"] == "admin":
        #user = request.form["nm"]
        pdf_path = "/static/pdf/sayyor.pdf"  # Replace with the actual path of your PDF file
        return render_template('pdf.html', pdf_path=pdf_path)
    else:
        return render_template("pdf.html")

@app.route('/table', methods=["POST", "GET"])
def table():
    msgg = ''
    app.config["DEBUG"] = True
    app.secret_key = "789456asd"
    username = 'hbnnarzullayev'
    password = 'Sersarson7'
    host = 'hbnnarzullayev.mysql.pythonanywhere-services.com'
    mysql = 'hbnnarzullayev$flask3'
    dbs = msd.connect(host,username,password, mysql)
    if 'loggedin' in session:
        msgg = "Akkountga kirilgan!"
        if not session.get("nma") or (session['nma'] not in ('admin','Admin')):
            msgg = "Session lost or not admin!"
            return render_template("index.html", msg=msgg) #, value=data o'chirildi
        elif (session['nma'] in ('admin','Admin')) and (session['pwda'] in ('123')):
        # Connect to the MySQL database
            conn = MySQLdb.connect("hbnnarzullayev.mysql.pythonanywhere-services.com","hbnnarzullayev","Sersarson7","hbnnarzullayev$flask3" )
            cursor = conn.cursor()
            cursor.execute("select * from users2")
            data = cursor.fetchall() #data from database
            #msgg = "Muvaffaqqiyatli ro'yxatdan o'tdingiz11!"
            # #return render_template("table.html", value=data, msg=msgg)
            if request.method == "POST":
                idsa = request.form['ids']
                cursor = dbs.cursor()
                cursor.execute("DELETE FROM users2 where rowid in ('%s') and username!='Admin'" % (idsa))
                flash("Foydalanuvchi ro'yxatdan o'chirildi")
                dbs.commit()
                msgg = "Mazkur ID raqamli foydalanuvch muvaffaqqiyatli o'chirildi!"
                return render_template("table.html", value=data, msg=msgg)
            else:
                msgg = "Admin, hush kelibsiz!"
                return render_template("table.html", msg=msgg, value=data)
            return render_template("table.html", value=data, msg=msgg)
        else:
            msgg = "No admin!"
            return render_template("table.html", msg=msgg)
    return render_template("index.html", msg=msgg)

@app.route('/cut', methods=['GET', 'POST'])
def cut():
    app.secret_key = "789456asd"
    idsa = request.form['ids']
    if request.method == "POST":
        cursor = dbs.cursor()
        cursor.execute("DELETE FROM users2 where rowid in ('%s') and username!='Admin'" % (idsa))
        flash("Foydalanuvchi ro'yxatdan o'chirildi")
        dbs.commit()
        msgg = "Mazkur ID raqamli foydalanuvch muvaffaqqiyatli o'chirildi!"
        return render_template("table.html")
    else:
        msgg = "Admin, hush kelibsiz!"
        return render_template("table.html")

@app.route('/penya', methods=["POST", "GET"])
def penya():
    msgg = ''
    app.config["DEBUG"] = True
    app.secret_key = "super secret key"
    username = 'hbnnarzullayev'
    password = 'Sersarson7'
    host = 'hbnnarzullayev.mysql.pythonanywhere-services.com'
    mysql = 'hbnnarzullayev$flask3'
    dbs = msd.connect(host,username,password, mysql)
    # Get the current page number from the query string parameter
    page = int(request.args.get('page', 1))

    # Number of records to display per page
    per_page = 10

    # Calculate the start and end indices of the records to fetch from the database
    start_index = (page - 1) * per_page
    end_index = start_index + per_page
    # Connect to the MySQL database
    conn = MySQLdb.connect("hbnnarzullayev.mysql.pythonanywhere-services.com","hbnnarzullayev","Sersarson7","hbnnarzullayev$flask3" )
    cursor = conn.cursor()
    cursor.execute("select * from penya LIMIT %s, %s", (start_index, per_page))
    records = cursor.fetchall()

# Count the total number of records in the table
    cursor.execute("SELECT COUNT(*) FROM penya")
    total_records = cursor.fetchone()[0]#data from database
    total_pages = total_records // per_page + (total_records % per_page > 0)
    #msgg = "Muvaffaqqiyatli ro'yxatdan o'tdingiz11!"
    #return render_template("table.html", value=data, msg=msgg)
    if request.method == "POST":
        idsa = request.form['ids']
        if session['nma'] in ('admin','Admin'):
            cursor = dbs.cursor()
            cursor.execute("DELETE FROM penya where rowid in ('%s') and username!='Admin'" % (idsa))
            flash("Foydalanuvchi ro'yxatdan o'chirildi")
            dbs.commit()
            msgg = "Mazkur ID raqamli foydalanuvch muvaffaqqiyatli o'chirildi!"
            return render_template("penya.html", msg=msgg, records=records, page=page, total_pages=total_pages)
    else:
        msgg = "Admin, hush kelibsiz!"
        return render_template("penya.html", msg=msgg, records=records, page=page, total_pages=total_pages)
    return render_template("penya.html", msg=msgg, records=records, page=page, total_pages=total_pages)

@app.route('/min_zp', methods=["POST", "GET"])
def min_zp():
    msgg = ''
    app.config["DEBUG"] = True
    app.secret_key = "789456asd"
    username = 'hbnnarzullayev'
    password = 'Sersarson7'
    host = 'hbnnarzullayev.mysql.pythonanywhere-services.com'
    mysql = 'hbnnarzullayev$flask3'
    dbs = msd.connect(host,username,password, mysql)
    # Get the current page number from the query string parameter
    page = int(request.args.get('page', 1))

    # Number of records to display per page
    per_page = 10

    # Calculate the start and end indices of the records to fetch from the database
    start_index = (page - 1) * per_page
    end_index = start_index + per_page
    # Connect to the MySQL database
    conn = MySQLdb.connect("hbnnarzullayev.mysql.pythonanywhere-services.com","hbnnarzullayev","Sersarson7","hbnnarzullayev$flask3" )
    cursor = conn.cursor()
    cursor.execute("select * from min_zp LIMIT %s, %s", (start_index, per_page))
    records = cursor.fetchall()

# Count the total number of records in the table
    cursor.execute("SELECT COUNT(*) FROM min_zp")
    total_records = cursor.fetchone()[0]#data from database
    total_pages = total_records // per_page + (total_records % per_page > 0)
    #msgg = "Muvaffaqqiyatli ro'yxatdan o'tdingiz11!"
    #return render_template("table.html", value=data, msg=msgg)
    if request.method == "POST":
        idsa = request.form['ids']
        if session['nma'] in ('admin','Admin'):
            cursor = dbs.cursor()
            cursor.execute("DELETE FROM min_zp where rowid in ('%s') and username!='Admin'" % (idsa))
            flash("Foydalanuvchi ro'yxatdan o'chirildi")
            dbs.commit()
            msgg = "Mazkur ID raqamli foydalanuvch muvaffaqqiyatli o'chirildi!"
            return render_template("min_zp.html", msg=msgg, records=records, page=page, total_pages=total_pages)
    else:
        msgg = "Admin, hush kelibsiz!"
        return render_template("min_zp.html", msg=msgg, records=records, page=page, total_pages=total_pages)
    return render_template("min_zp.html", msg=msgg, records=records, page=page, total_pages=total_pages)

@app.route('/na2', methods=["POST", "GET"])
def na2():
    msgg = ''
    app.config["DEBUG"] = True
    app.secret_key = "super secret key"
    username = 'hbnnarzullayev'
    password = 'Sersarson7'
    host = 'hbnnarzullayev.mysql.pythonanywhere-services.com'
    mysql = 'hbnnarzullayev$flask3'
    dbs = msd.connect(host,username,password, mysql)
    # Get the current page number from the query string parameter
    page = int(request.args.get('page', 1))

    # Number of records to display per page
    per_page = 10

    # Calculate the start and end indices of the records to fetch from the database
    start_index = (page - 1) * per_page
    end_index = start_index + per_page
    # Connect to the MySQL database
    conn = MySQLdb.connect("hbnnarzullayev.mysql.pythonanywhere-services.com","hbnnarzullayev","Sersarson7","hbnnarzullayev$flask3" )
    cursor = conn.cursor()
    cursor.execute("select * from na2 LIMIT %s, %s", (start_index, per_page))
    records = cursor.fetchall()

# Count the total number of records in the table
    cursor.execute("SELECT COUNT(*) FROM na2")
    total_records = cursor.fetchone()[0]#data from database
    total_pages = total_records // per_page + (total_records % per_page > 0)
    #msgg = "Muvaffaqqiyatli ro'yxatdan o'tdingiz11!"
    #return render_template("table.html", value=data, msg=msgg)
    if request.method == "POST":
        idsa = request.form['ids']
        if session['nma'] in ('admin','Admin'):
            cursor = dbs.cursor()
            cursor.execute("DELETE FROM na2 where rowid in ('%s') and username!='Admin'" % (idsa))
            flash("Foydalanuvchi ro'yxatdan o'chirildi")
            dbs.commit()
            msgg = "Mazkur ID raqamli foydalanuvch muvaffaqqiyatli o'chirildi!"
            return render_template("na2.html", msg=msgg, records=records, page=page, total_pages=total_pages)
    else:
        msgg = "Admin, hush kelibsiz!"
        return render_template("na2.html", msg=msgg, records=records, page=page, total_pages=total_pages)
    return render_template("na2.html", msg=msgg, records=records, page=page, total_pages=total_pages)

@app.route('/water', methods=["POST", "GET"])
def water():
    msgg = ''
    app.config["DEBUG"] = True
    app.secret_key = "super secret key"
    username = 'hbnnarzullayev'
    password = 'Sersarson7'
    host = 'hbnnarzullayev.mysql.pythonanywhere-services.com'
    mysql = 'hbnnarzullayev$flask3'
    dbs = msd.connect(host,username,password, mysql)
    # Get the current page number from the query string parameter
    page = int(request.args.get('page', 1))

    # Number of records to display per page
    per_page = 10

    # Calculate the start and end indices of the records to fetch from the database
    start_index = (page - 1) * per_page
    end_index = start_index + per_page
    # Connect to the MySQL database
    conn = MySQLdb.connect("hbnnarzullayev.mysql.pythonanywhere-services.com","hbnnarzullayev","Sersarson7","hbnnarzullayev$flask3" )
    cursor = conn.cursor()
    cursor.execute("select * from water LIMIT %s, %s", (start_index, per_page))
    records = cursor.fetchall()

# Count the total number of records in the table
    cursor.execute("SELECT COUNT(*) FROM water")
    total_records = cursor.fetchone()[0]#data from database
    total_pages = total_records // per_page + (total_records % per_page > 0)
    #msgg = "Muvaffaqqiyatli ro'yxatdan o'tdingiz11!"
    #return render_template("table.html", value=data, msg=msgg)
    if request.method == "POST":
        idsa = request.form['ids']
        if session['nma'] in ('admin','Admin'):
            cursor = dbs.cursor()
            cursor.execute("DELETE FROM na2 where rowid in ('%s') and username!='Admin'" % (idsa))
            flash("Foydalanuvchi ro'yxatdan o'chirildi")
            dbs.commit()
            msgg = "Mazkur ID raqamli foydalanuvch muvaffaqqiyatli o'chirildi!"
            return render_template("water.html", msg=msgg, records=records, page=page, total_pages=total_pages)
    else:
        msgg = "Admin, hush kelibsiz!"
        return render_template("water.html", msg=msgg, records=records, page=page, total_pages=total_pages)
    return render_template("water.html", msg=msgg, records=records, page=page, total_pages=total_pages)

@app.route('/get_subcategories', methods=['GET'])
def get_subcategories():
    selected_category = request.args.get('tin')
    conn = MySQLdb.connect("hbnnarzullayev.mysql.pythonanywhere-services.com","hbnnarzullayev","Sersarson7","hbnnarzullayev$flask3" )
    cur = conn.cursor()
    cur.execute('SELECT ns10_code FROM nla WHERE tin = %s group by ns10_code', (selected_category,))
    subcategories = [row[0] for row in cur.fetchall()]
    cur.execute("select max(last_date_srok) from loop_progress where tin=%s", (selected_category, ))
    subcategories = [row[0] for row in cur.fetchall()]
    cur.close()
    return jsonify(subcategories)

@app.route('/get_subcategories1', methods=['GET'])
def get_subcategories1():
    conn = MySQLdb.connect("hbnnarzullayev.mysql.pythonanywhere-services.com","hbnnarzullayev","Sersarson7","hbnnarzullayev$flask3" )
    cur = conn.cursor()
    cur.execute("select distinct(date_srok) as date_srok from nl where (nachislen_n+umenshen_n)<>0 and na2_code not in ('46','101','199','191') and date_real<date_srok order by date_srok asc;")
    subcategories = [row[0] for row in cur.fetchall()]
    #cur.execute("select max(last_date_srok) from loop_progress where tin=%s", (selected_category, ))
    #subcategories = [row[0] for row in cur.fetchall()]
    cur.close()
    return jsonify(subcategories)

@app.route('/nl', methods=["POST", "GET"])
def nl():
    msgg = ''
    app.config["DEBUG"] = True
    app.secret_key = "super secret key"
    username = 'hbnnarzullayev'
    password = 'Sersarson7'
    host = 'hbnnarzullayev.mysql.pythonanywhere-services.com'
    mysql = 'hbnnarzullayev$flask3'
    dbs = msd.connect(host,username,password, mysql)
    # Get the current page number from the query string parameter
    page = int(request.args.get('page', 1))

    # Number of records to display per page
    per_page = 50

    # Calculate the start and end indices of the records to fetch from the database
    start_index = (page - 1) * per_page
    end_index = start_index + per_page
    max_columns_to_display=12
    # Connect to the MySQL database
    conn = MySQLdb.connect("hbnnarzullayev.mysql.pythonanywhere-services.com","hbnnarzullayev","Sersarson7","hbnnarzullayev$flask3" )
    cursor = conn.cursor()
    cursor.execute("select * from nla order by date_srok desc, na2_code asc LIMIT %s, %s", (start_index, per_page))
    records = cursor.fetchall()
    column_names = [desc[0] for desc in cursor.description]

# Count the total number of records in the table
    cursor.execute("SELECT COUNT(*) FROM nla")
    total_records = cursor.fetchone()[0]#data from database
    total_pages = total_records // per_page + (total_records % per_page > 0)
    cursor = dbs.cursor()
    cursor.execute("select * from nla order by date_srok desc, na2_code asc LIMIT %s, %s", (start_index, per_page))
    records = cursor.fetchall()
    column_names = [desc[0] for desc in cursor.description]

# Count the total number of records in the table
    cursor.execute("SELECT COUNT(*) FROM nla")
    total_records = cursor.fetchone()[0]#data from database
    total_pages = total_records // per_page + (total_records % per_page > 0)
    cursor = dbs.cursor()
    cursor.execute("select count(id) from loop_progress")
    cout = cursor.fetchall()[0]
    cursor.execute("select max(last_date_srok) from loop_progress")
    last_saldo_date1 = cursor.fetchall()
    cursor.execute("select max(date_srok) from saldo")
    last_saldo_date2 = cursor.fetchall()
    if cout !=0:
        last_saldo_date =last_saldo_date1[0]
    else:
        last_saldo_date='2007'
    cursor.execute("select ynl as sum from nla where (nachislen_n+umenshen_n+uploch_n+vozvrat)>0 order by date_srok asc limit 1;")
    min_ynl1 = cursor.fetchone()
    if min_ynl1 is not None:
        min_ynl = [min_ynl1[0]]
    else:
        min_ynl='2007'
    cursor.execute("select distinct(date_srok) as date_srok from nl where (nachislen_n+umenshen_n)<>0 and na2_code not in ('46','101','199','191') and date_real<date_srok order by date_srok asc;")
    pr_min_ynl = [row[0] for row in cursor.fetchall()]
    cursor.execute("select na2_code from nla group by na2_code order by na2_code asc;")
    na2_codes = cursor.fetchall()
    cursor.execute("select tin from nla group by tin;")
    tins = cursor.fetchall()
    cursor.execute("select tin from nl group by tin;")
    nl_tins = cursor.fetchall()
    if request.method == "POST" and (session['nma'] in ('admin','Admin')) and (session['pwda'] in ('123')):
        if 'nla' in request.form:
            cursor.execute("select tin from nl group by tin")
            tin1 = cursor.fetchone()[0]
            cursor.execute("delete from nla where tin=%s;", (tin1,))
            cursor.execute("select min(date_srok) from nl where tin=%s", (tin1, ))
            min_date_nl = cursor.fetchone()[0]
            cursor.execute("SELECT STR_TO_DATE('01, 01, 2020', '%d, %m, %Y')")
            simple_year = cursor.fetchone()[0]
            cursor.execute("INSERT INTO nla (YNL,	NS10_CODE,	NS11_CODE,	tin, NA2_CODE,	DATE_SROK,	NACHISLEN_N,	UMENSHEN_N, UPLOCH_N, VOZVRAT, SPISANO_P, other_penya, shtraf, OWN_NS10_CODE,	OWN_NS11_CODE) select YNL,	NS10_CODE,	NS11_CODE,	tin, NA2_CODE,	DATE_SROK,	sum(NACHISLEN_N),	sum(UMENSHEN_N),	sum(ULPOCH_N), sum(VOZVRAT), sum(SPISANO_P), sum(ifnull(nachislen_p-umenshen_p,0)), sum(ifnull(shtraf,0)),	OWN_NS10_CODE,	OWN_NS11_CODE from nl where xar not in ('19','20', '1','99') and na2_code not in ('46','101','199','191') group by YNL, NS10_CODE,	NS11_CODE,	tin, NA2_CODE,	DATE_SROK,	OWN_NS10_CODE,	OWN_NS11_CODE order by date_srok desc, na2_code asc;")
            dbs.commit()
            cursor.execute("INSERT INTO nla (YNL,	NS10_CODE,	NS11_CODE,	tin, NA2_CODE,	DATE_SROK,	NACHISLEN_N,	UMENSHEN_N, UPLOCH_N, VOZVRAT, SPISANO_P, other_penya, shtraf, OWN_NS10_CODE,	OWN_NS11_CODE) select YNL,	NS10_CODE,	NS11_CODE,	tin, NA2_CODE,	DATE_SROK,	sum(saldo_tek_n),	sum(pereplata),	0, 0, 0, 0, 0,	OWN_NS10_CODE,	OWN_NS11_CODE from nl where xar in ('1') and na2_code not in ('46','101','199','191') and date_srok<=%s group by YNL, NS10_CODE,	NS11_CODE,	tin, NA2_CODE,	DATE_SROK,	OWN_NS10_CODE,	OWN_NS11_CODE order by date_srok desc, na2_code asc;", (min_date_nl, ))
            dbs.commit()
            cursor.execute("INSERT INTO nla (YNL,	NS10_CODE,	NS11_CODE,	tin, NA2_CODE,	DATE_SROK,	NACHISLEN_N,	UMENSHEN_N, UPLOCH_N, VOZVRAT, SPISANO_P, other_penya, shtraf, OWN_NS10_CODE,	OWN_NS11_CODE) select YNL,	NS10_CODE,	NS11_CODE,	tin, NA2_CODE,	DATE_SROK,	0,	0,	0, 0, 0, 0, 0,	OWN_NS10_CODE,	OWN_NS11_CODE from nl where xar in ('1','99') and na2_code not in ('46','101','199','191') and date_srok>=%s group by YNL, NS10_CODE,	NS11_CODE,	tin, NA2_CODE,	DATE_SROK,	OWN_NS10_CODE,	OWN_NS11_CODE order by date_srok desc, na2_code asc;", (min_date_nl, ))
            dbs.commit()
            # cursor.execute("INSERT INTO nla (YNL,	NS10_CODE,	NS11_CODE,	tin, NA2_CODE,	DATE_SROK,	NACHISLEN_N,	UMENSHEN_N, UPLOCH_N, VOZVRAT, SPISANO_P, other_penya, shtraf, OWN_NS10_CODE,	OWN_NS11_CODE) select YNL,	NS10_CODE,	NS11_CODE,	tin, NA2_CODE,	DATE_SROK,	sum(NACHISLEN_N),	sum(UMENSHEN_N),	sum(ULPOCH_N), sum(VOZVRAT), sum(SPISANO_P), sum(ifnull(nachislen_p-umenshen_p,0)), sum(ifnull(shtraf,0)),	OWN_NS10_CODE,	OWN_NS11_CODE from nl where xar in ('1') and na2_code not in ('46','101','199','191')  and date_srok='2020-01-01' group by YNL, NS10_CODE,	NS11_CODE,	tin, NA2_CODE,	DATE_SROK,	OWN_NS10_CODE,	OWN_NS11_CODE order by date_srok desc, na2_code asc;")
            # dbs.commit()
            cursor.execute("SELECT STR_TO_DATE('01, 01, 2020', '%d, %m, %Y')")
            simple_year = cursor.fetchone()[0]
            cursor.execute("select date_srok from nla where tin=%s and date_srok<%s group by date_srok", (tin1, simple_year, ))
            dates1 = [row[0] for row in cursor.fetchall()]
            for date1 in dates1:
                cursor.execute("SELECT DISTINCT na2_code FROM nla where tin=%s GROUP BY na2_code ORDER BY na2_code", (tin1, ))
                na2_codes = [row[0] for row in cursor.fetchall()]
                cursor.execute("select ynl from nla where tin=%s and date_srok=%s group by ynl", (tin1, date1,))
                ynl = cursor.fetchone()[0]
                cursor.execute("select ns10_code from nla where tin=%s and date_srok=%s group by ns10_code", (tin1, date1,))
                ns10_code = cursor.fetchone()[0]
                cursor.execute("select ns11_code from nla where tin=%s and date_srok=%s group by ns11_code", (tin1, date1,))
                ns11_code = cursor.fetchone()[0]
                cursor.execute("select own_ns10_code from nla where tin=%s and date_srok=%s group by own_ns10_code", (tin1, date1,))
                own_ns10_code = cursor.fetchone()[0]
                cursor.execute("select own_ns11_code from nla where tin=%s and date_srok=%s group by own_ns11_code", (tin1, date1,))
                own_ns11_code = cursor.fetchone()[0]
                for na2_code in na2_codes:
                    cursor.execute("SELECT na2_code FROM nla where tin=%s  and date_srok=%s and na2_code =%s GROUP BY na2_code ORDER BY na2_code", (tin1, date1, na2_code, ))
                    na2ss = cursor.fetchone()
                    # cursor.execute("SELECT DISTINCT na2_code FROM nla WHERE tin = %s AND na2_code NOT IN ({}) GROUP BY na2_code ORDER BY na2_code".format(placeholders), (1,) + tuple(na2_code))
                    if na2ss is not None:
                        cursor.execute("select ynl from nla where tin=%s and date_srok=%s group by ynl", (tin1, date1,))
                    else:
                        now = datetime.now(tz=timezone(timedelta(hours=5)))
                        cursor.execute("delete from loop_progress where last_date_srok!=%s and na2_code!=%s and tin=%s", (date1, na2_code, tin1,))
                        cursor.execute("INSERT INTO loop_progress (tin, last_date_srok, na2_code, datetime) VALUES (%s, %s, %s, %s)", (tin1, date1, na2_code,now.strftime('%Y-%m-%d %H:%M:%S')))
                        cursor.execute("insert into nla (ynl,ns10_code, ns11_code, tin, date_srok, na2_code, own_ns10_code, own_ns11_code) values(%s, %s, %s, %s, %s, %s, %s, %s)", (ynl, ns10_code, ns11_code, tin1, date1, na2_code, own_ns10_code, own_ns11_code,))
                    dbs.commit()
                dbs.commit()
            dbs.close()

            msgg = "NLAga yuklandi!"
            return render_template("nl.html", msg=msgg, column_names=column_names, max_columns_to_display=max_columns_to_display, records=records, page=page, total_pages=total_pages, last_saldo_date=last_saldo_date, last_saldo_date1=last_saldo_date1, min_ynl=min_ynl, pr_min_ynl=pr_min_ynl, na2_codes=na2_codes, tins=tins, nl_tins=nl_tins)
        elif 'nla_copy' in request.form:
            cursor.execute("select tin from nl_copy group by tin")
            tin1 = cursor.fetchone()[0]
            cursor.execute("delete from nla_copy where tin=%s;", (tin1,))
            cursor.execute("INSERT INTO nla_copy (YNL,	NS10_CODE,	NS11_CODE,	tin, NA2_CODE,	DATE_SROK,	NACHISLEN_N,	UMENSHEN_N, UPLOCH_N, VOZVRAT, SPISANO_P, other_penya, shtraf, OWN_NS10_CODE,	OWN_NS11_CODE) select YNL,	NS10_CODE,	NS11_CODE,	tin, NA2_CODE,	DATE_SROK,	sum(NACHISLEN_N),	sum(UMENSHEN_N),	sum(ULPOCH_N), sum(VOZVRAT), sum(SPISANO_P), sum(ifnull(nachislen_p-umenshen_p,0)), sum(ifnull(shtraf,0)),	OWN_NS10_CODE,	OWN_NS11_CODE from nl_copy where xar not in ('19','20') and na2_code not in ('46','101','199','191') group by YNL, NS10_CODE,	NS11_CODE,	tin, NA2_CODE,	DATE_SROK,	OWN_NS10_CODE,	OWN_NS11_CODE order by date_srok desc, na2_code asc;")
            dbs.commit()
            cursor.execute("SELECT STR_TO_DATE('01, 01, 2020', '%d, %m, %Y')")
            simple_year = cursor.fetchone()[0]
            cursor.execute("select date_srok from nla_copy where tin=%s and date_srok<%s group by date_srok", (tin1, simple_year, ))
            dates1 = [row[0] for row in cursor.fetchall()]
            for date1 in dates1:
                cursor.execute("SELECT DISTINCT na2_code FROM nla_copy where tin=%s GROUP BY na2_code ORDER BY na2_code", (tin1, ))
                na2_codes = [row[0] for row in cursor.fetchall()]
                cursor.execute("select ynl from nla_copy where tin=%s and date_srok=%s group by ynl", (tin1, date1,))
                ynl = cursor.fetchone()[0]
                cursor.execute("select ns10_code from nla_copy where tin=%s and date_srok=%s group by ns10_code", (tin1, date1,))
                ns10_code = cursor.fetchone()[0]
                cursor.execute("select ns11_code from nla_copy where tin=%s and date_srok=%s group by ns11_code", (tin1, date1,))
                ns11_code = cursor.fetchone()[0]
                cursor.execute("select own_ns10_code from nla_copy where tin=%s and date_srok=%s group by own_ns10_code", (tin1, date1,))
                own_ns10_code = cursor.fetchone()[0]
                cursor.execute("select own_ns11_code from nla_copy where tin=%s and date_srok=%s group by own_ns11_code", (tin1, date1,))
                own_ns11_code = cursor.fetchone()[0]
                for na2_code in na2_codes:
                    cursor.execute("SELECT na2_code FROM nla_copy where tin=%s  and date_srok=%s and na2_code =%s GROUP BY na2_code ORDER BY na2_code", (tin1, date1, na2_code, ))
                    na2ss = cursor.fetchone()
                    # cursor.execute("SELECT DISTINCT na2_code FROM nla_copy WHERE tin = %s AND na2_code NOT IN ({}) GROUP BY na2_code ORDER BY na2_code".format(placeholders), (1,) + tuple(na2_code))
                    if na2ss is not None:
                        cursor.execute("select ynl from nla_copy where tin=%s and date_srok=%s group by ynl", (tin1, date1,))
                    else:
                        now = datetime.now(tz=timezone(timedelta(hours=5)))
                        cursor.execute("delete from loop_progress where last_date_srok!=%s and na2_code!=%s and tin=%s", (date1, na2_code, tin1,))
                        cursor.execute("INSERT INTO loop_progress (tin, last_date_srok, na2_code, datetime) VALUES (%s, %s, %s, %s)", (tin1, date1, na2_code,now.strftime('%Y-%m-%d %H:%M:%S')))
                        cursor.execute("insert into nla_copy (ynl,ns10_code, ns11_code, tin, date_srok, na2_code, own_ns10_code, own_ns11_code) values(%s, %s, %s, %s, %s, %s, %s, %s)", (ynl, ns10_code, ns11_code, tin1, date1, na2_code, own_ns10_code, own_ns11_code,))
                    dbs.commit()
                dbs.commit()
            dbs.close()

            msgg = "NLAga yuklandi!"
            return render_template("nl.html", msg=msgg, column_names=column_names, max_columns_to_display=max_columns_to_display, last_saldo_date2=last_saldo_date2, pr_min_ynl=pr_min_ynl, records=records, page=page, total_pages=total_pages, last_saldo_date=last_saldo_date, last_saldo_date1=last_saldo_date1, min_ynl=min_ynl, na2_codes=na2_codes, tins=tins, nl_tins=nl_tins)

        # # if session['nma'] in ('admin','Admin'):
        #     with open("WEB\SQL\Penya.sql", 'r') as file:
        #         sql_script = file.read()
        #         # Execute the SQL script
        #     sql_statements = sql_script.split(';')
        #     dbs.commit()
        #     # Execute each SQL statement
        #     for statement in sql_statements:
        #         if statement.strip():
        #             cursor.execute(statement)
        #             cursor.fetchone()
        #     dbs.commit()
        #     result = cursor.fetchall()
            # cursor.execute("""
            #     WITH RECURSIVE cte AS (SELECT date_srok, na2_code, nachislen_n, umenshen_n, last_date_srok AS max_date, nachislen_n-umenshen_n AS saldo_all
            #       FROM nla
            #       WHERE date_srok = (SELECT MIN(date_srok) FROM nla order by date_srok)

            #       UNION ALL

            #       SELECT t.date_srok, t.na2_code, t.nachislen_n, t.umenshen_n, GREATEST(t.date_srok, cte.max_date), t.nachislen_n + t.umenshen_n + cte.saldo_all AS saldo_all
            #       FROM nla t
            #       JOIN cte ON t.date_srok = cte.date_srok + INTERVAL 1 DAY and t.na2_code=cte.na2_code
            #     )
            #     SELECT date_srok, saldo_all FROM cte;
            # """)

            # results = cursor.fetchall()

            # for row in results:
            #     date, d_value = row
            #     # Update the "D" column in the database for each row
            #     cursor.execute("UPDATE nla SET saldo_all = %s WHERE date_srok = %s", (d_value, date))
        elif 'last_date_srok' in request.form:
            cursor.execute("select tin from nl group by tin")
            tin1 = cursor.fetchone()[0]
            cursor.execute("SELECT STR_TO_DATE('01, 01, 2020', '%d, %m, %Y')")
            simple_year = cursor.fetchone()[0]
            cursor.execute("select date_srok from nla where tin=%s group by date_srok", (tin1, ))
            dates = [row[0] for row in cursor.fetchall()]
            for date in dates:
                cursor.execute("select min(date_srok) from nla where tin=%s", (tin1, ))
                min_date = cursor.fetchone()[0]
                cursor.execute("select max(date_srok) from nla where tin=%s and date_srok<%s", (tin1, date, ))
                ldate1 = cursor.fetchone()
                if ldate1 is None:
                    ldate=date
                else:
                    ldate=ldate1[0]
                    if date<simple_year and date==min_date:
                        cursor.execute("update nla set last_date_srok=date_srok where tin=%s and date_srok=%s", (tin1, ldate, ))
                        dbs.commit()
                    elif date<=simple_year:
                        now = datetime.now(tz=timezone(timedelta(hours=5)))
                        cursor.execute("update nla set last_date_srok=%s where tin=%s and date_srok=%s", (ldate, tin1, date, ))
                        cursor.execute("delete from loop_progress where last_date_srok!=%s and na2_code!=%s and tin=%s", (date, 1000, tin1,))
                        cursor.execute("INSERT INTO loop_progress (tin, last_date_srok, na2_code, datetime) VALUES (%s, %s, %s, %s)", (tin1, date, 1000,now.strftime('%Y-%m-%d %H:%M:%S')))
                        dbs.commit()
                    else:
                        cursor.execute("select na2_code from nla where tin=%s and date_srok=%s group by na2_code", (tin1, date, ))
                        na2s = [row[0] for row in cursor.fetchall()]
                        for na2 in na2s:
                            now = datetime.now(tz=timezone(timedelta(hours=5)))
                            cursor.execute("delete from loop_progress where last_date_srok!=%s and na2_code!=%s and tin=%s", (date, na2, tin1,))
                            cursor.execute("INSERT INTO loop_progress (tin, last_date_srok, na2_code, datetime) VALUES (%s, %s, %s, %s)", (tin1, date, na2,now.strftime('%Y-%m-%d %H:%M:%S')))
                            cursor.execute("select max(date_srok) from nla where tin=%s and date_srok<%s and na2_code=%s", (tin1, date, na2, ))
                            max_date = cursor.fetchone()[0]
                            cursor.execute("update nla set last_date_srok=%s where tin=%s and date_srok=%s and na2_code=%s", (max_date, tin1, date, na2, ))
                        dbs.commit()
                    dbs.commit()
            cursor.execute("update nla set nachislen_n=if(nachislen_n is null, 0, nachislen_n), umenshen_n=if(umenshen_n is null, 0, umenshen_n), uploch_n=if(uploch_n is null, 0, uploch_n), vozvrat=if(vozvrat is null, 0, vozvrat), saldo_all=if(saldo_all is null,0, saldo_all), saldo_sum_p=if(saldo_sum_p is null, 0, saldo_sum_p) where tin=(select tin from nl limit 1);")
            cursor.execute("update nla set saldo_all=(0-nachislen_n+umenshen_n+uploch_n-vozvrat) where tin=(select tin from nl limit 1);")
            # -- update nla set saldo_all=((select t1.saldo_all from (SELECT * FROM nla) as t1 where t1.date_srok=nla.last_date_srok and t1.na2_code=nla.na2_code and t1.tin=(select tin from nl limit 1) order by nla.date_srok, nla.na2_code)-nla.nachislen_n+nla.umenshen_n-nla.vozvrat+nla.uploch_n), saldo_sum_p=if(((select Foiz from (select * from penya) as t2 where t2.boshlanish_vaqti<=(select t1.date_srok from (SELECT * FROM nla) as t1 where t1.date_srok=nla.last_date_srok and t1.na2_code=nla.na2_code) order by boshlanish_vaqti desc limit 1)*(nla.date_srok-(select t1.date_srok from (SELECT * FROM nla) as t1 where t1.date_srok=nla.last_date_srok and t1.na2_code=nla.na2_code and t1.tin=(select tin from nl limit 1)))*((select t1.saldo_all from (SELECT * FROM nla) as t1 where t1.date_srok=nla.last_date_srok and t1.na2_code=nla.na2_code and t1.tin=(select tin from nl limit 1))-nla.nachislen_n+nla.umenshen_n-nla.vozvrat+nla.uploch_n)/30000)*0.6<0,(-1*(((select Foiz from (select * from penya) as t2 where t2.boshlanish_vaqti<=(select t1.date_srok from (SELECT * FROM nla) as t1 where t1.date_srok=nla.last_date_srok and t1.na2_code=nla.na2_code and t1.tin=(select tin from nl limit 1)) order by boshlanish_vaqti desc limit 1)*(nla.date_srok-(select t1.date_srok from (SELECT * FROM nla) as t1 where t1.date_srok=nla.last_date_srok and t1.na2_code=nla.na2_code and t1.tin=(select tin from nl limit 1)))*((select t1.saldo_all from (SELECT * FROM nla) as t1 where t1.date_srok=nla.last_date_srok and t1.na2_code=nla.na2_code and t1.tin=(select tin from nl limit 1))-nla.nachislen_n+nla.umenshen_n-nla.vozvrat+nla.uploch_n)/30000)*0.6)),0);
            cursor.execute("update nla set last_date_srok = date_srok where last_date_srok is null and tin=(select tin from nl limit 1);")
            cursor.execute("update nla set saldo_all=(0-nachislen_n+umenshen_n+uploch_n-vozvrat), uploch_p = 0,  pr_sum =0, pr_saldo_all=0,pr_saldo=0, pr_penya_all=0, pr_penya=0 where tin=(select tin from nl limit 1);")
            cursor.execute("delete from pereraschet where tin =%s", (tin1,))
            cursor.execute("insert into pereraschet (ns10_code, ns11_code, tin, na2_code, date_srok, nachislen_n, umenshen_n, date_real, own_ns10_code, own_ns11_code) select ns10_code, ns11_code, tin, na2_code, date_srok, nachislen_n, umenshen_n, date_real, own_ns10_code, own_ns11_code from nl where tin = %s and text>text1", (tin1,))
            # cursor.execute("insert into start (tin, date, date_srok, datetime) values(%s, %s, %s, %s)",(tin, start_value,pr_date1, now.strftime('%Y-%m-%d %H:%M:%S')))
            dbs.commit()
            dbs.close()

            msgg = "NLAga yuklandi!"
            return render_template("nl.html", msg=msgg, column_names=column_names, max_columns_to_display=max_columns_to_display, pr_min_ynl=pr_min_ynl, last_saldo_date2=last_saldo_date2, records=records, page=page, total_pages=total_pages, last_saldo_date=last_saldo_date, last_saldo_date1=last_saldo_date1, min_ynl=min_ynl, na2_codes=na2_codes, tins=tins, nl_tins=nl_tins)
        elif 'selected_year' in request.form:
            selected_tin=request.form.get('tin2')
            selected_year=request.form.get('selected_year')
            selected_na2=request.form["selected_na2"]
            # Connect to the MySQL database
            conn = MySQLdb.connect("hbnnarzullayev.mysql.pythonanywhere-services.com","hbnnarzullayev","Sersarson7","hbnnarzullayev$flask3" )
            cursor = conn.cursor()
            cursor.execute("select last_date_srok from loop_progress LIMIT 1")
            last_saldo_date = cursor.fetchall()[0]
            if selected_na2 != "":
                cursor.execute("select * from nla where tin= %s and ynl=%s and na2_code=%s order by date_srok asc, na2_code asc", (selected_tin, selected_year, selected_na2, ))
                records = cursor.fetchall()
            else:
                cursor.execute("select * from nla where tin= %s and ynl=%s order by date_srok asc, na2_code asc", (selected_tin, selected_year, ))
                records = cursor.fetchall()

        # Count the total number of records in the table
            cursor.execute("SELECT COUNT(*) FROM nla")
            total_records = cursor.fetchone()[0]#data from database
            total_pages = total_records // per_page + (total_records % per_page > 0)
            cursor = dbs.cursor()
            cursor.execute("select max(last_date_srok) from loop_progress")
            last_saldo_date = cursor.fetchall()[0]
            cursor.execute("select ynl as sum from nla where (nachislen_n+umenshen_n+uploch_n+vozvrat)>0 order by date_srok asc limit 1;")
            min_ynl = cursor.fetchall()[0]
            cursor.execute("select na2_code from nla group by na2_code order by na2_code asc;")
            na2_codes = cursor.fetchall()
            # saldo_kun=request.form['date_saldo']
            last_saldo_date1="{0}-01-01".format(min_ynl)
            dbs.commit()
            dbs.close()
            return render_template("nl.html", msg=msgg, records=records, max_columns_to_display=max_columns_to_display, column_names=column_names, pr_min_ynl=pr_min_ynl, page=page, total_pages=total_pages, last_saldo_date=last_saldo_date, last_saldo_date1=last_saldo_date1, last_saldo_date2=last_saldo_date2, min_ynl=min_ynl, na2_codes=na2_codes, tins=tins, nl_tins=nl_tins)
    else:
        msgg = "Admin bo'lishingiz zarur!"
        return render_template("nl.html", msg=msgg, records=records, max_columns_to_display=max_columns_to_display, column_names=column_names, pr_min_ynl=pr_min_ynl, page=page, total_pages=total_pages, last_saldo_date=last_saldo_date, last_saldo_date1=last_saldo_date1, min_ynl=min_ynl, na2_codes=na2_codes, tins=tins, nl_tins=nl_tins)

@app.route('/saldolash', methods=["POST", "GET"])
def saldolash():
    msgg = ''
    # user_ip = request.remote_addr  # Get user's IP address
    # store_ip_in_database(user_ip)
    app.config["DEBUG"] = True
    app.secret_key = "789456asd"
    user = 'hbnnarzullayev'
    password = 'Sersarson7'
    host = 'hbnnarzullayev.mysql.pythonanywhere-services.com'
    mysql = 'hbnnarzullayev$flask3'
    dbs = msd.connect(host,user,password, mysql)
    cursor = dbs.cursor()
    tin = request.form.get('tin_saldo')
    cursor.execute("select distinct(date_srok) from nl where tin=%s", (tin,))
    dates = [row[0] for row in cursor.fetchall()]
    cursor.execute("select distinct(na2_code) from nl where tin=%s AND na2_code NOT IN ('46', '101', '199', '191')", (tin,))
    na2s = [row[0] for row in cursor.fetchall()]
    cursor.execute("delete from saldo where tin=%s", (tin,))
    cursor.execute("SELECT STR_TO_DATE('01, 01, 2020', '%d, %m, %Y')")
    simple_year = cursor.fetchone()[0]
    for date in dates:
        if date<simple_year:
            now = datetime.now(tz=timezone(timedelta(hours=5)))
            for na2 in na2s:
                cursor.execute("SELECT max(date_srok) as max_date FROM nl WHERE date_srok<=%s AND na2_code=%s and tin=%s", (date, na2, tin, ))
                max_date = cursor.fetchone()[0]
                cursor.execute("SELECT max(date_srok) as max_date FROM nl WHERE date_srok<%s AND na2_code=%s and tin=%s", (date, na2, tin, ))
                max_date1 = cursor.fetchone()[0]
                cursor.execute("SELECT max(ord) as max_ord FROM nl WHERE date_srok=%s AND na2_code=%s and tin=%s", (max_date, na2, tin, ))
                max_ord = cursor.fetchone()[0]
                cursor.execute("SELECT saldo_tek_n FROM nl WHERE date_srok=%s AND na2_code=%s and tin=%s and ord=%s", (max_date, na2, tin, max_ord, ))
                saldo_tek_n = cursor.fetchone()[0]
                cursor.execute("SELECT pereplata FROM nl WHERE date_srok=%s AND na2_code=%s and tin=%s and ord=%s", (max_date, na2, tin, max_ord, ))
                pereplata = cursor.fetchone()[0]
                cursor.execute("insert into saldo (date_srok, max_date_srok, tin, na2_code, saldo_tek_n, pereplata, date_sys, last_date_srok) values(%s, %s, %s, %s, %s, %s, %s, %s)",(date, max_date,tin, na2, saldo_tek_n, pereplata, now.strftime('%Y-%m-%d %H:%M:%S'), max_date1))
            dbs.commit()
        else:
            cursor.execute("select distinct(na2_code) from nl where tin=%s AND na2_code NOT IN ('46', '101', '199', '191') and date_srok=%s", (tin, date, ))
            na22 = [row[0] for row in cursor.fetchall()]
            now = datetime.now(tz=timezone(timedelta(hours=5)))
            for na2 in na22:
                cursor.execute("SELECT max(date_srok) as max_date FROM nl WHERE date_srok<%s AND na2_code=%s and tin=%s", (date, na2, tin, ))
                max_date = cursor.fetchone()[0]
                cursor.execute("INSERT INTO saldo (date_srok, tin, na2_code, saldo_tek_n, pereplata, date_sys, last_date_srok) SELECT f.date_srok, f.tin, f.na2_code, f.saldo_tek_n, f.pereplata, %s, %s FROM nl f WHERE f.date_srok = %s AND f.tin = %s and f.na2_code=%s AND ord = (SELECT MAX(t.ord) FROM nl t WHERE t.date_srok = f.date_srok AND t.tin = f.tin AND t.na2_code = f.na2_code) order by f.na2_code", (now.strftime('%Y-%m-%d %H:%M:%S'), max_date, date, tin, na2))
                # cursor.execute("insert into saldo (date_srok, tin, na2_code, saldo_tek_n, pereplata, date_sys, last_date_srok) values(SELECT f.date_srok, f.tin, f.na2_code, f.saldo_tek_n, f.pereplata FROM nl f WHERE f.date_srok = %s and f.tin=%s AND ord = (SELECT MAX(t.ord) FROM nl t WHERE t.date_srok = f.date_srok and t.tin=f.tin AND t.na2_code = f.na2_code), %s)",(date, tin, now.strftime('%Y-%m-%d %H:%M:%S'), max_date1))
            dbs.commit()

    return redirect(url_for("nl"))

@app.route('/pr_saldolash', methods=["POST", "GET"])
def pr_saldolash():
    msgg = ''
    # user_ip = request.remote_addr  # Get user's IP address
    # store_ip_in_database(user_ip)
    app.config["DEBUG"] = True
    app.secret_key = "789456asd"
    user = 'hbnnarzullayev'
    password = 'Sersarson7'
    host = 'hbnnarzullayev.mysql.pythonanywhere-services.com'
    mysql = 'hbnnarzullayev$flask3'
    dbs = msd.connect(host,user,password, mysql)
    cursor = dbs.cursor()
    tin = request.form.get('tin_saldo')
    radio_choice = request.form.get('radio_choice1')
    if radio_choice == 'option3':  # User chose to select from dropdown
        max_date_srok = request.form.get('selected_date')
        # min_date_srok = selected_date
    else:  # User chose to enter date manually
        max_date_srok = request.form.get('manual_date')
    cursor.execute("select distinct(date_srok) from saldo where tin=%s and date_srok<=%s", (tin,max_date_srok, ))
    dates = [row[0] for row in cursor.fetchall()]
    cursor.execute("select distinct(na2_code) from saldo where tin=%s", (tin,))
    na2s = [row[0] for row in cursor.fetchall()]
    cursor.execute("SELECT STR_TO_DATE('01, 01, 2020', '%d, %m, %Y')")
    simple_year = cursor.fetchone()[0]
    cursor.execute("SELECT date_srok FROM nl WHERE date_srok > date_real AND (nachislen_n + umenshen_n) <> 0 AND na2_code NOT IN ('46', '101', '199', '191') and tin=%s GROUP BY date_srok ORDER BY date_srok asc", (tin,))
    pr_dates = [row[0] for row in cursor.fetchall()]
    cursor.execute("UPDATE saldo SET penya1 = 0, penya2 = 0, peny_all = 0, pr_peny_all = 0, sum_pr_penya = 0, saldo_pr=0, saldo_all_codes_pr=0")
    for date in dates:
        cursor.execute("select (procent/koef) from ulush where data=%s", (date,))
        foiz = cursor.fetchone()[0]
        cursor.execute("select ulush as ulush from ulush where data=%s", (date,))
        ulush = cursor.fetchone()[0]
        if date < simple_year:
            if date in pr_dates:
                for na2 in na2s:
                    now = datetime.now(tz=timezone(timedelta(hours=5)))
                    cursor.execute("SELECT sum(pereplata-saldo_tek_n) FROM saldo WHERE date_srok=%s and tin=%s and na2_code=%s", (date, tin, na2, ))
                    saldo_all = cursor.fetchone()[0]
                    cursor.execute("update saldo set saldo_all=%s where date_srok=%s and tin=%s and na2_code=%s", (saldo_all, date, tin, na2, ))
                    dbs.commit()
                    cursor.execute("SELECT sum(saldo_all) FROM saldo WHERE date_srok=%s and tin=%s", (date, tin, ))
                    saldo_all_codes = cursor.fetchone()[0]
                    cursor.execute("update saldo set saldo_all_codes=%s where date_srok=%s and tin=%s", (saldo_all_codes, date, tin, ))
                    dbs.commit()
                dbs.commit()
                cursor.execute("SELECT min(date_real) FROM nl WHERE date_srok = %s AND tin = %s", (date, tin,))
                date_real = cursor.fetchone()[0]

                while date_real is not None:
                    cursor.execute("SELECT min(date_real) FROM nl WHERE date_srok > %s AND (nachislen_n + umenshen_n) <> 0 AND na2_code NOT IN ('46', '101', '199', '191') AND date_real < %s AND tin = %s", (date_real, date_real, tin,))
                    new_date_real = cursor.fetchone()[0]

                    if new_date_real is None:
                        break

                    date_real = new_date_real

                latest_date_real = date_real
                cursor.execute("UPDATE saldo SET last_date_real = %s WHERE date_srok = %s AND tin = %s", (latest_date_real, date, tin,))
                dbs.commit()

                cursor.execute("SELECT date_srok FROM saldo WHERE date_srok >= %s and date_srok<=%s and tin=%s GROUP BY date_srok ORDER BY date_srok asc", (latest_date_real, date, tin,))
                p_dates = [row[0] for row in cursor.fetchall()]
                for p_date in p_dates:
                    cursor.execute("SELECT max(date_srok) FROM saldo WHERE date_srok >= %s and date_srok<=%s and tin=%s", (latest_date_real, date, tin,))
                    max_p_date = cursor.fetchone()[0]
                    if p_date!=max_p_date:
                        for na2 in na2s:
                            cursor.execute("SELECT sum(nachislen_n-umenshen_n) FROM nl WHERE date_srok>%s and date_srok<=%s and date_real<=%s and tin=%s and na2_code=%s", (p_date, date, p_date, tin, na2, ))
                            pr_sum = cursor.fetchone()[0]
                            cursor.execute("update saldo set pr_sum=ifnull(%s,0) where date_srok=%s and tin=%s and na2_code=%s", (pr_sum, p_date, tin, na2, ))
                            dbs.commit()
                            #saldo1 = (saldo_all if saldo_all is not None else 0) - (pr_sum if pr_sum is not None else 0)
                            cursor.execute("update saldo set saldo_pr=saldo_all-ifnull(%s,0) where date_srok=%s and tin=%s and na2_code=%s", (pr_sum, p_date, tin, na2, ))
                            dbs.commit()
                            cursor.execute("SELECT sum(nachislen_n-umenshen_n) FROM nl WHERE date_srok>%s and date_srok<=%s and date_real<=%s and tin=%s", (p_date, date, p_date, tin, ))
                            pr_sum_all = cursor.fetchone()[0]
                            cursor.execute("update saldo set pr_sum_all=ifnull(%s,0) where date_srok=%s and tin=%s and na2_code=%s", (pr_sum_all, p_date, tin, na2, ))
                            dbs.commit()
                            cursor.execute("update saldo set saldo_all_codes_pr=saldo_all_codes-ifnull(%s,0) where date_srok=%s and tin=%s", (pr_sum_all, p_date, tin, ))
                            dbs.commit()

                            last_date=cursor.execute("select max(last_date_srok) from saldo where date_srok=%s and tin=%s", (p_date, tin, ))
                            result = cursor.fetchone()
                            if result is not None:
                                min_last_date = result[0]
                            else:
                                min_last_date=p_date
                            cursor.execute("SELECT DATEDIFF(date_srok, %s) from saldo where date_srok=%s and tin=%s", (min_last_date, p_date, tin, ))
                            farq = cursor.fetchone()
                            cursor.execute("SELECT sum(saldo_pr) FROM saldo WHERE date_srok=%s and tin=%s", (min_last_date, tin, ))
                            last_saldo_all_codes = cursor.fetchone()[0]
                            cursor.execute("SELECT sum(saldo_pr) FROM saldo WHERE date_srok=%s and tin=%s and saldo_pr<0", (min_last_date, tin, ))
                            last_saldo_all_codes_ned = cursor.fetchone()[0]
                            cursor.execute("SELECT saldo_pr FROM saldo WHERE date_srok=%s and tin=%s and na2_code=%s", (min_last_date, tin, na2, ))
                            last_saldo_all = cursor.fetchone()[0]
                            cursor.execute("update saldo set pr_peny_all=(-1)*(%s)*(%s)*(%s)*(%s) where date_srok=%s and %s<0 and tin=%s", (last_saldo_all_codes, farq, foiz, ulush, p_date, last_saldo_all_codes, tin, ))
                            dbs.commit()
                            cursor.execute("update saldo set penya2=pr_peny_all*((%s)/(%s)) where %s<0 and %s<0 and date_srok=%s and tin=%s and na2_code=%s", (last_saldo_all, last_saldo_all_codes_ned, last_saldo_all, last_saldo_all_codes, p_date, tin, na2, ))
                            dbs.commit()
                        dbs.commit()
                    else:
                        for na2 in na2s:
                            cursor.execute("SELECT sum(nachislen_n-umenshen_n) FROM nl WHERE date_srok>%s and date_srok<=%s and date_real<=%s and tin=%s and na2_code=%s", (p_date, date, p_date, tin, na2, ))
                            pr_sum = cursor.fetchone()[0]
                            cursor.execute("update saldo set pr_sum=ifnull(%s,0) where date_srok=%s and tin=%s and na2_code=%s", (pr_sum, p_date, tin, na2, ))
                            dbs.commit()
                            #saldo1 = (saldo_all if saldo_all is not None else 0) - (pr_sum if pr_sum is not None else 0)
                            cursor.execute("update saldo set saldo_pr=saldo_all-ifnull(%s,0) where date_srok=%s and tin=%s and na2_code=%s", (pr_sum, p_date, tin, na2, ))
                            dbs.commit()
                            cursor.execute("SELECT sum(nachislen_n-umenshen_n) FROM nl WHERE date_srok>%s and date_srok<=%s and date_real<=%s and tin=%s", (p_date, date, p_date, tin, ))
                            pr_sum_all = cursor.fetchone()[0]
                            cursor.execute("update saldo set pr_sum_all=ifnull(%s,0) where date_srok=%s and tin=%s and na2_code=%s", (pr_sum_all, p_date, tin, na2, ))
                            dbs.commit()
                            cursor.execute("update saldo set saldo_all_codes_pr=saldo_all_codes-ifnull(%s,0) where date_srok=%s and tin=%s", (pr_sum_all, p_date, tin, ))
                            dbs.commit()

                            last_date=cursor.execute("select max(last_date_srok) from saldo where date_srok=%s and tin=%s", (p_date, tin, ))
                            result = cursor.fetchone()
                            if result is not None:
                                min_last_date = result[0]
                            else:
                                min_last_date=p_date
                            cursor.execute("SELECT DATEDIFF(date_srok, %s) from saldo where date_srok=%s and tin=%s", (min_last_date, p_date, tin, ))
                            farq = cursor.fetchone()
                            cursor.execute("SELECT sum(saldo_pr) FROM saldo WHERE date_srok=%s and tin=%s", (min_last_date, tin, ))
                            last_saldo_all_codes = cursor.fetchone()[0]
                            cursor.execute("SELECT saldo_pr FROM saldo WHERE date_srok=%s and tin=%s and na2_code=%s", (min_last_date, tin, na2, ))
                            last_saldo_all = cursor.fetchone()[0]
                            cursor.execute("SELECT sum(saldo_pr) FROM saldo WHERE date_srok=%s and tin=%s and saldo_pr<0", (min_last_date, tin, ))
                            last_saldo_all_codes_ned = cursor.fetchone()[0]
                            cursor.execute("update saldo set pr_peny_all=(-1)*(%s)*(%s)*(%s)*(%s) where date_srok=%s and %s<0 and tin=%s", (last_saldo_all_codes, farq, foiz, ulush, p_date, last_saldo_all_codes, tin, ))
                            dbs.commit()
                            cursor.execute("SELECT ifnull(saldo_all,0) FROM saldo WHERE date_srok=%s and tin=%s and na2_code=%s", (min_last_date, tin, na2, ))
                            last_saldo_all1 = cursor.fetchone()
                            if last_saldo_all1 is not None:
                                last_saldo_all = last_saldo_all1[0]
                            else:
                                last_saldo_all=0
                            cursor.execute("update saldo set penya2=pr_peny_all*(%s/(%s)) where %s<0 and %s<0 and date_srok=%s and tin=%s and na2_code=%s", (last_saldo_all, last_saldo_all_codes_ned, last_saldo_all, last_saldo_all_codes, date, tin, na2, ))
                            dbs.commit()
                            #cursor.execute("update saldo set penya2=peny_all*((%s)/(%s)) where %s<0 and %s<0 and date_srok=%s and tin=%s and na2_code=%s", (last_saldo_all, last_saldo_all_codes_ned, last_saldo_all, last_saldo_all_codes, p_date, tin, na2, ))
                            #dbs.commit()
                            cursor.execute("SELECT sum(penya1) FROM saldo WHERE date_srok>=%s and date_srok<=%s and tin=%s and na2_code=%s", (latest_date_real, p_date, tin, na2, ))
                            sum_penya1 = cursor.fetchone()[0]
                            cursor.execute("SELECT sum(penya2) FROM saldo WHERE date_srok>=%s and date_srok<=%s and tin=%s and na2_code=%s", (latest_date_real, p_date, tin, na2, ))
                            sum_penya2 = cursor.fetchone()[0]
                            cursor.execute("SELECT sum(sum_pr_penya) FROM saldo WHERE date_srok>=%s and date_srok<%s and tin=%s and na2_code=%s", (latest_date_real, p_date, tin, na2, ))
                            sum_pr_penya = cursor.fetchone()[0]
                            cursor.execute("update saldo set sum_pr_penya=%s-%s-%s where date_srok=%s and tin=%s and na2_code=%s", (sum_penya2, sum_penya1, sum_pr_penya, p_date, tin, na2, ))
                            dbs.commit()
                        dbs.commit()
                dbs.commit()
            else:
                for na2 in na2s:
                    last_date=cursor.execute("select max(last_date_srok) from saldo where date_srok=%s and tin=%s", (date, tin, ))
                    result = cursor.fetchone()
                    if result is not None:
                        min_last_date = result[0]
                    else:
                        min_last_date=date
                    cursor.execute("SELECT DATEDIFF(date_srok, %s) from saldo where date_srok=%s and tin=%s", (min_last_date, date, tin, ))
                    farq = cursor.fetchone()
                    now = datetime.now(tz=timezone(timedelta(hours=5)))
                    cursor.execute("SELECT sum(pereplata-saldo_tek_n) FROM saldo WHERE date_srok=%s and tin=%s and na2_code=%s", (date, tin, na2, ))
                    saldo_all = cursor.fetchone()[0]
                    cursor.execute("update saldo set saldo_all=%s where date_srok=%s and tin=%s and na2_code=%s", (saldo_all, date, tin, na2, ))
                    dbs.commit()
                    cursor.execute("SELECT sum(saldo_all) FROM saldo WHERE date_srok=%s and tin=%s", (date, tin, ))
                    saldo_all_codes = cursor.fetchone()[0]
                    cursor.execute("update saldo set saldo_all_codes=%s where date_srok=%s and tin=%s", (saldo_all_codes, date, tin, ))
                    dbs.commit()
                    cursor.execute("SELECT sum(saldo_all) FROM saldo WHERE date_srok=%s and tin=%s", (min_last_date, tin, ))
                    last_saldo_all_codes1 = cursor.fetchone()
                    if last_saldo_all_codes1 is not None:
                        last_saldo_all_codes = last_saldo_all_codes1[0]
                    else:
                        last_saldo_all_codes=0
                    cursor.execute("SELECT ifnull(saldo_all,0) FROM saldo WHERE date_srok=%s and tin=%s and na2_code=%s", (min_last_date, tin, na2, ))
                    last_saldo_all1 = cursor.fetchone()
                    if last_saldo_all1 is not None:
                        last_saldo_all = last_saldo_all1[0]
                    else:
                        last_saldo_all=0
                    cursor.execute("SELECT sum(saldo_all) FROM saldo WHERE date_srok=%s and tin=%s and saldo_all<0", (min_last_date, tin, ))
                    last_saldo_all_codes_ned = cursor.fetchone()[0]
                    cursor.execute("update saldo set peny_all=(-1)*(%s)*(%s)*(%s)*(%s) where date_srok=%s and %s<0 and tin=%s", (last_saldo_all_codes, farq, foiz, ulush, date, last_saldo_all_codes, tin, ))
                    dbs.commit()
                    cursor.execute("update saldo set penya1=peny_all*(%s/(%s)) where %s<0 and %s<0 and date_srok=%s and tin=%s and na2_code=%s", (last_saldo_all, last_saldo_all_codes_ned, last_saldo_all, last_saldo_all_codes, date, tin, na2, ))
                    dbs.commit()
                dbs.commit()

        #2020dan keyin
        else:
            #pr_bolsa
            if date in pr_dates:
                for na2 in na2s:
                    now = datetime.now(tz=timezone(timedelta(hours=5)))
                    cursor.execute("SELECT sum(pereplata-saldo_tek_n) FROM saldo WHERE date_srok=%s and tin=%s and na2_code=%s", (date, tin, na2, ))
                    saldo_all = cursor.fetchone()[0]
                    cursor.execute("update saldo set saldo_all=%s where date_srok=%s and tin=%s and na2_code=%s", (saldo_all, date, tin, na2, ))
                    dbs.commit()
                    cursor.execute("SELECT sum(saldo_all) FROM saldo WHERE date_srok=%s and tin=%s", (date, tin, ))
                    saldo_all_codes = cursor.fetchone()[0]
                    cursor.execute("update saldo set saldo_all_codes=%s where date_srok=%s and tin=%s", (saldo_all_codes, date, tin, ))
                    dbs.commit()
                dbs.commit()
                cursor.execute("SELECT min(date_real) FROM nl WHERE date_srok = %s AND tin = %s", (date, tin,))
                date_real = cursor.fetchone()[0]

                while date_real is not None:
                    cursor.execute("SELECT min(date_real) FROM nl WHERE date_srok > %s AND (nachislen_n + umenshen_n) <> 0 AND na2_code NOT IN ('46', '101', '199', '191') AND date_real < %s AND tin = %s", (date_real, date_real, tin,))
                    new_date_real = cursor.fetchone()[0]

                    if new_date_real is None:
                        break

                    date_real = new_date_real

                latest_date_real = date_real
                cursor.execute("UPDATE saldo SET last_date_real = %s WHERE date_srok = %s AND tin = %s", (latest_date_real, date, tin,))
                dbs.commit()

                cursor.execute("SELECT date_srok FROM saldo WHERE date_srok >= %s and date_srok<=%s and tin=%s GROUP BY date_srok ORDER BY date_srok asc", (latest_date_real, date, tin,))
                p_dates = [row[0] for row in cursor.fetchall()]
                for p_date in p_dates:
                    #pr bolsa
                    if p_date<simple_year:
                        cursor.execute("SELECT max(date_srok) FROM saldo WHERE date_srok >= %s and date_srok<=%s and tin=%s", (latest_date_real, date, tin,))
                        max_p_date = cursor.fetchone()[0]
                        if p_date!=max_p_date:
                            for na2 in na2s:
                                cursor.execute("SELECT sum(nachislen_n-umenshen_n) FROM nl WHERE date_srok>%s and date_srok<=%s and date_real<=%s and tin=%s and na2_code=%s", (p_date, date, p_date, tin, na2, ))
                                pr_sum = cursor.fetchone()[0]
                                cursor.execute("update saldo set pr_sum=ifnull(%s,0) where date_srok=%s and tin=%s and na2_code=%s", (pr_sum, p_date, tin, na2, ))
                                dbs.commit()
                                #saldo1 = (saldo_all if saldo_all is not None else 0) - (pr_sum if pr_sum is not None else 0)
                                cursor.execute("update saldo set saldo_pr=saldo_all-ifnull(%s,0) where date_srok=%s and tin=%s and na2_code=%s", (pr_sum, p_date, tin, na2, ))
                                dbs.commit()
                                cursor.execute("SELECT sum(nachislen_n-umenshen_n) FROM nl WHERE date_srok>%s and date_srok<=%s and date_real<=%s and tin=%s", (p_date, date, p_date, tin, ))
                                pr_sum_all = cursor.fetchone()[0]
                                cursor.execute("update saldo set pr_sum_all=ifnull(%s,0) where date_srok=%s and tin=%s and na2_code=%s", (pr_sum_all, p_date, tin, na2, ))
                                dbs.commit()
                                cursor.execute("update saldo set saldo_all_codes_pr=saldo_all_codes-ifnull(%s,0) where date_srok=%s and tin=%s", (pr_sum_all, p_date, tin, ))
                                dbs.commit()

                                last_date=cursor.execute("select max(last_date_srok) from saldo where date_srok=%s and tin=%s", (p_date, tin, ))
                                result = cursor.fetchone()
                                if result is not None:
                                    min_last_date = result[0]
                                else:
                                    min_last_date=p_date
                                cursor.execute("SELECT DATEDIFF(date_srok, %s) from saldo where date_srok=%s and tin=%s", (min_last_date, p_date, tin, ))
                                farq = cursor.fetchone()
                                cursor.execute("SELECT sum(saldo_pr) FROM saldo WHERE date_srok=%s and tin=%s", (min_last_date, tin, ))
                                last_saldo_all_codes = cursor.fetchone()[0]
                                cursor.execute("SELECT saldo_pr FROM saldo WHERE date_srok=%s and tin=%s and na2_code=%s", (min_last_date, tin, na2, ))
                                last_saldo_all = cursor.fetchone()[0]
                                cursor.execute("SELECT sum(saldo_pr) FROM saldo WHERE date_srok=%s and tin=%s and saldo_pr<0", (min_last_date, tin, ))
                                last_saldo_all_codes_ned = cursor.fetchone()[0]
                                cursor.execute("update saldo set pr_peny_all=(-1)*(%s)*(%s)*(%s)*(%s) where date_srok=%s and %s<0 and tin=%s", (last_saldo_all_codes, farq, foiz, ulush, p_date, last_saldo_all_codes, tin, ))
                                dbs.commit()
                                cursor.execute("update saldo set penya2=pr_peny_all*((%s)/(%s)) where %s<0 and %s<0 and date_srok=%s and tin=%s and na2_code=%s", (last_saldo_all, last_saldo_all_codes_ned, last_saldo_all, last_saldo_all_codes, p_date, tin, na2, ))
                                dbs.commit()
                            dbs.commit()
                        else:
                            for na2 in na2s:
                                cursor.execute("SELECT sum(nachislen_n-umenshen_n) FROM nl WHERE date_srok>%s and date_srok<=%s and date_real<=%s and tin=%s and na2_code=%s", (p_date, date, p_date, tin, na2, ))
                                pr_sum = cursor.fetchone()[0]
                                cursor.execute("update saldo set pr_sum=ifnull(%s,0) where date_srok=%s and tin=%s and na2_code=%s", (pr_sum, p_date, tin, na2, ))
                                dbs.commit()
                                #saldo1 = (saldo_all if saldo_all is not None else 0) - (pr_sum if pr_sum is not None else 0)
                                cursor.execute("update saldo set saldo_pr=saldo_all-ifnull(%s,0) where date_srok=%s and tin=%s and na2_code=%s", (pr_sum, p_date, tin, na2, ))
                                dbs.commit()
                                cursor.execute("SELECT sum(nachislen_n-umenshen_n) FROM nl WHERE date_srok>%s and date_srok<=%s and date_real<=%s and tin=%s", (p_date, date, p_date, tin, ))
                                pr_sum_all = cursor.fetchone()[0]
                                cursor.execute("update saldo set pr_sum_all=ifnull(%s,0) where date_srok=%s and tin=%s and na2_code=%s", (pr_sum_all, p_date, tin, na2, ))
                                dbs.commit()
                                cursor.execute("update saldo set saldo_all_codes_pr=saldo_all_codes-ifnull(%s,0) where date_srok=%s and tin=%s", (pr_sum_all, p_date, tin, ))
                                dbs.commit()

                                last_date=cursor.execute("select max(last_date_srok) from saldo where date_srok=%s and tin=%s", (p_date, tin, ))
                                result = cursor.fetchone()
                                if result is not None:
                                    min_last_date = result[0]
                                else:
                                    min_last_date=p_date
                                cursor.execute("SELECT DATEDIFF(date_srok, %s) from saldo where date_srok=%s and tin=%s", (min_last_date, p_date, tin, ))
                                farq = cursor.fetchone()
                                cursor.execute("SELECT sum(saldo_pr) FROM saldo WHERE date_srok=%s and tin=%s", (min_last_date, tin, ))
                                last_saldo_all_codes = cursor.fetchone()[0]
                                cursor.execute("SELECT saldo_pr FROM saldo WHERE date_srok=%s and tin=%s and na2_code=%s", (min_last_date, tin, na2, ))
                                last_saldo_all = cursor.fetchone()[0]
                                cursor.execute("SELECT sum(saldo_pr) FROM saldo WHERE date_srok=%s and tin=%s and saldo_pr<0", (min_last_date, tin, ))
                                last_saldo_all_codes_ned = cursor.fetchone()[0]
                                cursor.execute("update saldo set pr_peny_all=(-1)*(%s)*(%s)*(%s)*(%s) where date_srok=%s and %s<0 and tin=%s", (last_saldo_all_codes, farq, foiz, ulush, p_date, last_saldo_all_codes, tin, ))
                                dbs.commit()
                                cursor.execute("update saldo set penya2=pr_peny_all*((%s)/(%s)) where %s<0 and %s<0 and date_srok=%s and tin=%s and na2_code=%s", (last_saldo_all, last_saldo_all_codes_ned, last_saldo_all, last_saldo_all_codes, p_date, tin, na2, ))
                                dbs.commit()
                            dbs.commit()
                    #pr>2020
                    else:
                        cursor.execute("select distinct(na2_code) from nl where tin=%s AND na2_code NOT IN ('46', '101', '199', '191') and date_srok=%s", (tin, date, ))
                        na22 = [row[0] for row in cursor.fetchall()]
                        cursor.execute("SELECT max(date_srok) FROM saldo WHERE date_srok >= %s and date_srok<=%s and tin=%s", (latest_date_real, date, tin,))
                        max_p_date = cursor.fetchone()[0]
                        if p_date!=max_p_date:
                            for na2 in na22:
                                cursor.execute("SELECT sum(nachislen_n-umenshen_n) FROM nl WHERE date_srok>%s and date_srok<=%s and date_real<=%s and tin=%s and na2_code=%s", (p_date, date, p_date, tin, na2, ))
                                pr_sum = cursor.fetchone()[0]
                                cursor.execute("update saldo set pr_sum=ifnull(%s,0) where date_srok=%s and tin=%s and na2_code=%s", (pr_sum, p_date, tin, na2, ))
                                dbs.commit()
                                cursor.execute("update saldo set saldo_pr=saldo_all-ifnull(%s,0) where date_srok=%s and tin=%s and na2_code=%s", (pr_sum, p_date, tin, na2, ))
                                dbs.commit()

                                last_date=cursor.execute("select last_date_srok from saldo where date_srok=%s and tin=%s and na2_code=%s", (p_date, tin, na2, ))
                                result = cursor.fetchone()
                                if result is not None:
                                    min_last_date = result[0]
                                else:
                                    min_last_date=p_date
                                cursor.execute("SELECT DATEDIFF(date_srok, last_date_srok) from saldo where date_srok=%s and tin=%s and na2_code=%s", (p_date, tin, na2, ))
                                farq = cursor.fetchone()
                                cursor.execute("SELECT saldo_pr FROM saldo WHERE date_srok=%s and tin=%s and na2_code=%s", (min_last_date, tin, na2, ))
                                last_saldo_all3 = cursor.fetchone()
                                if last_saldo_all3 is not None:
                                    last_saldo_all = last_saldo_all3[0]
                                else:
                                    last_saldo_all=0
                                #cursor.execute("update saldo set penya2=peny_all*((saldo_pr)/(%s)) where saldo_pr<0 and %s<0 and date_srok=%s and tin=%s and na2_code=%s", (last_saldo_all_codes, last_saldo_all_codes, date, tin, na2, ))
                                #dbs.commit()
                                cursor.execute("update saldo set penya2=(-1)*(%s)*(%s)*(%s)*(%s) where date_srok=%s and %s<0 and tin=%s and na2_code=%s", (last_saldo_all, farq, foiz, ulush, p_date, last_saldo_all, tin, na2, ))
                                dbs.commit()
                            dbs.commit()
                        else:
                            for na2 in na22:
                                cursor.execute("SELECT sum(nachislen_n-umenshen_n) FROM nl WHERE date_srok>%s and date_srok<=%s and date_real<=%s and tin=%s and na2_code=%s", (p_date, date, p_date, tin, na2, ))
                                pr_sum = cursor.fetchone()[0]
                                cursor.execute("update saldo set pr_sum=ifnull(%s,0) where date_srok=%s and tin=%s and na2_code=%s", (pr_sum, p_date, tin, na2, ))
                                dbs.commit()
                                cursor.execute("update saldo set saldo_pr=saldo_all-ifnull(%s,0) where date_srok=%s and tin=%s and na2_code=%s", (pr_sum, p_date, tin, na2, ))
                                dbs.commit()

                                last_date=cursor.execute("select last_date_srok from saldo where date_srok=%s and tin=%s and na2_code=%s", (p_date, tin, na2, ))
                                result = cursor.fetchone()
                                if result is not None:
                                    min_last_date = result[0]
                                else:
                                    min_last_date=p_date
                                cursor.execute("SELECT DATEDIFF(date_srok, last_date_srok) from saldo where date_srok=%s and tin=%s and na2_code=%s", (p_date, tin, na2, ))
                                farq = cursor.fetchone()
                                cursor.execute("SELECT saldo_pr FROM saldo WHERE date_srok=%s and tin=%s and na2_code=%s", (min_last_date, tin, na2, ))
                                last_saldo_all = cursor.fetchone()[0]
                                cursor.execute("update saldo set penya2=(-1)*(%s)*(%s)*(%s)*(%s) where date_srok=%s and %s<0 and tin=%sand na2_code=%s", (last_saldo_all, farq, foiz, ulush, p_date, last_saldo_all, tin, na2, ))
                                dbs.commit()
                                cursor.execute("SELECT sum(penya1) FROM saldo WHERE date_srok>=%s and date_srok<=%s and tin=%s and na2_code=%s", (latest_date_real, p_date, tin, na2, ))
                                sum_penya1 = cursor.fetchone()[0]
                                cursor.execute("SELECT sum(penya2) FROM saldo WHERE date_srok>=%s and date_srok<=%s and tin=%s and na2_code=%s", (latest_date_real, p_date, tin, na2, ))
                                sum_penya2 = cursor.fetchone()[0]
                                cursor.execute("SELECT sum(sum_pr_penya) FROM saldo WHERE date_srok>=%s and date_srok<%s and tin=%s and na2_code=%s", (latest_date_real, p_date, tin, na2, ))
                                sum_pr_penya = cursor.fetchone()[0]
                                cursor.execute("update saldo set sum_pr_penya=%s-%s-%s where date_srok=%s and tin=%s and na2_code=%s", (sum_penya2, sum_penya1, sum_pr_penya, p_date, tin, na2, ))
                                dbs.commit()
                            dbs.commit()
                    dbs.commit()
            #pr_bolmasa
            else:
                for na2 in na2s:
                    last_date=cursor.execute("select last_date_srok from saldo where date_srok=%s and tin=%s and na2_code=%s", (date, tin, na2, ))
                    result = cursor.fetchone()
                    if result is not None:
                        min_last_date = result[0]
                    else:
                        min_last_date=date
                    cursor.execute("SELECT DATEDIFF(date_srok, last_date_srok) from saldo where date_srok=%s and tin=%s", (date, tin, ))
                    farq = cursor.fetchone()
                    now = datetime.now(tz=timezone(timedelta(hours=5)))
                    cursor.execute("SELECT sum(pereplata-saldo_tek_n) FROM saldo WHERE date_srok=%s and tin=%s and na2_code=%s", (date, tin, na2, ))
                    saldo_all = cursor.fetchone()[0]
                    cursor.execute("update saldo set saldo_all=%s where date_srok=%s and tin=%s and na2_code=%s", (saldo_all, date, tin, na2, ))
                    dbs.commit()
                    cursor.execute("SELECT saldo_all FROM saldo WHERE date_srok=%s and tin=%s and na2_code=%s", (min_last_date, tin, na2, ))
                    last_saldo_all1 = cursor.fetchone()
                    if last_saldo_all1 is not None:
                        last_saldo_all = last_saldo_all1[0]
                    else:
                        last_saldo_all=0
                    cursor.execute("update saldo set penya1=(-1)*(%s)*(%s)*(%s)*(%s) where date_srok=%s and %s<0 and tin=%s and na2_code=%s", (last_saldo_all, farq, foiz, ulush, date, last_saldo_all, tin, na2, ))
                    dbs.commit()
                    #cursor.execute("update saldo set penya1=peny_all*((%s)/(%s)) where %s<0 and %s<0 and date_srok=%s and tin=%s and na2_code=%s", (last_saldo_all, last_saldo_all_codes, last_saldo_all, last_saldo_all_codes, date, tin, na2, ))
                    #dbs.commit()
                dbs.commit()

    return redirect(url_for("nl"))


@app.route('/filter_nl', methods=["POST", "GET"])
def filter_nl():
    msgg = ''
    # user_ip = request.remote_addr  # Get user's IP address
    # store_ip_in_database(user_ip)
    app.config["DEBUG"] = True
    app.secret_key = "789456asd"
    user = 'hbnnarzullayev'
    password = 'Sersarson7'
    host = 'hbnnarzullayev.mysql.pythonanywhere-services.com'
    mysql = 'hbnnarzullayev$flask3'
    dbs = msd.connect(host,user,password, mysql)
    cursor = dbs.cursor()
    tin = request.form.get('tin1')
    cursor.execute("SELECT date_srok FROM pereraschet WHERE date_srok > date_real AND (nachislen_n + umenshen_n) <> 0 AND na2_code NOT IN ('46', '101', '199', '191') and tin=%s GROUP BY date_srok ORDER BY date_srok asc", (tin,))
    pr_dates = [row[0] for row in cursor.fetchall()]
    finish_result = None
    cursor.execute("delete from start where tin=%s", (tin, ))
    # cursor.execute("INSERT INTO start (date_srok) SELECT text FROM nl WHERE ns51_code='2' AND text > text1 AND (nachislen_n + umenshen_n) <> 0 AND na2_code NOT IN ('46', '101', '199', '191') GROUP BY text ORDER BY text ASC")
    for pr_date1 in pr_dates:
        now = datetime.now(tz=timezone(timedelta(hours=5)))
        cursor.execute("SELECT MIN(date_real) as start_value FROM pereraschet WHERE date_srok=%s AND na2_code NOT IN ('46', '101', '199', '191') and tin=%s", (pr_date1,tin, ))
        start_value = cursor.fetchone()[0]
        cursor.execute("insert into start (tin, date, date_srok, datetime) values(%s, %s, %s, %s)",(tin, start_value,pr_date1, now.strftime('%Y-%m-%d %H:%M:%S')))
        while start_value is not None:
            now = datetime.now(tz=timezone(timedelta(hours=5)))
            cursor.execute("SELECT MIN(date_real) as result FROM pereraschet WHERE date_srok >= %s AND date_real < %s AND (nachislen_n + umenshen_n) <> 0 AND date_srok <= %s AND na2_code NOT IN ('46', '101', '199', '191') and date_srok>date_real and tin=%s", (start_value, start_value, pr_date1,tin,))
            result = cursor.fetchone()
            next_value = result[0]
            start_value = next_value
            if result is not None:
                now = datetime.now(tz=timezone(timedelta(hours=5)))
                start_value = next_value
                cursor.execute("insert into start (tin, date, date_srok, datetime) values(%s, %s, %s, %s)",(tin, result,pr_date1, now.strftime('%Y-%m-%d %H:%M:%S')))
                dbs.commit()
                # cursor.execute("update start set date=%s, datetime=%s where date_srok=%s", (result,now.strftime('%Y-%m-%d %H:%M:%S'), pr_date1, ))
            else:
                cursor.execute("insert into start (tin, date, date_srok, datetime) values(%s, %s, %s, %s)",(tin, result,pr_date1, now.strftime('%Y-%m-%d %H:%M:%S')))
                dbs.commit()
            dbs.commit()
    cursor.execute("SELECT STR_TO_DATE('01, 01, 2020', '%d, %m, %Y')")
    simple_year = cursor.fetchone()[0]
    cursor.execute("select distinct(date_srok) from start where tin=%s  and date_srok>=%s group by date_srok", (tin, simple_year, ))
    pr_dates1 = [row[0] for row in cursor.fetchall()]
    cursor.execute("select distinct(date_end) from fine_percent")
    pr_dates3 = [row[0] for row in cursor.fetchall()]
    # cursor.execute("SELECT DISTINCT(date_srok) AS mixed_dates from(SELECT DISTINCT(date_srok) FROM start JOIN ulush t ON start.date_srok = t.data WHERE tin = %s AND start.date_srok >= %s union select Date_end as date_column from fine_percent where Date_end>='2020-01-01') as combined_dates;", (tin, simple_year, ))
    cursor.execute("SELECT DISTINCT date_srok AS mixed_dates FROM start WHERE tin = %s AND start.date_srok >= %s union all SELECT DISTINCT t.date_end AS mixed_dates FROM fine_percent t WHERE t.Date_end >= %s and not exists (select 1 from nla f where f.date_srok =t.Date_end) order by mixed_dates asc;", (tin, simple_year, simple_year, ))
    pr_dates2 = [row[0] for row in cursor.fetchall()]
    for pr_date1 in pr_dates2:
        cursor.execute("select 1 from start where date_srok=%s and tin=%s",(pr_date1,tin, ))
        min_pr_date1 = cursor.fetchone()
        if min_pr_date1 != (1,):
            cursor.execute("select %s from nla where tin=%s",(pr_date1,tin, ))
            min_pr_date = cursor.fetchone()[0]
            if pr_date1>simple_year:
                cursor.execute("select min(ynl) from nla where date_srok<>%s  and tin=%s group by ynl;", (pr_date1, tin, ))
                ynl = cursor.fetchone()[0]
                cursor.execute("select ns10_code from nla where tin=%s group by ns10_code;", (tin, ))
                ns10_code = cursor.fetchone()[0]
                cursor.execute("select ns11_code from nla where tin=%s group by ns11_code;", (tin, ))
                ns11_code = cursor.fetchone()[0]
                for na2_code in na2_codes:
                    cursor.execute("select max(date_srok) from nla where date_srok<=%s and na2_code=%s and tin=%s;",(pr_date1, na2_code, tin, ))
                    last_date_pr = cursor.fetchone()[0]
                    cursor.execute("select min(date_srok) from nla where date_srok>=%s and na2_code=%s and tin=%s;",(pr_date1, na2_code, tin, ))
                    max_date_pr = cursor.fetchone()[0]
                    cursor.execute("INSERT INTO nla (ynl, ns10_code, ns11_code, tin, na2_code, date_srok, nachislen_n, umenshen_n, sum_n, uploch_n, vozvrat, pr_penya, last_date_srok) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)", (ynl, ns10_code, ns11_code, tin, na2_code, pr_date1, 0, 0, 0, 0, 0, 0, last_date_pr,))
                    # cursor.execute("update nla set last_date_srok=%s where date_srok=%s and na2_code=%s", (pr_date1, max_date_pr, na2_code, ))
                dbs.commit()
        elif pr_date1>simple_year:
            cursor.execute("select min(date) from start where date_srok=%s and tin=%s",(pr_date1,tin, ))
            min_pr_date = cursor.fetchone()[0]
            if pr_date1>simple_year and min_pr_date<simple_year:
                cursor.execute("SELECT DISTINCT na2_code FROM nla where tin=%s  and date_srok=%s GROUP BY na2_code ORDER BY na2_code", (tin, pr_date1, ))
                pr_na2ss = cursor.fetchall()
                pr_na2ss2 = [row[0] for row in pr_na2ss]
                placeholders = ', '.join(['%s'] * len(pr_na2ss2))
                cursor.execute("SELECT DISTINCT na2_code FROM nla WHERE tin = %s AND na2_code NOT IN ({}) GROUP BY na2_code ORDER BY na2_code".format(placeholders), (tin,) + tuple(pr_na2ss2))
                na2_codes = [row[0] for row in cursor.fetchall()]
                cursor.execute("select ynl from nla where date_srok=%s group by ynl;", (pr_date1, ))
                ynl = cursor.fetchone()[0]
                cursor.execute("select ns10_code from nla where tin=%s group by ns10_code;", (tin, ))
                ns10_code = cursor.fetchone()[0]
                cursor.execute("select ns11_code from nla where tin=%s group by ns11_code;", (tin, ))
                ns11_code = cursor.fetchone()[0]
                for na2_code in na2_codes:
                    cursor.execute("select max(date_srok) from nla where date_srok<%s and na2_code=%s and tin=%s;",(pr_date1, na2_code, tin, ))
                    last_date_pr = cursor.fetchone()[0]
                    cursor.execute("select min(date_srok) from nla where date_srok>=%s and na2_code=%s and tin=%s;",(pr_date1, na2_code, tin, ))
                    max_date_pr = cursor.fetchone()[0]
                    cursor.execute("INSERT INTO nla (ynl, ns10_code, ns11_code, tin, na2_code, date_srok, nachislen_n, umenshen_n, sum_n, uploch_n, vozvrat, pr_penya, datetime, last_date_srok) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)", (ynl, ns10_code, ns11_code, tin, na2_code, pr_date1, 0, 0, 0, 0, 0, 0, now.strftime('%Y-%m-%d %H:%M:%S'), last_date_pr,))
                    cursor.execute("update nla set last_date_srok=%s where date_srok=%s and na2_code=%s", (pr_date1, max_date_pr, na2_code, ))
                dbs.commit()
        # elif pr_date1>simple_year and min_pr_date<simple_year:
        #     cursor.execute("select min(date) from start where date_srok=%s and tin=%s",(pr_date1,tin, ))
        #     min_pr_date = cursor.fetchone()[0]
        #     if pr_date1>simple_year and min_pr_date<simple_year:
        #         cursor.execute("SELECT DISTINCT na2_code FROM nla where tin=%s  and date_srok=%s GROUP BY na2_code ORDER BY na2_code", (tin, pr_date1, ))
        #         pr_na2ss = cursor.fetchall()
        #         pr_na2ss2 = [row[0] for row in pr_na2ss]
        #         placeholders = ', '.join(['%s'] * len(pr_na2ss2))
        #         cursor.execute("SELECT DISTINCT na2_code FROM nla WHERE tin = %s AND na2_code NOT IN ({}) GROUP BY na2_code ORDER BY na2_code".format(placeholders), (tin,) + tuple(pr_na2ss2))
        #         na2_codes = [row[0] for row in cursor.fetchall()]
        #         cursor.execute("select ynl from nla where date_srok=%s  and tin=%s group by ynl;", (pr_date1, tin, ))
        #         ynl = cursor.fetchone()[0]
        #         cursor.execute("select ns10_code from nla where tin=%s group by ns10_code;", (tin, ))
        #         ns10_code = cursor.fetchone()[0]
        #         cursor.execute("select ns11_code from nla where tin=%s group by ns11_code;", (tin, ))
        #         ns11_code = cursor.fetchone()[0]
        #         for na2_code in na2_codes:
        #             cursor.execute("select max(date_srok) from nla where date_srok<%s and na2_code=%s and tin=%s;",(pr_date1, na2_code, tin, ))
        #             last_date_pr = cursor.fetchone()[0]
        #             cursor.execute("select max(date_srok) from nla where date_srok<%s and na2_code=%s and tin=%s;",(pr_date1, na2_code, tin, ))
        #             max_date_pr = cursor.fetchone()[0]
        #             cursor.execute("INSERT INTO nla (ynl, ns10_code, ns11_code, tin, na2_code, date_srok, nachislen_n, umenshen_n, sum_n, uploch_n, vozvrat, pr_penya, datetime, last_date_srok) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)", (ynl, ns10_code, ns11_code, tin, na2_code, pr_date1, 0, 0, 0, 0, 0, 0, now.strftime('%Y-%m-%d %H:%M:%S'), last_date_pr,))
        #             cursor.execute("update nla set last_date_srok=%s where date_srok=%s and na2_code=%s", (pr_date1, max_date_pr, na2_code, ))
        #         dbs.commit()
        # dbs.close()
    cursor.execute("select distinct(date_end) from fine_percent")
    fine_dates = [row[0] for row in cursor.fetchall()]
    cursor.execute("SELECT DISTINCT date_srok AS mixed_dates FROM start WHERE tin = %s AND start.date_srok >= %s union all SELECT DISTINCT t.date_end AS mixed_dates FROM fine_percent t WHERE t.Date_end >= %s and not exists (select 1 from nla f where f.date_srok =t.Date_end) order by mixed_dates asc;", (tin, simple_year, simple_year, ))
    pr_dates2 = [row[0] for row in cursor.fetchall()]
    for fine_date in fine_dates:
        # if fine_date>simple_year and fine_date not in pr_dates2:
        #     cursor.execute("SELECT DISTINCT na2_code FROM nla where tin = %s and date_srok = %s GROUP BY na2_code ORDER BY na2_code", (tin, fine_date, ))
        #     pr_na2ss = cursor.fetchall()
        #     pr_na2ss2 = [row[0] for row in pr_na2ss]
        #     placeholders = ', '.join(['%s'] * len(pr_na2ss2))
        #     if pr_na2ss2 is not None:
        #         query = "SELECT DISTINCT na2_code FROM nla WHERE tin = %s AND na2_code NOT IN ({}) ORDER BY na2_code".format(placeholders)
        #         params = (tin,) + tuple(pr_na2ss2)
        #         cursor.execute(query, params)
        #         na2_codes = [row[0] for row in cursor.fetchall()]
        #     else:
        #         cursor.execute("SELECT DISTINCT na2_code FROM nla WHERE tin = %s ORDER BY na2_code", (tin, ))
        #         na2_codes = [row[0] for row in cursor.fetchall()]
        if fine_date > simple_year and fine_date not in pr_dates2:
            cursor.execute("SELECT DISTINCT na2_code FROM nla where tin = %s and date_srok = %s GROUP BY na2_code ORDER BY na2_code", (tin, fine_date, ))
            pr_na2ss = cursor.fetchall()
            if pr_na2ss:  # Checking if pr_na2ss is not empty
                pr_na2ss2 = [row[0] for row in pr_na2ss]
                placeholders = ', '.join(['%s'] * len(pr_na2ss2))
                query = "SELECT DISTINCT na2_code FROM nla WHERE tin = %s AND na2_code NOT IN ({}) GROUP BY na2_code ORDER BY na2_code".format(placeholders)
                params = (tin,) + tuple(pr_na2ss2)
                cursor.execute(query, params)
                na2_codes = [row[0] for row in cursor.fetchall()]
            else:
                cursor.execute("SELECT DISTINCT na2_code FROM nla WHERE tin = %s ORDER BY na2_code", (tin, ))
                na2_codes = [row[0] for row in cursor.fetchall()]
            cursor.execute("select ynl from nla where date_srok=%s group by ynl;", (fine_date, ))
            ynl = cursor.fetchone()[0]
            cursor.execute("select ns10_code from nla where tin=%s group by ns10_code;", (tin, ))
            ns10_code = cursor.fetchone()[0]
            cursor.execute("select ns11_code from nla where tin=%s group by ns11_code;", (tin, ))
            ns11_code = cursor.fetchone()[0]
            for na2_code in na2_codes:
                cursor.execute("select max(date_srok) from nla where date_srok<%s and na2_code=%s and tin=%s;",(fine_date, na2_code, tin, ))
                last_date_pr = cursor.fetchone()[0]
                cursor.execute("select min(date_srok) from nla where date_srok>=%s and na2_code=%s and tin=%s;",(fine_date, na2_code, tin, ))
                max_date_pr = cursor.fetchone()[0]
                cursor.execute("INSERT INTO nla (ynl, ns10_code, ns11_code, tin, na2_code, date_srok, nachislen_n, umenshen_n, sum_n, uploch_n, vozvrat, pr_penya, datetime, last_date_srok) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)", (ynl, ns10_code, ns11_code, tin, na2_code, fine_date, 0, 0, 0, 0, 0, 0, now.strftime('%Y-%m-%d %H:%M:%S'), last_date_pr,))
                cursor.execute("update nla set last_date_srok=%s where date_srok=%s and na2_code=%s", (fine_date, max_date_pr, na2_code, ))
            dbs.commit()
    return redirect(url_for("nl"))


# @app.route('/saldo', methods=["POST", "GET"])
# # def trigger_mysql_function():
# #     background_thread = threading.Thread(target=saldo_task, args=(session['nma'], request.form.get('radio_choice'),request.form.get('selected_date'), request.form.get('manual_date'),request.form.get('tin_t'),))
# #     background_thread.start()
# #     return redirect(url_for("nl"))

# def trigger_mysql_function():
#     # Instead of creating a thread, call the Celery task
#     saldo_task.delay(session['nma'], request.form.get('radio_choice'),
#                      request.form.get('selected_date'), request.form.get('manual_date'),
#                      request.form.get('tin_t'))
#     print('Shuyer+')
#     return redirect(url_for("nl"))

# @celery.task
@socketio.on('start_processing')
# def saldo_task(session_nma, radio_choice, selected_date, manual_date, tin_t):
@app.route('/saldo', methods=["POST", "GET"])
def saldo():
    msgg = ''
    app.config["DEBUG"] = True
    app.secret_key = "789456asd"
    user = 'hbnnarzullayev'
    password = 'Sersarson7'
    host = 'hbnnarzullayev.mysql.pythonanywhere-services.com'
    mysql = 'hbnnarzullayev$flask3'
    dbs = msd.connect(host,user,password, mysql)
    # if session_nma in ('admin', 'Admin'):
    if 'loggedin' in session and session['nma'] in ('admin','Admin'):
        while True:
            radio_choice = request.form.get('radio_choice')
            if radio_choice == 'option1':  # User chose to select from dropdown
                min_date_srok = request.form.get('selected_date')
                # min_date_srok = selected_date
            else:  # User chose to enter date manually
                min_date_srok = request.form.get('manual_date')
                # min_date_srok = manual_date
            tin = request.form.get('tin_t')
            # tin=tin_t
            cursor = dbs.cursor()
            cursor.execute("update nla set pr_penya=0, sum_p=0 where date_srok>=%s and tin=%s", (min_date_srok,tin,))
            cursor.execute("SELECT STR_TO_DATE('01, 01, 2020', '%d, %m, %Y')")
            simple_year = cursor.fetchone()[0]
            if request.form.get('end_date') is not None and request.form.get('end_date') != "":
                end_date=request.form.get('end_date')
                cursor.execute("SELECT DISTINCT date_srok FROM nla where date_srok>=%s and date_srok<=%s and tin=%s GROUP BY date_srok ORDER BY date_srok asc",(min_date_srok, end_date, tin, ))
                dates = [row[0] for row in cursor.fetchall()]
            else:  # User chose to enter date manually
                cursor.execute("SELECT DISTINCT date_srok FROM nla where date_srok>=%s and tin=%s GROUP BY date_srok ORDER BY date_srok asc",(min_date_srok, tin, ))
                dates = [row[0] for row in cursor.fetchall()]
            #cursor.execute("SELECT DISTINCT date_srok FROM nla where date_srok>=%s and tin=%s GROUP BY date_srok ORDER BY date_srok asc",(min_date_srok, tin, ))
            #dates = [row[0] for row in cursor.fetchall()]
            # Fetch distinct na2_codes
            cursor.execute("SELECT DISTINCT na2_code FROM nla where tin=%s GROUP BY na2_code ORDER BY na2_code", (tin, ))
            na2_codes = [row[0] for row in cursor.fetchall()]
            cursor.execute("select date_srok from start  where tin=%s group by date_srok", (tin, ))
            pr_dates = [row[0] for row in cursor.fetchall()]
            cursor.execute("select DATE_SROK from nl where xar in ('1') and na2_code not in ('46','101','199','191') group by YNL, NS10_CODE,  NS11_CODE,  tin, NA2_CODE,  DATE_SROK,  OWN_NS10_CODE,  OWN_NS11_CODE order by date_srok desc, na2_code asc")
            xar_1_dates = [row[0] for row in cursor.fetchall()]
            for current_date in dates:
                #cursor.execute("update nla set nachislen_n=0, umenshen_n=0, uploch_n=0 where date_srok>%s and tin=%s", (current_date,tin,))
                #cursor.execute("update nla set pr_penya=0, sum_p=0 where date_srok>%s and tin=%s", (current_date,tin,))
                cursor.execute("SELECT SUM(saldo_all) FROM nla WHERE date_srok = %s  and tin=%s group by date_srok", (current_date, tin, ))
                saldo_all_codes = cursor.fetchone()[0]
                cursor.execute("UPDATE nla SET saldo_all_codes = %s WHERE last_date_srok = %s and tin=%s", (saldo_all_codes, current_date, tin, ))
                cursor.execute("select (procent/koef) from ulush where data=%s", (current_date,))
                foiz = cursor.fetchone()[0]
                cursor.execute("select ulush as ulush from ulush where data=%s", (current_date,))
                ulush = cursor.fetchone()[0]
                cursor.execute("select sum(saldo_all) as summa from nla where date_srok=%s and saldo_all<'0' and tin=%s", (current_date, tin, ))
                last_neds_sum = cursor.fetchone()[0]
                cursor.execute("select sum(saldo_all) as sum from nla where saldo_all<0 and date_srok=%s and tin=%s", (current_date, tin, ))
                neds = cursor.fetchone()[0]
                if current_date < simple_year:
                    if current_date in pr_dates:
                        cursor.execute("select max(last_date_srok) from nlb where pr_saldo_all_codes<>0 and tin=%s", (tin, ))
                        max_last_pr_date = cursor.fetchone()[0]
                        cursor.execute("select max(last_date_srok) from nlb where tin=%s", (tin, ))
                        max_last_pr_dates = cursor.fetchone()[0]
                        if max_last_pr_date is not None and (max_last_pr_dates!=max_last_pr_date):
                            cursor.execute("select Distinct last_date_srok from nlb where last_date_srok>=%s and tin=%s", (max_last_pr_date, tin, ))
                            nlb_dates1 = [row[0] for row in cursor.fetchall()]
                            for nlb_date1 in nlb_dates1:
                                cursor.execute("select (procent/koef) from ulush where data=%s", (nlb_date1,))
                                foiz = cursor.fetchone()[0]
                                cursor.execute("select last_date_srok from nlb where date_srok=%s and tin=%s group by last_date_srok", (nlb_date1, tin, ))
                                last_date_srok_p = cursor.fetchone()
                                cursor.execute("select ulush as ulush from ulush where data=ifnull(%s,%s)", (nlb_date1, last_date_srok_p, ))
                                ulush = cursor.fetchone()[0]
                                cursor.execute("select sum(shtraf) from nlb where date_srok=%s and tin=%s", (nlb_date1, tin, ))
                                shtraf_all = cursor.fetchone()[0]
                                cursor.execute("SELECT sum(ifnull(nachislen_n-umenshen_n,0)) as sums_all FROM pereraschet WHERE date_real<= %s and na2_code not in ('46','101') and date_srok>%s and date_srok<=%s and date_srok>date_real and tin=%s", (nlb_date1, nlb_date1, current_date, tin, ))
                                pr_sum_all = cursor.fetchone()[0]
                                cursor.execute("UPDATE nlb SET pr_saldo_all_codes=(ifnull(%s,0)-ifnull(saldo_all_codes,0)+ifnull(%s,0)) where last_date_srok=%s and tin=%s", (pr_sum_all, shtraf_all, nlb_date1, tin, ))
                                cursor.execute("select max(last_date_srok) from nlb where tin=%s", (tin, ))
                                last_date=cursor.fetchone()[0]
                                for na2_code1 in na2_codes:
                                    now = datetime.now(tz=timezone(timedelta(hours=5)))
                                    cursor.execute("SELECT sum(nachislen_n-umenshen_n) as sums FROM pereraschet WHERE  date_srok<=%s and date_real<= %s and date_srok>%s and na2_code=%s and date_srok>date_real and tin=%s", (current_date, nlb_date1, nlb_date1, na2_code1, tin, ))
                                    farq = cursor.fetchone()
                                    cursor.execute("SELECT sum(nachislen_n-umenshen_n) as sums FROM pereraschet WHERE  date_srok<=%s and date_real<= %s and date_srok>%s and na2_code=%s and date_srok>date_real and tin=%s", (current_date, nlb_date1, nlb_date1, na2_code1, tin, ))
                                    pr_sum = cursor.fetchone()[0]
                                    cursor.execute("UPDATE nlb SET pr_saldo_all=(ifnull(%s,0)+ifnull(shtraf,0)-oldingi_kun_saldo) where last_date_srok=%s and na2_code=%s and tin=%s", (pr_sum, nlb_date1, na2_code1, tin, ))
                                    cursor.execute("SELECT sum(pr_saldo_all) as sums FROM nlb WHERE pr_saldo_all>0 and last_date_srok=%s and tin=%s", (nlb_date1, tin, ))
                                    pr_ned_sum = cursor.fetchone()[0]
                                    cursor.execute("UPDATE nlb SET pr_penya_all = if(pr_saldo_all_codes>0,(pr_saldo_all_codes*(%s)*(%s)*(%s)),0) where last_date_srok=%s and na2_code=%s and tin=%s", (foiz,ulush,farq,nlb_date1, na2_code1, tin, ))
                                    # cursor.execute("UPDATE nlb SET pr_penya_all = if((saldo_all_codes-ifnull(%s,0))<0,((-1)*(saldo_all_codes-ifnull(%s,0))*(%s)*(%s)*(%s)),0) where last_date_srok=%s and na2_code=%s", (pr_sum_all, pr_sum_all, foiz,ulush,farq,nlb_date, na2_code1,))
                                    cursor.execute("SELECT date_srok from nlb where last_date_srok=%s and na2_code=%s and tin=%s", (nlb_date1, na2_code1, tin, ))
                                    penya_write_date1 = cursor.fetchone()
                                    if penya_write_date1 is not None:
                                        penya_write_date = penya_write_date1[0]
                                        if penya_write_date in xar_1_dates:
                                            cursor.execute("UPDATE nlb SET pr_penya = 0 where last_date_srok=%s and tin=%s", (nlb_date1, tin, ))
                                        else:
                                            cursor.execute("UPDATE nlb SET pr_penya = if((ifnull(pr_saldo_all,0))>0,((ifnull(pr_saldo_all,0))/%s)*pr_penya_all,0) where last_date_srok=%s and tin=%s", (pr_ned_sum, nlb_date1, tin, ))
                                    else:
                                        cursor.execute("UPDATE nlb SET pr_penya = if((ifnull(pr_saldo_all,0))>0,((ifnull(pr_saldo_all,0))/%s)*pr_penya_all,0) where last_date_srok=%s and tin=%s", (pr_ned_sum, nlb_date1, tin, ))
                                    # cursor.execute("UPDATE nlb SET pr_penya = if((ifnull(pr_saldo_all,0))>0,((ifnull(pr_saldo_all,0))/%s)*pr_penya_all,0) where last_date_srok=%s and tin=%s", (pr_ned_sum, nlb_date1, tin, ))
                                    cursor.execute("delete from pr_progress where pr_date!=%s and tin=%s", (nlb_date1, tin, ))
                                    cursor.execute("INSERT INTO pr_progress (tin, pr_date, na2_code, datetime) VALUES (%s, %s, %s, %s)", (tin, nlb_date1, na2_code1,now.strftime('%Y-%m-%d %H:%M:%S')))
                                dbs.commit()
                        else:
                            cursor.execute("delete from nlb where tin=%s", (tin, ))
                            cursor.execute("select min(date) from start where date_srok=%s and tin=%s",(current_date,tin, ))
                            min_pr_date = cursor.fetchone()[0]
                            cursor.execute("insert into nlb (tin, na2_code, date_srok, saldo_all, shtraf, oldingi_kun_saldo, saldo_all_codes,sum_p, saldo_tek_p, last_date_srok) select tin, na2_code, date_srok, saldo_all, ifnull(shtraf,0), oldingi_kun_saldo, saldo_all_codes, sum_p, saldo_tek_p, last_date_srok from nla where date_srok<=%s and last_date_srok>=%s and tin=%s order by date_srok, na2_code;", (current_date, min_pr_date, tin, ))
                            cursor.execute("select last_date_srok from nlb where tin=%s group by last_date_srok", (tin, ))
                            nlb_dates = [row[0] for row in cursor.fetchall()]
                            cursor.execute("UPDATE nlb SET pr_saldo_all_codes=0, pr_saldo_all=0, pr_penya_all=0, pr_penya=0 where tin=%s", (tin, ))
                            for nlb_date in nlb_dates:
                                cursor.execute("select (procent/koef) from ulush where data=%s", (nlb_date,))
                                foiz = cursor.fetchone()[0]
                                cursor.execute("select last_date_srok from nlb where date_srok=%s and tin=%s group by last_date_srok", (nlb_date, tin, ))
                                last_date_srok_p = cursor.fetchone()
                                cursor.execute("select ulush as ulush from ulush where data=ifnull(%s,%s)", (nlb_date, last_date_srok_p, ))
                                ulush = cursor.fetchone()[0]
                                cursor.execute("select sum(shtraf) from nlb where date_srok=%s and tin=%s", (nlb_date, tin, ))
                                shtraf_all = cursor.fetchone()[0]
                                cursor.execute("SELECT sum(ifnull(nachislen_n-umenshen_n,0)) as sums_all FROM pereraschet WHERE date_real<= %s and na2_code not in ('46','101') and date_srok>%s and date_srok<=%s and date_srok>date_real and tin=%s", (nlb_date, nlb_date, current_date, tin, ))
                                pr_sum_all = cursor.fetchone()[0]
                                cursor.execute("UPDATE nlb SET pr_saldo_all_codes=(ifnull(%s,0)-ifnull(saldo_all_codes,0)+ifnull(%s,0)) where last_date_srok=%s and tin=%s", (pr_sum_all, shtraf_all, nlb_date, tin, ))
                                cursor.execute("select max(last_date_srok) from nlb where tin=%s", (tin, ))
                                last_date=cursor.fetchone()[0]
                                for na2_code1 in na2_codes:
                                    now = datetime.now(tz=timezone(timedelta(hours=5)))
                                    cursor.execute("select DATEDIFF(date_srok,last_date_srok) as farq from nlb where last_date_srok=%s and na2_code=%s and tin=%s", (nlb_date,na2_code1,tin, ))
                                    farq = cursor.fetchone()
                                    cursor.execute("SELECT sum(nachislen_n-umenshen_n) as sums FROM pereraschet WHERE  date_srok<=%s and date_real<= %s and date_srok>%s and na2_code=%s and date_srok>date_real and tin=%s", (current_date, nlb_date, nlb_date, na2_code1, tin, ))
                                    pr_sum = cursor.fetchone()[0]
                                    cursor.execute("UPDATE nlb SET pr_saldo_all=(ifnull(%s,0)+ifnull(shtraf,0)-oldingi_kun_saldo) where last_date_srok=%s and na2_code=%s and tin=%s", (pr_sum, nlb_date, na2_code1, tin, ))
                                    cursor.execute("SELECT sum(pr_saldo_all) as sums FROM nlb WHERE pr_saldo_all>0 and last_date_srok=%s and tin=%s", (nlb_date, tin, ))
                                    pr_ned_sum = cursor.fetchone()[0]
                                    cursor.execute("UPDATE nlb SET pr_penya_all = if(pr_saldo_all_codes>0,(pr_saldo_all_codes*(%s)*(%s)*(%s)),0) where last_date_srok=%s and na2_code=%s and tin=%s", (foiz,ulush,farq,nlb_date, na2_code1,tin, ))
                                    # cursor.execute("UPDATE nlb SET pr_penya_all = if((saldo_all_codes-ifnull(%s,0))<0,((-1)*(saldo_all_codes-ifnull(%s,0))*(%s)*(%s)*(%s)),0) where last_date_srok=%s and na2_code=%s", (pr_sum_all, pr_sum_all, foiz,ulush,farq,nlb_date, na2_code1,))
                                    cursor.execute("SELECT date_srok from nla where last_date_srok=%s and na2_code=%s and tin=%s", (nlb_date, na2_code1, tin, ))
                                    penya_write_date = cursor.fetchone()[0]
                                    # cursor.execute("UPDATE nla SET sum_p_all = if((saldo_all_codes+%s)<0,(-1*(saldo_all_codes+%s)*(%s)*(%s)*(%s)),0) where last_date_srok=%s and na2_code=%s and tin=%s", (sum_shtraf_all, sum_shtraf_all, foiz,ulush,farq,current_date, na2_code1, tin, ))
                                    if penya_write_date in xar_1_dates:
                                        cursor.execute("UPDATE nlb SET pr_penya = '0' where last_date_srok=%s and tin=%s and na2_code=%s", (nlb_date, tin, na2_code1, ))
                                    else:
                                        cursor.execute("UPDATE nlb SET pr_penya = if((ifnull(pr_saldo_all,0))>0,((ifnull(pr_saldo_all,0))/%s)*pr_penya_all,0) where last_date_srok=%s and tin=%s and na2_code=%s", (pr_ned_sum, nlb_date, tin, na2_code1,))
                                    # cursor.execute("UPDATE nlb SET pr_penya = if((ifnull(pr_saldo_all,0))>0,((ifnull(pr_saldo_all,0))/%s)*pr_penya_all,0) where last_date_srok=%s and tin=%s", (pr_ned_sum, nlb_date, tin, ))
                                    cursor.execute("delete from pr_progress where pr_date!=%s and tin=%s", (nlb_date, tin, ))
                                    cursor.execute("INSERT INTO pr_progress (tin, pr_date, na2_code, datetime) VALUES (%s, %s, %s, %s)", (tin, nlb_date, na2_code1,now.strftime('%Y-%m-%d %H:%M:%S')))
                                dbs.commit()
                        cursor.execute("select max(date_srok) from nlb where tin=%s", (tin, ))
                        last_pr_date1 = cursor.fetchone()
                        if last_pr_date1 is not None:
                            last_pr_date = last_pr_date1[0]
                        else:
                            last_pr_date = 0
                        if current_date == last_pr_date:
                            for na2_code1 in na2_codes:
                                cursor.execute("select sum(pr_penya-ifnull(sum_p,0)) as sum_pr from nlb where na2_code=%s and tin=%s",(na2_code1, tin, ))
                                sum_pr = cursor.fetchone()[0]
                                cursor.execute("select ifnull(saldo_tek_p,0) from nlb where date_srok=%s and na2_code=%s and tin=%s", (current_date, na2_code1, tin, ))
                                saldo_tek_p1 = cursor.fetchone()[0]
                                cursor.execute("select ifnull(sum_p,0) from nlb where date_srok=%s and na2_code=%s and tin=%s", (current_date, na2_code1, tin, ))
                                sum_p1 = cursor.fetchone()[0]
                                cursor.execute("update nlb set nla_pr_penya=%s where date_srok=%s and na2_code=%s and tin=%s",(sum_pr, current_date, na2_code1, tin, ))
                                cursor.execute("update nla set pr_penya=%s, saldo_tek_p=%s+%s where date_srok=%s and na2_code=%s and tin=%s", (sum_pr, saldo_tek_p1, sum_pr, current_date, na2_code1, tin, ))
                                cursor.execute("update nla set sum_p=%s+%s where date_srok=%s and na2_code=%s and tin=%s", (sum_p1, sum_pr, current_date, na2_code1, tin, ))
                                cursor.execute("update nla set datetime=%s where date_srok=%s and tin=%s", (now.strftime('%Y-%m-%d %H:%M:%S'), current_date, tin, ))
                                dbs.commit()
                                now = datetime.now(tz=timezone(timedelta(hours=5)))
                                # Find the corresponding saldo_all value for the current date and na2_code
                                cursor.execute("SELECT IFNULL(saldo_all, 0) FROM nla WHERE date_srok = %s and na2_code=%s and tin=%s", (current_date, na2_code1, tin, ))
                                saldo_all = cursor.fetchone()[0]
                                cursor.execute("select sum(shtraf1) as summa from nla where date_srok=%s and shtraf1<>0 and na2_code=%s and tin=%s;", (current_date,na2_code1, tin, ))
                                last_shtraf = cursor.fetchone()[0]
                                cursor.execute("SELECT IFNULL(saldo_tek_p, 0) FROM nla WHERE date_srok = %s and na2_code=%s and tin=%s", (current_date, na2_code1, tin, ))
                                saldo_tek_p = cursor.fetchone()[0]
                                cursor.execute("UPDATE nla SET oldingi_kun_saldo = %s WHERE last_date_srok = %s AND na2_code = %s and tin=%s", (saldo_all, current_date, na2_code1, tin, ))
                                cursor.execute("select DATEDIFF(date_srok,last_date_srok) as farq from nla where last_date_srok=%s and na2_code=%s and tin=%s", (current_date,na2_code1,tin, ))
                                farq = cursor.fetchone()[0]
                                cursor.execute("UPDATE nla SET shtraf1 = case when (-1)*(%s+uploch_n+umenshen_n-nachislen_n-vozvrat-uploch_p)>0 and (-1)*(%s+uploch_n+umenshen_n-nachislen_n-vozvrat-uploch_p)>=(%s+shtraf) then (%s+shtraf) when (-1)*(%s+uploch_n+umenshen_n-nachislen_n-vozvrat-uploch_p)>0 and (-1)*(%s+uploch_n+umenshen_n-nachislen_n-vozvrat-uploch_p)<(%s+shtraf) then (-1)*(%s+uploch_n+umenshen_n-nachislen_n-vozvrat-uploch_p) else 0 end where last_date_srok=%s and na2_code=%s and tin=%s", (saldo_all, saldo_all, last_shtraf, last_shtraf, saldo_all, saldo_all, last_shtraf, saldo_all, current_date, na2_code1, tin, ))
                                cursor.execute("select sum(shtraf1) as sum_shtraf from nla where last_date_srok=%s and na2_code=%s and tin=%s", (current_date,na2_code1,tin, ))
                                sum_shtraf = cursor.fetchone()[0]
                                cursor.execute("select sum(shtraf1) as sum_shtraf from nla where last_date_srok=%s and tin=%s", (current_date,tin, ))
                                sum_shtraf_all = cursor.fetchone()[0]
                                cursor.execute("UPDATE nla SET sum_p_all = if((saldo_all_codes+%s)<0,(-1*(saldo_all_codes+%s)*(%s)*(%s)*(%s)),0) where last_date_srok=%s and na2_code=%s and tin=%s", (sum_shtraf_all, sum_shtraf_all, foiz,ulush,farq,current_date, na2_code1, tin, ))
                                cursor.execute("UPDATE nla SET sum_p = if((oldingi_kun_saldo+%s)<0,(((-1*(oldingi_kun_saldo+%s))*(-1)/%s)*sum_p_all+ifnull(other_penya,0)),ifnull(pr_penya,0)) where last_date_srok=%s and tin=%s", (sum_shtraf, sum_shtraf, last_neds_sum, current_date, tin, ))
                                cursor.execute("UPDATE nla SET saldo_tek_p = (%s+sum_p) WHERE last_date_srok = %s AND na2_code = %s and tin=%s", (saldo_tek_p, current_date, na2_code1, tin))
                                cursor.execute("UPDATE nla SET uploch_p = 0 WHERE last_date_srok = %s AND na2_code = %s and tin=%s", (current_date, na2_code1, tin))
                                # cursor.execute("UPDATE nla SET uploch_p = if((%s+uploch_n+umenshen_n-nachislen_n-vozvrat)>=saldo_tek_p,saldo_tek_p,(%s+uploch_n+umenshen_n-nachislen_n-vozvrat)), saldo_tek_p=if(uploch_p=%s+sum_p,0,%s+sum_p-uploch_p) WHERE (%s+uploch_n+umenshen_n-nachislen_n-vozvrat-uploch_p)>=0 and last_date_srok = %s AND na2_code = %s and tin=%s", (saldo_all, saldo_all, saldo_tek_p, saldo_tek_p, saldo_all, current_date, na2_code1, tin, ))
                                cursor.execute("UPDATE nla SET uploch_p = case when (%s+uploch_n+umenshen_n-nachislen_n-vozvrat)>=saldo_tek_p and saldo_tek_p>0 then saldo_tek_p when (%s+uploch_n+umenshen_n-nachislen_n-vozvrat)>0 and (%s+uploch_n+umenshen_n-nachislen_n-vozvrat)<saldo_tek_p then (%s+uploch_n+umenshen_n-nachislen_n-vozvrat) when saldo_tek_p<0 then saldo_tek_p else 0 end, saldo_tek_p=if(uploch_p>=%s+sum_p,0,%s+sum_p-uploch_p) WHERE (%s+uploch_n+umenshen_n-nachislen_n-vozvrat-uploch_p)>=0 and last_date_srok = %s AND na2_code = %s and tin=%s", (saldo_all, saldo_all, saldo_all, saldo_all, saldo_tek_p, saldo_tek_p, saldo_all, current_date, na2_code1, tin, ))
                                # cursor.execute("UPDATE nla SET datetime = %s where last_date_srok=%s and na2_code=%s", (now, current_date,na2_code1, ))
                                cursor.execute("UPDATE nla SET saldo_all = if(date_srok=last_date_srok,(uploch_n+umenshen_n-nachislen_n-vozvrat-uploch_p),(%s+uploch_n+umenshen_n-nachislen_n-vozvrat-uploch_p)) WHERE last_date_srok = %s AND na2_code = %s and tin=%s", (saldo_all, current_date, na2_code1, tin, ))
                                cursor.execute("UPDATE nla SET last_sum_neds = %s where last_date_srok=%s and tin=%s", (neds, current_date, tin, ))
                                # cursor.execute("UPDATE nla SET pereplata = if(saldo_all>0,saldo_all,0), saldo_tek_n = if(saldo_all<0,(-1*saldo_all),0)")
                                cursor.execute("delete from loop_progress where last_date_srok!=%s and na2_code!=%s and tin=%s", (current_date, na2_code1, tin,))
                                cursor.execute("INSERT INTO loop_progress (tin, last_date_srok, na2_code, datetime) VALUES (%s, %s, %s, %s)", (tin, current_date, na2_code1,now.strftime('%Y-%m-%d %H:%M:%S')))
                                dbs.commit()
                    for na2_code1 in na2_codes:
                        now = datetime.now(tz=timezone(timedelta(hours=5)))
                        # Find the corresponding saldo_all value for the current date and na2_code
                        cursor.execute("SELECT IFNULL(saldo_all, 0) FROM nla WHERE date_srok = %s and na2_code=%s and tin=%s", (current_date, na2_code1, tin, ))
                        saldo_all = cursor.fetchone()[0]
                        cursor.execute("select sum(shtraf1) as summa from nla where date_srok=%s and shtraf1<>0 and na2_code=%s and tin=%s;", (current_date,na2_code1, tin, ))
                        last_shtraf = cursor.fetchone()[0]
                        cursor.execute("SELECT IFNULL(saldo_tek_p, 0) FROM nla WHERE date_srok = %s and na2_code=%s and tin=%s", (current_date, na2_code1, tin, ))
                        saldo_tek_p = cursor.fetchone()[0]
                        cursor.execute("UPDATE nla SET oldingi_kun_saldo = %s WHERE last_date_srok = %s AND na2_code = %s and tin=%s", (saldo_all, current_date, na2_code1, tin, ))
                        cursor.execute("select DATEDIFF(date_srok,last_date_srok) as farq from nla where last_date_srok=%s and na2_code=%s and tin=%s", (current_date,na2_code1,tin, ))
                        farq1 = cursor.fetchone()
                        if farq1 is not None:
                            farq = farq1[0]
                        else:
                            farq = 0
                        cursor.execute("UPDATE nla SET shtraf1 = case when (-1)*(%s+uploch_n+umenshen_n-nachislen_n-vozvrat-uploch_p)>0 and (-1)*(%s+uploch_n+umenshen_n-nachislen_n-vozvrat-uploch_p)>=(%s+shtraf) then (%s+shtraf) when (-1)*(%s+uploch_n+umenshen_n-nachislen_n-vozvrat-uploch_p)>0 and (-1)*(%s+uploch_n+umenshen_n-nachislen_n-vozvrat-uploch_p)<(%s+shtraf) then (-1)*(%s+uploch_n+umenshen_n-nachislen_n-vozvrat-uploch_p) else 0 end where last_date_srok=%s and na2_code=%s and tin=%s", (saldo_all, saldo_all, last_shtraf, last_shtraf, saldo_all, saldo_all, last_shtraf, saldo_all, current_date, na2_code1, tin, ))
                        cursor.execute("select sum(shtraf1) as sum_shtraf from nla where last_date_srok=%s and na2_code=%s and tin=%s", (current_date,na2_code1,tin, ))
                        sum_shtraf = cursor.fetchone()[0]
                        cursor.execute("select sum(shtraf1) as sum_shtraf from nla where last_date_srok=%s and tin=%s", (current_date,tin, ))
                        sum_shtraf_all = cursor.fetchone()[0]
                        cursor.execute("UPDATE nla SET sum_p_all = if((saldo_all_codes+%s)<0,(-1*(saldo_all_codes+%s)*(%s)*(%s)*(%s)),0) where last_date_srok=%s and na2_code=%s and tin=%s", (sum_shtraf_all, sum_shtraf_all, foiz,ulush,farq,current_date, na2_code1, tin, ))
                        cursor.execute("SELECT date_srok from nla where last_date_srok=%s and na2_code=%s and tin=%s", (current_date, na2_code1, tin, ))
                        penya_write_date = cursor.fetchone()[0]
                        # cursor.execute("UPDATE nla SET sum_p_all = if((saldo_all_codes+%s)<0,(-1*(saldo_all_codes+%s)*(%s)*(%s)*(%s)),0) where last_date_srok=%s and na2_code=%s and tin=%s", (sum_shtraf_all, sum_shtraf_all, foiz,ulush,farq,current_date, na2_code1, tin, ))
                        if penya_write_date in xar_1_dates:
                            cursor.execute("UPDATE nla SET sum_p = '0' where last_date_srok=%s and tin=%s and na2_code=%s and date_srok=%s", (current_date, tin, na2_code1, penya_write_date, ))
                        else:
                            cursor.execute("UPDATE nla SET sum_p = if((oldingi_kun_saldo+%s)<0,(((-1*(oldingi_kun_saldo+%s))*(-1)/%s)*sum_p_all+ifnull(other_penya,0)),ifnull(pr_penya,0)) where last_date_srok=%s and tin=%s and na2_code=%s", (sum_shtraf, sum_shtraf, last_neds_sum, current_date, tin, na2_code1, ))
                        # cursor.execute("UPDATE nla SET sum_p = if((oldingi_kun_saldo+%s)<0,(((-1*(oldingi_kun_saldo+%s))*(-1)/%s)*sum_p_all+ifnull(other_penya,0)),ifnull(pr_penya,0)) where last_date_srok=%s and tin=%s", (sum_shtraf, sum_shtraf, last_neds_sum, current_date, tin, ))
                        cursor.execute("UPDATE nla SET saldo_tek_p = (%s+sum_p) WHERE last_date_srok = %s AND na2_code = %s and tin=%s", (saldo_tek_p, current_date, na2_code1, tin))
                        cursor.execute("UPDATE nla SET uploch_p = 0 WHERE last_date_srok = %s AND na2_code = %s and tin=%s", (current_date, na2_code1, tin))
                        # cursor.execute("UPDATE nla SET uploch_p = if((%s+uploch_n+umenshen_n-nachislen_n-vozvrat)>=saldo_tek_p,saldo_tek_p,(%s+uploch_n+umenshen_n-nachislen_n-vozvrat)), saldo_tek_p=if(uploch_p=%s+sum_p,0,%s+sum_p-uploch_p) WHERE (%s+uploch_n+umenshen_n-nachislen_n-vozvrat-uploch_p)>=0 and last_date_srok = %s AND na2_code = %s and tin=%s", (saldo_all, saldo_all, saldo_tek_p, saldo_tek_p, saldo_all, current_date, na2_code1, tin, ))
                        cursor.execute("UPDATE nla SET uploch_p = case when (%s+uploch_n+umenshen_n-nachislen_n-vozvrat)>=saldo_tek_p and saldo_tek_p>0 then saldo_tek_p when (%s+uploch_n+umenshen_n-nachislen_n-vozvrat)>0 and (%s+uploch_n+umenshen_n-nachislen_n-vozvrat)<saldo_tek_p then (%s+uploch_n+umenshen_n-nachislen_n-vozvrat) when saldo_tek_p<0 then saldo_tek_p else 0 end, saldo_tek_p=if(uploch_p>=%s+sum_p,0,%s+sum_p-uploch_p) WHERE (%s+uploch_n+umenshen_n-nachislen_n-vozvrat-uploch_p)>=0 and last_date_srok = %s AND na2_code = %s and tin=%s", (saldo_all, saldo_all, saldo_all, saldo_all, saldo_tek_p, saldo_tek_p, saldo_all, current_date, na2_code1, tin, ))
                        # cursor.execute("UPDATE nla SET datetime = %s where last_date_srok=%s and na2_code=%s", (now, current_date,na2_code1, ))
                        cursor.execute("UPDATE nla SET saldo_all = if(date_srok=last_date_srok,(uploch_n+umenshen_n-nachislen_n-vozvrat-uploch_p),(%s+uploch_n+umenshen_n-nachislen_n-vozvrat-uploch_p)) WHERE last_date_srok = %s AND na2_code = %s and tin=%s", (saldo_all, current_date, na2_code1, tin, ))
                        cursor.execute("UPDATE nla SET last_sum_neds = %s where last_date_srok=%s and tin=%s", (neds, current_date, tin, ))
                        # cursor.execute("UPDATE nla SET pereplata = if(saldo_all>0,saldo_all,0), saldo_tek_n = if(saldo_all<0,(-1*saldo_all),0)")
                        cursor.execute("delete from loop_progress where last_date_srok!=%s and na2_code!=%s and tin=%s", (current_date, na2_code1, tin,))
                        cursor.execute("INSERT INTO loop_progress (tin, last_date_srok, na2_code, datetime) VALUES (%s, %s, %s, %s)", (tin, current_date, na2_code1,now.strftime('%Y-%m-%d %H:%M:%S')))
                    dbs.commit()
                # 2020dan keyin
                elif current_date > simple_year:
                    if current_date in pr_dates:
                        cursor.execute("select max(last_date_srok) from nlb where pr_saldo_all_codes<>0 and tin=%s", (tin, ))
                        max_last_pr_date = cursor.fetchone()[0]
                        cursor.execute("select max(last_date_srok) from nlb where tin=%s", (tin, ))
                        max_last_pr_dates = cursor.fetchone()[0]
                        cursor.execute("SELECT DISTINCT na2_code FROM nla where tin=%s and last_date_srok=%s GROUP BY na2_code ORDER BY na2_code", (tin, current_date, ))
                        na2_codes = [row[0] for row in cursor.fetchall()]
                        cursor.execute("select distinct(na2_code) from pereraschet where date_srok=%s and tin=%s", (current_date, tin, ))
                        na2_codes2 = [row[0] for row in cursor.fetchall()]
                        cursor.execute("delete from nlb where tin=%s", (tin, ))
                        cursor.execute("select min(date) from start where date_srok=%s and tin=%s",(current_date,tin, ))
                        min_pr_date = cursor.fetchone()[0]
                        if min_pr_date<=simple_year:
                            cursor.execute("insert into nlb (tin, na2_code, date_srok, saldo_all, shtraf, oldingi_kun_saldo, saldo_all_codes,sum_p, saldo_tek_p, last_date_srok) select tin, na2_code, date_srok, saldo_all, ifnull(shtraf,0), oldingi_kun_saldo, saldo_all_codes, sum_p, saldo_tek_p, last_date_srok from nla where date_srok<=%s and last_date_srok>=%s  and tin=%s order by date_srok, na2_code;", (current_date, min_pr_date, tin, ))
                            dbs.commit()
                        else:
                            for na2_code in na2_codes2:
                                cursor.execute("insert into nlb (tin, na2_code, date_srok, saldo_all, shtraf, oldingi_kun_saldo, saldo_all_codes,sum_p, saldo_tek_p, last_date_srok) select tin, na2_code, date_srok, saldo_all, ifnull(shtraf,0), oldingi_kun_saldo, saldo_all_codes, sum_p, saldo_tek_p, last_date_srok from nla where date_srok<=%s and date_srok>=%s and na2_code in (%s) and tin=%s order by date_srok, na2_code;", (current_date, min_pr_date, na2_code, tin, ))
                                dbs.commit()
                        cursor.execute("select last_date_srok from nlb where tin=%s group by last_date_srok order by last_date_srok", (tin, ))
                        nlb_dates1 = [row[0] for row in cursor.fetchall()]
                        for nlb_date1 in nlb_dates1:
                            if nlb_date1<=simple_year:
                                cursor.execute("SELECT DISTINCT na2_code FROM nla where tin=%s GROUP BY na2_code ORDER BY na2_code", (tin, ))
                                na2_codes1 = [row[0] for row in cursor.fetchall()]
                                cursor.execute("select (procent/koef) from ulush where data=%s", (nlb_date1,))
                                foiz = cursor.fetchone()[0]
                                cursor.execute("select last_date_srok from nlb where last_date_srok=%s and tin=%s group by last_date_srok", (nlb_date1, tin, ))
                                last_date_srok_p = cursor.fetchone()
                                cursor.execute("select ulush as ulush from ulush where data=ifnull(%s,%s)", (nlb_date1, last_date_srok_p, ))
                                ulush = cursor.fetchone()[0]
                                cursor.execute("select sum(shtraf) from nlb where date_srok=%s and tin=%s", (nlb_date1, tin, ))
                                shtraf_all = cursor.fetchone()[0]
                                cursor.execute("SELECT sum(ifnull(nachislen_n-umenshen_n,0)) as sums_all FROM pereraschet WHERE date_real<= %s and date_srok>=%s and date_srok<=%s and date_srok>date_real and tin=%s", (nlb_date1, nlb_date1, current_date, tin, ))
                                pr_sum_all = cursor.fetchone()[0]
                                cursor.execute("UPDATE nlb SET pr_saldo_all_codes=(ifnull(%s,0)-ifnull(saldo_all_codes,0)+ifnull(%s,0)) where last_date_srok=%s and tin=%s", (pr_sum_all, shtraf_all, nlb_date1, tin, ))
                                cursor.execute("select max(last_date_srok) from nlb where tin=%s", (tin, ))
                                for na2_code1 in na2_codes1:
                                    now = datetime.now(tz=timezone(timedelta(hours=5)))
                                    cursor.execute("select DATEDIFF(date_srok,last_date_srok) as farq from nlb where last_date_srok=%s and na2_code=%s and tin=%s", (nlb_date1,na2_code1, tin, ))
                                    farq = cursor.fetchone()
                                    cursor.execute("SELECT sum(nachislen_n-umenshen_n) as sums FROM pereraschet WHERE  date_srok<=%s and date_real<= %s and date_srok>=%s and na2_code=%s and date_srok>date_real and tin=%s", (current_date, nlb_date1, nlb_date1, na2_code1, tin, ))
                                    pr_sum = cursor.fetchone()[0]
                                    cursor.execute("UPDATE nlb SET pr_saldo_all=(ifnull(%s,0)+ifnull(shtraf,0)-oldingi_kun_saldo) where last_date_srok=%s and na2_code=%s and tin=%s", (pr_sum, nlb_date1, na2_code1, tin, ))
                                    cursor.execute("SELECT sum(pr_saldo_all) as sums FROM nlb WHERE pr_saldo_all>0 and last_date_srok=%s and tin=%s", (nlb_date1, tin, ))
                                    pr_ned_sum = cursor.fetchone()[0]
                                    cursor.execute("UPDATE nlb SET pr_penya_all = if(pr_saldo_all_codes>0,(pr_saldo_all_codes*(%s)*(%s)*(%s)),0) where last_date_srok=%s and na2_code=%s and tin=%s", (foiz,ulush,farq,nlb_date1, na2_code1, tin, ))
                                    # cursor.execute("UPDATE nlb SET pr_penya_all = if((saldo_all_codes-ifnull(%s,0))<0,((-1)*(saldo_all_codes-ifnull(%s,0))*(%s)*(%s)*(%s)),0) where last_date_srok=%s and na2_code=%s", (pr_sum_all, pr_sum_all, foiz,ulush,farq,nlb_date, na2_code1,))
                                    cursor.execute("SELECT date_srok from nlb where last_date_srok=%s and na2_code=%s and tin=%s", (nlb_date1, na2_code1, tin, ))
                                    penya_write_date1 = cursor.fetchone()
                                    if penya_write_date1 is not None:
                                        penya_write_date = penya_write_date1[0]
                                        if penya_write_date in xar_1_dates:
                                            cursor.execute("UPDATE nlb SET pr_penya = '0' where last_date_srok=%s and tin=%s and na2_code=%s and date_srok=%s", (current_date, tin, na2_code1, penya_write_date, ))
                                        else:
                                            cursor.execute("UPDATE nlb SET pr_penya = if((ifnull(pr_saldo_all,0))>0,((ifnull(pr_saldo_all,0))/%s)*pr_penya_all,0) where last_date_srok=%s and tin=%s", (pr_ned_sum, nlb_date1, tin, ))
                                    else:
                                        cursor.execute("UPDATE nlb SET pr_penya = if((ifnull(pr_saldo_all,0))>0,((ifnull(pr_saldo_all,0))/%s)*pr_penya_all,0) where last_date_srok=%s and tin=%s", (pr_ned_sum, nlb_date1, tin, ))
                                    # cursor.execute("UPDATE nlb SET pr_penya = if((ifnull(pr_saldo_all,0))>0,((ifnull(pr_saldo_all,0))/%s)*pr_penya_all,0) where last_date_srok=%s and tin=%s", (pr_ned_sum, nlb_date1, tin, ))
                                    cursor.execute("delete from pr_progress where pr_date!=%s and tin=%s", (nlb_date1, tin, ))
                                    cursor.execute("INSERT INTO pr_progress (tin, pr_date, na2_code, datetime) VALUES (%s, %s, %s, %s)", (tin, nlb_date1, na2_code1,now.strftime('%Y-%m-%d %H:%M:%S')))

                                dbs.commit()
                            else:
                                cursor.execute("select (procent/koef) from ulush where data=%s", (nlb_date1,))
                                foiz = cursor.fetchone()[0]
                                cursor.execute("select last_date_srok from nlb where last_date_srok=%s and tin=%s group by last_date_srok", (nlb_date1, tin, ))
                                last_date_srok_p = cursor.fetchone()
                                cursor.execute("select ulush from ulush where data=ifnull(%s,%s)", (nlb_date1, last_date_srok_p, ))
                                ulush = cursor.fetchone()[0]
                                cursor.execute("select sum(shtraf) from nlb where date_srok=%s and tin=%s", (nlb_date1, tin, ))
                                shtraf_all = cursor.fetchone()[0]
                                cursor.execute("SELECT sum(ifnull(nachislen_n-umenshen_n,0)) as sums_all FROM pereraschet WHERE date_real<= %s and date_srok>=%s and date_srok<=%s and date_srok>date_real and tin=%s", (nlb_date1, nlb_date1, current_date, tin, ))
                                pr_sum_all = cursor.fetchone()[0]
                                # cursor.execute("UPDATE nlb SET pr_saldo_all_codes=(ifnull(%s,0)-ifnull(saldo_all_codes,0)+ifnull(%s,0)) where last_date_srok=%s and tin=%s", (pr_sum_all, shtraf_all, nlb_date1, tin, ))
                                cursor.execute("select max(last_date_srok) from nlb where tin=%s", (tin, ))
                                cursor.execute("SELECT DISTINCT na2_code FROM nlb where tin=%s GROUP BY na2_code ORDER BY na2_code", (tin, ))
                                na2_codess = [row[0] for row in cursor.fetchall()]
                                for na2_code1 in na2_codess:
                                    now = datetime.now(tz=timezone(timedelta(hours=5)))
                                    cursor.execute("select DATEDIFF(date_srok,last_date_srok) as farq from nlb where date_srok=%s and na2_code=%s and tin=%s", (nlb_date1,na2_code1, tin, ))
                                    farq = cursor.fetchone()
                                    cursor.execute("SELECT sum(nachislen_n-umenshen_n) as sums FROM pereraschet WHERE  date_srok<=%s and date_real<= %s and date_srok>%s and na2_code=%s and date_srok>date_real and tin=%s", (current_date, nlb_date1, nlb_date1, na2_code1, tin, ))
                                    pr_sum = cursor.fetchone()[0]
                                    cursor.execute("UPDATE nlb SET pr_saldo_all=(ifnull(%s,0)+ifnull(shtraf,0)-oldingi_kun_saldo) where last_date_srok=%s and na2_code=%s and tin=%s", (pr_sum, nlb_date1, na2_code1, tin, ))
                                    dbs.commit()
                                    cursor.execute("SELECT sum(pr_saldo_all) as sums FROM nlb WHERE pr_saldo_all>0 and date_srok=%s and tin=%s", (nlb_date1, tin, ))
                                    pr_ned_sum = cursor.fetchone()[0]
                                    cursor.execute("SELECT date_srok from nlb where date_srok=%s and na2_code=%s and tin=%s", (nlb_date1, na2_code1, tin, ))
                                    penya_write_date1 = cursor.fetchone()
                                    if penya_write_date1 is not None:
                                        penya_write_date = penya_write_date1[0]
                                        if penya_write_date in xar_1_dates:
                                            cursor.execute("UPDATE nlb SET pr_penya = 0 where date_srok=%s and na2_code=%s and tin=%s", (penya_write_date, na2_code1, tin, ))
                                        else:
                                            cursor.execute("UPDATE nlb SET pr_penya = if((ifnull(pr_saldo_all,0))>0,((ifnull(pr_saldo_all,0))*(%s)*(%s)*(%s)),0) where date_srok=%s and na2_code=%s and tin=%s", (foiz,ulush,farq,nlb_date1, na2_code1, tin, ))
                                    else:
                                        cursor.execute("UPDATE nlb SET pr_penya = if((ifnull(pr_saldo_all,0))>0,((ifnull(pr_saldo_all,0))*(%s)*(%s)*(%s)),0) where date_srok=%s and na2_code=%s and tin=%s", (foiz,ulush,farq,nlb_date1, na2_code1, tin, ))
                                    # cursor.execute("UPDATE nlb SET pr_penya = if((ifnull(pr_saldo_all,0))>0,((ifnull(pr_saldo_all,0))*(%s)*(%s)*(%s)),0) where date_srok=%s and na2_code=%s and tin=%s", (foiz,ulush,farq,nlb_date1, na2_code1, tin, ))
                                    cursor.execute("delete from pr_progress where pr_date!=%s and tin=%s", (nlb_date1, tin, ))
                                    cursor.execute("INSERT INTO pr_progress (tin, pr_date, na2_code, datetime) VALUES (%s, %s, %s, %s)", (tin, nlb_date1, na2_code1,now.strftime('%Y-%m-%d %H:%M:%S')))
                                    if nlb_date1==current_date:
                                        cursor.execute("UPDATE nlb SET sum_p = if((ifnull(oldingi_kun_saldo,0))<0,(-1*(ifnull(oldingi_kun_saldo,0))*(%s)*(%s)*(%s)),0) where date_srok=%s and na2_code=%s and tin=%s", (foiz,ulush,farq,nlb_date1, na2_code1, tin, ))
                                    dbs.commit()
                        cursor.execute("update nlb set pr_penya=sum_p where date_srok>'2020-01-01' and pr_penya is null and tin=%s", (tin, ))
                        cursor.execute("select max(date_srok) from nlb where tin=%s", (tin, ))
                        last_pr_date1 = cursor.fetchone()
                        if last_pr_date1 is not None:
                            last_pr_date = last_pr_date1[0]
                            if current_date == last_pr_date:
                                cursor.execute("select (procent/koef) from ulush where data=%s", (current_date,))
                                foiz = cursor.fetchone()[0]
                                cursor.execute("select ulush from ulush where data=%s", (current_date, ))
                                ulush = cursor.fetchone()[0]
                                # cursor.execute("SELECT DISTINCT na2_code FROM nla where tin=%s and date_srok=%s GROUP BY na2_code ORDER BY na2_code", (tin, current_date, ))
                                # na2_codes3 = [row[0] for row in cursor.fetchall()]
                                cursor.execute("SELECT DISTINCT na2_code FROM nlb where tin=%s GROUP BY na2_code ORDER BY na2_code", (tin, ))
                                na2_codes3 = [row[0] for row in cursor.fetchall()]
                                for na2_code1 in na2_codes3:
                                    cursor.execute("SELECT sum(nachislen_n-umenshen_n) as sums FROM pereraschet WHERE date_srok=%s and na2_code=%s and date_srok>date_real and tin=%s", (current_date, na2_code1, tin, ))
                                    pr_sum = cursor.fetchone()[0]
                                    cursor.execute("UPDATE nlb SET pr_saldo_all=(ifnull(%s,0)+ifnull(shtraf,0)-oldingi_kun_saldo) where date_srok=%s and na2_code=%s and tin=%s", (pr_sum, current_date, na2_code1, tin, ))
                                    dbs.commit()
                                    cursor.execute("select last_date_srok from nlb where date_srok=%s and tin=%s and na2_code=%s", (current_date, tin, na2_code1, ))
                                    last_d = cursor.fetchone()[0]
                                    cursor.execute("select DATEDIFF(date_srok,last_date_srok) as farq from nla where date_srok=%s and na2_code=%s and tin=%s", (current_date,na2_code1,tin,))
                                    farq = cursor.fetchone()[0]
                                    if current_date>simple_year:
                                        cursor.execute("select ifnull((pr_saldo_all*(%s)*(%s)*(%s)),0) from nlb where pr_saldo_all>0 and last_date_srok=%s and na2_code=%s and tin=%s", (foiz, ulush, farq, last_d, na2_code1, tin, ))
                                        last_sum_p2 = cursor.fetchone()
                                        if last_sum_p2 is not None:
                                            last_sum_pr = last_sum_p2[0]
                                            cursor.execute("update nlb set pr_penya=%s where last_date_srok=%s and na2_code=%s and tin=%s",(last_sum_pr, last_d, na2_code1, tin, ))
                                            dbs.commit()
                                        cursor.execute("select ifnull((saldo_all*(%s)*(%s)*(%s)*(-1)),0) from nlb where saldo_all<0 and date_srok=%s and na2_code=%s and tin=%s", (foiz, ulush, farq, last_d, na2_code1, tin, ))
                                        last_sum_p1 = cursor.fetchone()
                                        if last_sum_p1 is not None:
                                            last_sum_p = last_sum_p1[0]
                                            cursor.execute("update nlb set sum_p=%s where date_srok=%s and na2_code=%s and tin=%s",(last_sum_p, current_date, na2_code1, tin, ))
                                            dbs.commit()
                                    cursor.execute("select min(date_srok) from nlb where tin=%s", (tin, ))
                                    last_pr_date2 = cursor.fetchone()
                                    cursor.execute("SELECT DISTINCT na2_code FROM pereraschet where tin=%s and na2_code not in ('46','101','199','191') and date_srok=%s GROUP BY na2_code ORDER BY na2_code", (tin, current_date, ))
                                    na2_codes5 = [row[0] for row in cursor.fetchall()]
                                    if na2_code1 in na2_codes5:
                                        cursor.execute("select sum(pr_penya-ifnull(sum_p,0)) as sum_pr from nlb where na2_code=%s and tin=%s and date_srok>%s",(na2_code1, tin, last_pr_date2, ))
                                        sum_pr = cursor.fetchone()[0]
                                        cursor.execute("select ifnull(saldo_tek_p,0) from nlb where date_srok=%s and na2_code=%s and tin=%s", (current_date, na2_code1, tin, ))
                                        saldo_tek_p1 = cursor.fetchone()[0]
                                        cursor.execute("select ifnull(sum_p,0) from nlb where date_srok=%s and na2_code=%s and tin=%s", (current_date, na2_code1, tin, ))
                                        sum_p1 = cursor.fetchone()[0]
                                        if min_pr_date<=simple_year:
                                            cursor.execute("update nlb set nla_pr_penya=%s where date_srok=%s and na2_code=%s and tin=%s",(sum_pr, current_date, na2_code1, tin, ))
                                            cursor.execute("update nla set pr_penya=%s, saldo_tek_p=%s+%s where date_srok=%s and na2_code=%s and tin=%s", (sum_pr, saldo_tek_p1, sum_pr, current_date, na2_code1, tin, ))
                                            cursor.execute("update nla set sum_p=%s+%s where date_srok=%s and na2_code=%s and tin=%s", (sum_p1, sum_pr, current_date, na2_code1, tin, ))
                                            cursor.execute("update nla set datetime=%s where date_srok=%s and tin=%s", (now.strftime('%Y-%m-%d %H:%M:%S'), current_date, tin, ))
                                            dbs.commit()
                                        else:
                                            cursor.execute("SELECT DISTINCT na2_code FROM pereraschet where tin=%s and na2_code not in ('46','101','199','191') and date_srok=%s GROUP BY na2_code ORDER BY na2_code", (tin, current_date, ))
                                            na2_codes4 = [row[0] for row in cursor.fetchall()]
                                            for na2_code4 in na2_codes4:
                                                cursor.execute("update nlb set nla_pr_penya=%s where date_srok=%s and na2_code=%s and tin=%s",(sum_pr, current_date, na2_code1, tin, ))
                                                cursor.execute("update nla set pr_penya=%s, saldo_tek_p=%s+%s where date_srok=%s and na2_code=%s and tin=%s", (sum_pr, saldo_tek_p1, sum_pr, current_date, na2_code1, tin, ))
                                                cursor.execute("update nla set sum_p=%s+%s where date_srok=%s and na2_code=%s and tin=%s", (sum_p1, sum_pr, current_date, na2_code1, tin, ))
                                                cursor.execute("update nla set datetime=%s where date_srok=%s and tin=%s", (now.strftime('%Y-%m-%d %H:%M:%S'), current_date, tin, ))
                                            dbs.commit()
                                dbs.commit()
                    cursor.execute("update nla set sum_p=0 where date_srok=%s and tin=%s", (current_date,tin, ))
                    cursor.execute("SELECT DISTINCT na2_code FROM nla where tin=%s and date_srok=%s GROUP BY na2_code ORDER BY na2_code", (tin, current_date, ))
                    na2_codess = [row[0] for row in cursor.fetchall()]
                    for na2_code1 in na2_codess:
                        now = datetime.now(tz=timezone(timedelta(hours=5)))
                        cursor.execute("SELECT last_date_srok FROM nla WHERE date_srok = %s and na2_code=%s and tin=%s", (current_date, na2_code1, tin, ))
                        last_date_od = cursor.fetchone()[0]
                        cursor.execute("SELECT CASE WHEN last_date_srok < '2020-01-01' THEN '2020-01-01' ELSE last_date_srok END FROM nla WHERE date_srok = %s and na2_code=%s and tin=%s", (current_date, na2_code1, tin, ))
                        last_date_od1 = cursor.fetchone()[0]
                        cursor.execute("SELECT IFNULL(saldo_all, 0) FROM nla WHERE date_srok = %s and na2_code=%s and tin=%s", (last_date_od, na2_code1, tin,))
                        saldo_all = cursor.fetchone()[0]
                        cursor.execute("select sum(shtraf1) as summa from nla where date_srok=%s and shtraf1<>0 and na2_code=%s and tin=%s;", (last_date_od,na2_code1, tin, ))
                        last_shtraf = cursor.fetchone()[0]
                        cursor.execute("SELECT IFNULL(saldo_tek_p, 0) FROM nla WHERE date_srok = %s and na2_code=%s and tin=%s", (last_date_od, na2_code1, tin, ))
                        saldo_tek_p = cursor.fetchone()[0]
                        cursor.execute("UPDATE nla SET oldingi_kun_saldo = %s WHERE date_srok = %s AND na2_code = %s and tin=%s", (saldo_all, current_date, na2_code1, tin,))
                        cursor.execute("select DATEDIFF(date_srok,%s) as farq from nla where date_srok=%s and na2_code=%s and tin=%s", (last_date_od1, current_date,na2_code1,tin,))
                        farq = cursor.fetchone()[0]
                        cursor.execute("UPDATE nla SET shtraf1 = case when (-1)*(%s+uploch_n+umenshen_n-nachislen_n-vozvrat-uploch_p)>0 and (-1)*(%s+uploch_n+umenshen_n-nachislen_n-vozvrat-uploch_p)>=(%s+shtraf) then (%s+shtraf) when (-1)*(%s+uploch_n+umenshen_n-nachislen_n-vozvrat-uploch_p)>0 and (-1)*(%s+uploch_n+umenshen_n-nachislen_n-vozvrat-uploch_p)<(%s+shtraf) then (-1)*(%s+uploch_n+umenshen_n-nachislen_n-vozvrat-uploch_p) else 0 end where date_srok=%s and na2_code=%s and tin=%s", (saldo_all, saldo_all, last_shtraf, last_shtraf, saldo_all, saldo_all, last_shtraf, saldo_all, current_date, na2_code1, tin, ))
                        cursor.execute("select sum(ifnull(shtraf1,0)) as sum_shtraf from nla where date_srok=%s and na2_code=%s and tin=%s", (last_date_od,na2_code1,tin, ))
                        sum_shtraf = cursor.fetchone()[0]
                        cursor.execute("select sum(ifnull(shtraf1,0)) as sum_shtraf from nla where date_srok=%s and tin=%s", (last_date_od, tin, ))
                        sum_shtraf_all = cursor.fetchone()[0]
                        cursor.execute("UPDATE nla SET oldingi_kun_ned = if(oldingi_kun_saldo<0,(-1*oldingi_kun_saldo),0) where date_srok=%s and na2_code=%s and tin=%s", (current_date,na2_code1,tin, ))
                        cursor.execute("select sum(ifnull(shtraf1,0)) as sum_shtraf from nla where date_srok=%s and na2_code=%s and tin=%s", (last_date_od,na2_code1,tin, ))
                        sum_shtraf = cursor.fetchone()[0]
                        cursor.execute("SELECT IFNULL((-1*(oldingi_kun_saldo+%s)*(%s)*(%s)*(%s)), 0) FROM nla WHERE date_srok = %s and ((-1)*oldingi_kun_ned+%s)<0 and na2_code=%s and tin=%s", (sum_shtraf, foiz,ulush,farq,current_date, sum_shtraf, na2_code1,tin,))
                        sum_p1 = cursor.fetchone()
                        if sum_p1 is not None:
                            sum_p = sum_p1[0]
                        else:
                            sum_p = 0
                        cursor.execute("SELECT date_srok from nla where date_srok=%s and na2_code=%s and tin=%s", (current_date, na2_code1, tin, ))
                        penya_write_date = cursor.fetchone()[0]
                        if penya_write_date in xar_1_dates:
                            cursor.execute("UPDATE nla SET sum_p = '0' where date_srok=%s and tin=%s and na2_code=%s", (penya_write_date, tin, na2_code1, ))
                        else:
                            cursor.execute("UPDATE nla SET sum_p = if(((-1)*oldingi_kun_ned+%s)<0,(-1*(oldingi_kun_saldo+%s)*(%s)*(%s)*(%s)+ifnull(pr_penya,0)),ifnull(pr_penya,0)) where date_srok=%s and na2_code=%s and tin=%s", (sum_shtraf, sum_shtraf, foiz,ulush,farq,current_date, na2_code1,tin,))
                        # cursor.execute("UPDATE nla SET sum_p = if(((-1)*oldingi_kun_ned+%s)<0,(-1*(oldingi_kun_saldo+%s)*(%s)*(%s)*(%s)+ifnull(pr_penya,0)),ifnull(pr_penya,0)) where date_srok=%s and na2_code=%s and tin=%s", (sum_shtraf, sum_shtraf, foiz,ulush,farq,current_date, na2_code1,tin,))
                        cursor.execute("UPDATE nla SET uploch_p = 0 WHERE date_srok = %s AND na2_code = %s and tin=%s", (current_date, na2_code1, tin))
                        # cursor.execute("UPDATE nla SET uploch_p = if((%s+uploch_n+umenshen_n-nachislen_n-vozvrat)>=saldo_tek_p,saldo_tek_p,(%s+uploch_n+umenshen_n-nachislen_n-vozvrat)), saldo_tek_p=if(uploch_p=%s+sum_p,0,%s+sum_p-uploch_p) WHERE (%s+uploch_n+umenshen_n-nachislen_n-vozvrat-uploch_p)>=0 and last_date_srok = %s AND na2_code = %s and tin=%s", (saldo_all, saldo_all, saldo_tek_p, saldo_tek_p, saldo_all, current_date, na2_code1, tin, ))
                        # cursor.execute("UPDATE nla SET uploch_p = if((%s+uploch_n+umenshen_n-nachislen_n-vozvrat)>=saldo_tek_p,saldo_tek_p,(%s+uploch_n+umenshen_n-nachislen_n-vozvrat)), saldo_tek_p=if(uploch_p=%s+sum_p,0,%s+sum_p-uploch_p) WHERE (%s+uploch_n+umenshen_n-nachislen_n-vozvrat-uploch_p)>=0 and date_srok = %s AND na2_code = %s", (saldo_all, saldo_all, saldo_tek_p, saldo_tek_p, saldo_all, current_date, na2_code1))
                        cursor.execute("UPDATE nla SET saldo_tek_p = (%s+%s+ifnull(pr_penya,0)) WHERE date_srok = %s AND na2_code = %s and tin=%s", (saldo_tek_p, sum_p, current_date, na2_code1, tin,))
                        dbs.commit()
                        cursor.execute("UPDATE nla SET uploch_p = case when (%s+uploch_n+umenshen_n-nachislen_n-vozvrat)>=saldo_tek_p and saldo_tek_p>0 then saldo_tek_p when (%s+uploch_n+umenshen_n-nachislen_n-vozvrat)>0 and (%s+uploch_n+umenshen_n-nachislen_n-vozvrat)<saldo_tek_p then (%s+uploch_n+umenshen_n-nachislen_n-vozvrat) when saldo_tek_p<0 then saldo_tek_p else 0 end, saldo_tek_p=if(uploch_p>=%s+sum_p,0,%s+sum_p-uploch_p) WHERE (%s+uploch_n+umenshen_n-nachislen_n-vozvrat-uploch_p)>=0 and date_srok = %s AND na2_code = %s and tin=%s", (saldo_all, saldo_all, saldo_all, saldo_all, saldo_tek_p, saldo_tek_p, saldo_all, current_date, na2_code1, tin,))
                        # cursor.execute("UPDATE nla SET uploch_p = if((%s+uploch_n+umenshen_n-nachislen_n-vozvrat)>=saldo_tek_p,saldo_tek_p,(%s+uploch_n+umenshen_n-nachislen_n-vozvrat)), saldo_tek_p=if(uploch_p=(%s+%s+pr_penya),0,(%s+sum_p+pr_penya-ifnull(uploch_p,0))) WHERE (%s+uploch_n+umenshen_n-nachislen_n-vozvrat-uploch_p)>=0 and date_srok = %s AND na2_code = %s", (saldo_all, saldo_all, saldo_tek_p, sum_p, saldo_tek_p, saldo_all, current_date, na2_code1))
                        cursor.execute("UPDATE nla SET saldo_all = if(date_srok=last_date_srok,(uploch_n+umenshen_n-nachislen_n-vozvrat-uploch_p),(%s+uploch_n+umenshen_n-nachislen_n-vozvrat-uploch_p)) WHERE date_srok = %s AND na2_code = %s and tin=%s", (saldo_all, current_date, na2_code1, tin, ))
                        cursor.execute("delete from loop_progress where last_date_srok!=%s and na2_code!=%s and tin=%s", (current_date, na2_code1, tin,))
                        cursor.execute("INSERT INTO loop_progress (tin, last_date_srok, na2_code, datetime) VALUES (%s, %s, %s, %s)", (tin, current_date, na2_code1,now.strftime('%Y-%m-%d %H:%M:%S')))
                    dbs.commit()
                sleep(1)  # Simulating some processing time
                processed_date = current_date.strftime('%Y-%m-%d')
                socketio.emit('progress_update', {'current_date': processed_date, 'message': 'Processing date: ' + processed_date})

            socketio.emit('progress_update', {'current_date': '', 'message': 'Process completed!'})
            # time.sleep(10)

            cursor.close()
        # sleep(1)
            return redirect(url_for("nl"))

            # schedule.every(5).minutes.do(saldo)

            # while True:
            #     schedule.run_pending()
            #     time.sleep(1)
    else:
        msgg = 'not logged in'
    # User is not loggedin redirect to login page
        return redirect(url_for("login"))
# tasks = {}

@app.route('/st')
def st():
    return render_template('st.html')

@app.route('/start_task', methods=['POST'])
def start_task():
    task_id = str(int(time.time()))
    background_thread = threading.Thread(target=saldo)
    tasks[task_id] = {
        'thread': background_thread,
        'status': 'running'
    }
    background_thread.start()
    return jsonify({'task_id': task_id})

@app.route('/get_status/<task_id>', methods=['GET'])
def get_status(task_id):
    if task_id in tasks:
        status = tasks[task_id]['status']
        return jsonify({'status': status})
    else:
        return jsonify({'status': 'not_found'})

@app.route('/status', methods=['GET'])
def get_task_status():
    # Check task status here and return status information
    return jsonify({'status': 'running'})

@app.route("/portal", methods=["POST", "GET"])
def videos():
    if request.method == "POST" and request.form["nma"] == "admin":
        #user = request.form["nm"]
        return redirect(url_for("profile"))
    else:
        return render_template("portal.html")

@app.route("/videos", methods=["POST", "GET"])
def portal():
    if request.method == "POST" and request.form["nma"] == "admin":
        #user = request.form["nm"]
        return redirect(url_for("profile"))
    else:
        return render_template("videos.html")

@app.route("/<usr>")
def user(usr):
    return f"<h1>{usr}</h1>"


@app.route("/questions", methods=['POST','GET'])
def questions():
    msg = ''
    app.config["DEBUG"] = True
    app.secret_key = "super secret key"
    user = 'hbnnarzullayev'
    password = 'Sersarson7'
    host = 'hbnnarzullayev.mysql.pythonanywhere-services.com'
    mysql = 'hbnnarzullayev$flask3'
    dbs = msd.connect(host,user,password, mysql)
    # Get the current page number from the query string parameter
    page = int(request.args.get('page', 1))

    # Number of records to display per page
    per_page = 20

    # Calculate the start and end indices of the records to fetch from the database
    start_index = (page - 1) * per_page
    end_index = start_index + per_page
    # Check if user is loggedin#
    cur = dbs.cursor()
    cur.execute(f"select * from questions")
    data=cur.fetchall()
    column_names = [desc[0] for desc in cur.description]
    cur = dbs.cursor()
    cur.execute("select * from questions LIMIT %s, %s", (start_index, per_page))
    records = cur.fetchall()
    # cur.execute("select * from answers where question_id=%s LIMIT %s, %s", (question_id, start_index, per_page))
    # answers = cur.fetchall()

# Count the total number of records in the table
    cur.execute("SELECT COUNT(*) FROM questions")
    total_records = cur.fetchone()[0]#data from database
    total_pages = total_records // per_page + (total_records % per_page > 0)
    if request.method == "POST":
        question_id=request.form.get('id1')
        cur.execute("select * from answers where question_id=%s LIMIT %s, %s", (question_id, start_index, per_page))
        answers = cur.fetchall()
        column_names_1 = [desc[0] for desc in cur.description]
        return render_template('questions.html', records=records, page=page, total_pages=total_pages, values=data, column_names=column_names, answers=answers, column_names_1=column_names_1)

    return render_template('questions.html', records=records, page=page, total_pages=total_pages, values=data, column_names=column_names)

@app.route('/question_add', methods=['POST','GET'])
def question_add():
    msg=''
    app.config["DEBUG"] = True
    app.secret_key = "789456asd"
    user = 'hbnnarzullayev'
    password = 'Sersarson7'
    host = 'hbnnarzullayev.mysql.pythonanywhere-services.com'
    mysql = 'hbnnarzullayev$flask3'
    dbs = msd.connect(host,user,password, mysql)
    if 'loggedin' in session and request.method == "POST":
        question_head = request.form.get('question_head')
        question = request.form.get('query')
        #cur = dbs.cursor()
        conn = MySQLdb.connect("hbnnarzullayev.mysql.pythonanywhere-services.com","hbnnarzullayev","Sersarson7","hbnnarzullayev$flask3" )
        cursor = conn.cursor()
        sql = "INSERT INTO questions (questioner, question_header, question) VALUES (%s, %s, %s)"
        cursor.execute(sql, (session["nma"], question_head, question))

        # Commit changes and close the connection
        conn.commit()
        cursor.close()
        conn.close()
        msg="Yuklandi!"
        return render_template("question.html", error=msg)
    else:
        return render_template("question.html", error=msg)

@app.route('/answer', methods=['POST','GET'])
def answer():
    msg=''
    app.config["DEBUG"] = True
    app.secret_key = "789456asd"
    user = 'hbnnarzullayev'
    password = 'Sersarson7'
    host = 'hbnnarzullayev.mysql.pythonanywhere-services.com'
    mysql = 'hbnnarzullayev$flask3'
    dbs = msd.connect(host,user,password, mysql)
    question_id = request.form.get('question_id')
    if 'loggedin' in session and request.method == "POST" and request.form.get('answer'):
        answer = request.form.get('answer')
        #cur = dbs.cursor()
        conn = MySQLdb.connect("hbnnarzullayev.mysql.pythonanywhere-services.com","hbnnarzullayev","Sersarson7","hbnnarzullayev$flask3" )
        cursor = conn.cursor()
        sql = "INSERT INTO answers (answerer, question_id, answer) VALUES (%s, %s, %s)"
        cursor.execute(sql, (session["nma"], question_id, answer))

        # Commit changes and close the connection
        conn.commit()
        cursor.close()
        conn.close()
        msg="Yuklandi!"
        return render_template("answer.html", error=msg)
    else:
        msg="Yuklanmadi!"
        return render_template("answer.html", error=msg)

@app.route('/quiz', methods=['POST','GET'])
def quiz():
    msg=''
    app.config["DEBUG"] = True
    app.secret_key = "789456asd"
    user = 'hbnnarzullayev'
    password = 'Sersarson7'
    host = 'hbnnarzullayev.mysql.pythonanywhere-services.com'
    mysql = 'hbnnarzullayev$flask3'
    dbs = msd.connect(host,user,password, mysql)
    question_id = request.form.get('question_id')
    now = datetime.now(tz=timezone(timedelta(hours=5)))
    if 'loggedin' in session and session['nma'] in ('admin','Admin') and request.method == "POST" and request.form.get('quiz'):
        quiz = request.form.get('quiz')
        option1 = request.form.get('option1')
        option2 = request.form.get('option2')
        option3 = request.form.get('option3')
        option4 = request.form.get('option4')
        #cur = dbs.cursor()
        conn = MySQLdb.connect("hbnnarzullayev.mysql.pythonanywhere-services.com","hbnnarzullayev","Sersarson7","hbnnarzullayev$flask3" )
        cursor = conn.cursor()
        sql = "INSERT INTO quiz (quiz, option1, option2,option3,option4, datetime) VALUES (%s, %s, %s, %s, %s, %s)"
        cursor.execute(sql, (quiz, option1, option2, option3, option4, now.strftime('%Y-%m-%d %H:%M:%S')))

        # Commit changes and close the connection
        conn.commit()
        cursor.close()
        conn.close()
        msg="Yuklandi!"
        return render_template("quiz.html", error=msg)
    else:
        msg="Yuklanmadi!"
        return render_template("quiz.html", error=msg)

@app.route('/get_videos', methods=['GET'])
def get_videos():
    print("get_videos")
    data = request.args  # Using GET parameters
    print(data)
    app_id = data.get("app_id")
    username = data.get("username")  # Assume username is passed to check if the user is an admin

    # Database connection
    user = 'hbnnarzullayev'
    password = 'Sersarson7'
    host = 'hbnnarzullayev.mysql.pythonanywhere-services.com'
    mysql_db = 'hbnnarzullayev$flask3'

    try:
        dbs = msd.connect(host=host, user=user, password=password, database=mysql_db)
        cursor = dbs.cursor()

        # Check if the user is an admin
        cursor.execute("SELECT user_role FROM users2 WHERE username = %s and app_id=%s", (username, app_id,))
        user_data = cursor.fetchone()
        print(user_data)

        if user_data:
            user_role = user_data[0]
            if user_role == 1:
                is_admin = 'true'
            else:
                is_admin = False
        else:
            is_admin = False
        print(is_admin)
        # Fetch videos for the app_id
        cursor.execute("SELECT video_url, name, app FROM youtube_videos WHERE app = %s", (app_id,))
        videos = cursor.fetchall()
        cursor.close()
        dbs.close()

        # Return the videos along with admin status
        video_list = [{'url': video[0], 'name': video[1], 'app': video[2]} for video in videos]
        print(video_list)
        return jsonify({'videos': video_list, 'is_admin': is_admin})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get_food_videos', methods=['GET'])
def get_food_videos():
    print("get_food_videos")
    data = request.args  # Using GET parameters
    print(data)
    app_id = data.get("app_id")
    username = data.get("username")
    groupName = data.get("groupName")

    # Database connection
    user = 'hbnnarzullayev'
    password = 'Sersarson7'
    host = 'hbnnarzullayev.mysql.pythonanywhere-services.com'
    mysql_db = 'hbnnarzullayev$flask3'

    try:
        dbs = msd.connect(host=host, user=user, password=password, database=mysql_db)
        cursor = dbs.cursor()
        cursor.execute("SELECT user_role FROM users2 WHERE username = %s and app_id=%s", (username, app_id,))
        user_data = cursor.fetchone()
        print(user_data)

        if user_data:
            user_role = user_data[0]
            if user_role == 1:
                is_admin = 'true'
            else:
                is_admin = False
        else:
            is_admin = False
        print(is_admin)

        # Fetch videos for the app_id
        cursor.execute("SELECT video_url, name, app, group_name, id, PKEY FROM youtube_food_videos WHERE group_name=%s", (groupName,))
        videos = cursor.fetchall()
        cursor.close()
        dbs.close()

        # Return the videos along with admin status
        video_list = [{'url': video[0], 'name': video[1], 'app': video[2], 'group_name': video[3], 'id': video[4], 'pkey': video[5]} for video in videos]
        print(video_list)
        # return jsonify(video_list)
        return jsonify({'video_list': video_list, 'is_admin': is_admin})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get_food_video', methods=['GET','Post'])
def get_food_video():
    print("get_food_video")

    # Database connection
    user = 'hbnnarzullayev'
    password = 'Sersarson7'
    host = 'hbnnarzullayev.mysql.pythonanywhere-services.com'
    mysql_db = 'hbnnarzullayev$flask3'

    try:
        with msd.connect(host=host, user=user, password=password, database=mysql_db) as dbs:
            with dbs.cursor() as cursor:
                # Fetch videos for the app_id
                cursor.execute("""
                    SELECT video_url, name, app, group_name, id, PKEY, user_id, link_type, file_id, duration, file_url, file_id_for_video
                    FROM youtube_food_videos
                """)
                videos = cursor.fetchall()

        # Format results
        video_list = [
            {
                'url': video[0], 'name': video[1], 'app': video[2], 'group_name': video[3],
                'id': video[4], 'pkey': video[5], 'user_id': video[6], 'link_type': video[7],
                'file_id': video[8], 'duration': video[9], 'file_url': video[10], 'file_id_for_video': video[11]
            }
            for video in videos
        ]

        print(video_list)
        return jsonify(video_list)

    except msd.Error as e:
        print(f"Database Error: {e}")
        return jsonify({'error': 'Database query failed'}), 500


@app.route('/add_video', methods=['POST','GET'])
def add_video():
    print("add_video")
    data = request.get_json()  # Get JSON data from request
    print(data)
    video_url = data.get('video_url')
    video_name = data.get('name')
    app_id = data.get('app_id')
    now = datetime.now(tz=timezone(timedelta(hours=5)))  # Get current time with timezone UTC+5
    formatted_time = now.strftime('%Y-%m-%d %H:%M:%S')
    print(formatted_time)
    if not video_url or not video_name:
        return jsonify({'error': 'Both video_url and name are required'}), 400

    # Database connection
    user = 'hbnnarzullayev'
    password = 'Sersarson7'
    host = 'hbnnarzullayev.mysql.pythonanywhere-services.com'
    mysql_db = 'hbnnarzullayev$flask3'

    try:
        print("TRY")
        dbs = msd.connect(host=host, user=user, password=password, database=mysql_db)
        cursor = dbs.cursor()

        # Insert video into the database
        cursor.execute("INSERT INTO youtube_videos (video_url, name, app, datetime) VALUES (%s, %s, %s, %s)", (video_url, video_name, app_id, formatted_time))
        dbs.commit()
        cursor.close()
        dbs.close()
        print("SAVED")
        return jsonify({'message': 'Video added successfully'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/add_food_video', methods=['POST','GET'])
def add_food_video():
    print("add_video")
    data = request.get_json()  # Get JSON data from request
    print(data)
    video_url = data.get('video_url')
    video_name = data.get('name')
    app_id = data.get('app_id')
    groupName = data.get("groupName")
    now = datetime.now(tz=timezone(timedelta(hours=5)))  # Get current time with timezone UTC+5
    formatted_time = now.strftime('%Y-%m-%d %H:%M:%S')
    print(formatted_time)
    if not video_url or not video_name:
        return jsonify({'error': 'Both video_url and name are required'}), 400

    # Database connection
    user = 'hbnnarzullayev'
    password = 'Sersarson7'
    host = 'hbnnarzullayev.mysql.pythonanywhere-services.com'
    mysql_db = 'hbnnarzullayev$flask3'

    try:
        print("TRY")
        dbs = msd.connect(host=host, user=user, password=password, database=mysql_db)
        cursor = dbs.cursor()
        pkey = generate_random_pkey()

        # Insert video into the database
        cursor.execute("INSERT INTO youtube_food_videos (video_url, name, app, datetime, group_name, PKEY) VALUES (%s, %s, %s, %s, %s, %s)", (video_url, video_name, app_id, formatted_time, groupName, pkey))
        dbs.commit()
        cursor.close()
        dbs.close()
        print("SAVED")
        return jsonify({'message': 'Video added successfully'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def generate_random_pkey():
    """
    Generate a random 10-character alphanumeric string.
    """
    characters = string.ascii_letters + string.digits  # Includes a-z, A-Z, 0-9
    return ''.join(random.choices(characters, k=10))

@app.route('/search_videos', methods=['GET'])
def search_videos():

    # Database connection
    user = 'hbnnarzullayev'
    password = 'Sersarson7'
    host = 'hbnnarzullayev.mysql.pythonanywhere-services.com'
    mysql_db = 'hbnnarzullayev$flask3'
    search_query = request.args.get('query', '')

    if not search_query:
        return jsonify({"error": "No search query provided"}), 400


    dbs = msd.connect(host=host, user=user, password=password, database=mysql_db)
    cursor = dbs.cursor()

    sql = "SELECT * FROM youtube_food_videos WHERE name LIKE %s"
    cursor.execute(sql, ("%" + search_query + "%",))
    results = cursor.fetchall()

    cursor.close()
    dbs.close()
    video_list = [{'url': result[0], 'name': result[1], 'app': result[2], 'group_name': result[3], 'id': result[4], 'pkey': result[5]} for result in results]
    print(video_list)
    return jsonify(video_list)

    # return jsonify(results)

@app.route('/add_quiz', methods=['POST','GET'])
def add_quiz():
    print("add_quiz")

    data = request.get_json()  # Get JSON data from request
    print(data)

    test = data.get('test')  # Assuming test is a string
    quiz = data.get('quiz')
    option1 = data.get('option1')
    option2 = data.get('option2')
    option3 = data.get('option3')
    option4 = data.get('option4')
    answer = data.get('answer')
    app_id = data.get('app_id')
    if test and '-' in test:
        test = test.split('-')[0]  # Get the part before the first '-'
        print(f"Extracted part of test: {test}")

    # Validate that the necessary fields are provided
    if not all([quiz, option1, option2, option3, option4, answer, app_id]):
        return jsonify({'error': 'All fields must be provided'}), 400

    now = datetime.now(tz=timezone(timedelta(hours=5)))  # Get current time with timezone UTC+5
    formatted_time = now.strftime('%Y-%m-%d %H:%M:%S')
    print(f"Formatted time: {formatted_time}")

    # Database connection details
    user = 'hbnnarzullayev'
    password = 'Sersarson7'
    host = 'hbnnarzullayev.mysql.pythonanywhere-services.com'
    mysql_db = 'hbnnarzullayev$flask3'

    try:
        print("Attempting to connect to the database...")
        dbs = msd.connect(host=host, user=user, password=password, database=mysql_db)
        cursor = dbs.cursor()

        # Get the maximum science number for the given teacher (app_id)
        cursor.execute("SELECT max(science) AS science FROM quizes WHERE teacher=%s;", (app_id,))
        max_sciences = cursor.fetchone()

        print(f"Max science: {max_sciences}")

        # Determine the next science number
        if max_sciences and max_sciences[0] is not None:
            max_science = int(max_sciences[0])  # Convert the string to an integer
            if test == "":  # If 'test' is empty, increment the max science value
                next_science = max_science + 1
            else:  # If a test is provided, use the max_science value (or possibly set another logic)
                next_science = test
        else:
            next_science = 1  # If no records exist, start with 1


        print(f"Next science: {next_science}")

        # Now proceed to insert the quiz into the database
        cursor.execute("""
            INSERT INTO quizes (quiz, option1, option2, option3, option4, answer, datetime, teacher, science)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (quiz, option1, option2, option3, option4, answer, formatted_time, app_id, next_science))

        # Commit the changes and close the connection
        dbs.commit()

        cursor.close()
        dbs.close()

        print("Quiz saved successfully")

        return jsonify({'message': 'Quiz added successfully'}), 200

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/add_quiz_test', methods=['POST','GET'])
def add_quiz_test():
    print("add_quiz_test")

    data = request.get_json()  # Get JSON data from request
    print(data)

    test = data.get('test')  # Assuming test is a string
    quiz = data.get('quiz')
    option1 = data.get('option1')
    option2 = data.get('option2')
    option3 = data.get('option3')
    option4 = data.get('option4')
    answer = data.get('answer')
    app_id = data.get('app_id')
    testname = data.get('testname')

    # Database connection details
    user = 'hbnnarzullayev'
    password = 'Sersarson7'
    host = 'hbnnarzullayev.mysql.pythonanywhere-services.com'
    mysql_db = 'hbnnarzullayev$flask3'
    print("Attempting to connect to the database...")
    dbs = msd.connect(host=host, user=user, password=password, database=mysql_db)
    cursor = dbs.cursor()
    if test and '-' in test:
        test = test.split('-')[0]  # Get the part before the first '-'
        print(f"Extracted part of test: {test}")
    if testname =='':
        cursor.execute("SELECT max(Name) AS Name FROM quizes WHERE teacher=%s and science=%s;", (app_id,test,))
        max_names = cursor.fetchone()
        max_name = max_names[0]
        testname = max_name

    # Validate that the necessary fields are provided
    if not all([quiz, option1, option2, option3, option4, answer, app_id, testname]):
        return jsonify({'error': 'All fields must be provided'}), 400

    now = datetime.now(tz=timezone(timedelta(hours=5)))  # Get current time with timezone UTC+5
    formatted_time = now.strftime('%Y-%m-%d %H:%M:%S')
    print(f"Formatted time: {formatted_time}")

    try:
        print("Attempting to connect to the database...")
        dbs = msd.connect(host=host, user=user, password=password, database=mysql_db)
        cursor = dbs.cursor()

        # Get the maximum science number for the given teacher (app_id)
        cursor.execute("SELECT max(science) AS science FROM quizes WHERE teacher=%s;", (app_id,))
        max_sciences = cursor.fetchone()

        print(f"Max science: {max_sciences}")

        # Determine the next science number
        if max_sciences and max_sciences[0] is not None:
            max_science = int(max_sciences[0])  # Convert the string to an integer
            if test == "":  # If 'test' is empty, increment the max science value
                next_science = max_science + 1
            else:  # If a test is provided, use the max_science value (or possibly set another logic)
                next_science = test
        else:
            next_science = 1  # If no records exist, start with 1


        print(f"Next science: {next_science}")

        # Now proceed to insert the quiz into the database
        cursor.execute("""
            INSERT INTO quizes (Name, quiz, option1, option2, option3, option4, answer, datetime, teacher, science)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (testname, quiz, option1, option2, option3, option4, answer, formatted_time, app_id, next_science))

        # Commit the changes and close the connection
        dbs.commit()

        cursor.close()
        dbs.close()

        print("Quiz saved successfully")

        return jsonify({'message': 'Quiz added successfully'}), 200

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/edit_video', methods=['PUT'])
def edit_video():
    data = request.get_json()
    video_url = data['url']
    video_name = data['name']
    user = 'hbnnarzullayev'
    password = 'Sersarson7'
    host = 'hbnnarzullayev.mysql.pythonanywhere-services.com'
    mysql_db = 'hbnnarzullayev$flask3'
    dbs = msd.connect(host=host, user=user, password=password, database=mysql_db)
    cursor = dbs.cursor()

    try:
        cursor = dbs.cursor()
        cursor.execute(
            "UPDATE youtube_videos SET name=%s, video_url=%s WHERE video_url=%s",
            (video_name, video_url, video_url)
        )
        dbs.commit()
        return jsonify({'message': 'Video updated successfully!'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/edit_food_video', methods=['PUT'])
def edit_food_video():
    print('edit_food_video')
    data = request.get_json()
    print(data)
    video_url = data['url']
    video_name = data['name']
    pkey = data['pkey']
    user = 'hbnnarzullayev'
    password = 'Sersarson7'
    host = 'hbnnarzullayev.mysql.pythonanywhere-services.com'
    mysql_db = 'hbnnarzullayev$flask3'
    dbs = msd.connect(host=host, user=user, password=password, database=mysql_db)
    cursor = dbs.cursor()

    try:
        cursor = dbs.cursor()
        cursor.execute(
            "UPDATE youtube_food_videos SET name=%s, video_url=%s WHERE pkey=%s",
            (video_name, video_url, pkey)
        )
        dbs.commit()
        return jsonify({'message': 'Video updated successfully!'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/delete_video', methods=['DELETE', 'POST', 'GET'])
def delete_video():
    print("DELETING")

    # Get video_url from the request body (JSON)
    data = request.get_json()
    video_url = data.get('video_url')
    if not video_url:
        return jsonify({'error': 'No video_url provided in the request body'}), 400

    print(f"Received video_url: {video_url}")
    user = 'hbnnarzullayev'
    password = 'Sersarson7'
    host = 'hbnnarzullayev.mysql.pythonanywhere-services.com'
    mysql_db = 'hbnnarzullayev$flask3'

    try:
        # Connect to the database
        dbs = msd.connect(host=host, user=user, password=password, database=mysql_db)
        cursor = dbs.cursor()

        # Delete the video from the database
        cursor.execute("DELETE FROM youtube_videos WHERE video_url=%s", (video_url,))
        dbs.commit()
        print("Deleted")

        return jsonify({'message': 'Video deleted successfully!'}), 200
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/delete_food_video', methods=['DELETE', 'POST', 'GET'])
def delete_food_video():
    print("DELETING_FOOD")

    # Get video_url from the request body (JSON)
    data = request.get_json()
    video_url = data.get('video_url')
    video_pkey = data.get('pkey')
    if not video_url:
        return jsonify({'error': 'No video_url provided in the request body'}), 400

    print(f"Received video_url: {video_url}")
    user = 'hbnnarzullayev'
    password = 'Sersarson7'
    host = 'hbnnarzullayev.mysql.pythonanywhere-services.com'
    mysql_db = 'hbnnarzullayev$flask3'

    try:
        # Connect to the database
        dbs = msd.connect(host=host, user=user, password=password, database=mysql_db)
        cursor = dbs.cursor()

        # Delete the video from the database
        cursor.execute("DELETE FROM youtube_food_videos WHERE pkey=%s", (video_pkey,))
        dbs.commit()
        print("Deleted")

        return jsonify({'message': 'Video deleted successfully!'}), 200
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/get_quizzes_to_dl', methods=['POST'])
def get_quizzes_to_dl():
    user = 'hbnnarzullayev'
    password = 'Sersarson7'
    host = 'hbnnarzullayev.mysql.pythonanywhere-services.com'
    mysql_db = 'hbnnarzullayev$flask3'
    data = request.get_json()
    group = data.get('group')
    app_id = data.get('app_id')
    if group and '-' in group:
        group = group.split('-')[0]  # Get the part before the first '-'
        print(f"Extracted part of group: {group}")

    # Connect to database
    dbs = msd.connect(host=host, user=user, password=password, database=mysql_db)
    cursor = dbs.cursor()

    # Fetch quizzes from the database based on the group
    cursor.execute("SELECT * FROM quizes WHERE science = %s and Teacher=%s", (group,app_id,))
    quizzes = cursor.fetchall()
    cursor.close()
    dbs.close()

    # Return the quizzes as JSON
    quizzes_data = [{'id': quiz[0], 'quiz': quiz[1]} for quiz in quizzes]  # Assuming the quiz ID is the first column
    return jsonify(quizzes_data)

@app.route('/delete_quiz', methods=['POST'])
def delete_quiz():
    data = request.get_json()
    print("delete_quiz")
    print(data)
    quiz_id = data.get('quiz_id')
    app_id = data.get('app_id')
    user = 'hbnnarzullayev'
    password = 'Sersarson7'
    host = 'hbnnarzullayev.mysql.pythonanywhere-services.com'
    mysql_db = 'hbnnarzullayev$flask3'

    # Connect to the database
    dbs = msd.connect(host=host, user=user, password=password, database=mysql_db)
    cursor = dbs.cursor()

    # Delete the quiz from the database
    cursor.execute("DELETE FROM quizes WHERE id = %s and Teacher=%s", (quiz_id,app_id,))
    dbs.commit()
    cursor.close()
    dbs.close()

    return jsonify({'message': 'Quiz deleted successfully'}), 200

@app.route('/get_food_groups', methods=['GET'])
def get_food_groups():
    # Database connection details
    print("get_food_groups")
    data = request.args  # Using GET parameters
    print(data)
    app_id = data.get("app_id")
    user = 'hbnnarzullayev'
    password = 'Sersarson7'
    host = 'hbnnarzullayev.mysql.pythonanywhere-services.com'
    mysql_db = 'hbnnarzullayev$flask3'

    # Connect to the database
    dbs = msd.connect(host=host, user=user, password=password, database=mysql_db)
    cursor = dbs.cursor()
    # Fetch distinct group names from the database
    cursor.execute("SELECT DISTINCT group_name, Image FROM food_groups")
    food_groups = cursor.fetchall()
    cursor.close()
    dbs.close()

    # Convert the query result into the desired format
    food_groups_list = [{'name': group[0], 'group_name': group[0], 'image': group[1]} for group in food_groups]

    # Print for debugging purposes
    print(food_groups_list)

    # Return the food groups as JSON
    return jsonify(food_groups_list)

@app.route('/get_food_groups_admin', methods=['GET'])
def get_food_groups_admin():
    # Database connection details
    print("get_food_groups_admin")
    data = request.args  # Using GET parameters
    print(data)
    app_id = data.get("app_id")
    username = data.get("username")
    user = 'hbnnarzullayev'
    password = 'Sersarson7'
    host = 'hbnnarzullayev.mysql.pythonanywhere-services.com'
    mysql_db = 'hbnnarzullayev$flask3'

    # Connect to the database
    dbs = msd.connect(host=host, user=user, password=password, database=mysql_db)
    cursor = dbs.cursor()

        # Check if the user is an admin
    cursor.execute("SELECT user_role FROM users2 WHERE username = %s and app_id=%s", (username, app_id,))
    user_data = cursor.fetchone()
    print(user_data)

    if user_data:
        user_role = user_data[0]
        if user_role == 1:
            is_admin = 'true'
        else:
            is_admin = False
    else:
        is_admin = False
    print(is_admin)

    # Fetch distinct group names from the database
    cursor.execute("SELECT DISTINCT group_name, Image FROM food_groups")
    food_groups = cursor.fetchall()
    cursor.close()
    dbs.close()

    # Convert the query result into the desired format
    food_groups_list = [{'name': group[0], 'group_name': group[0], 'image': group[1]} for group in food_groups]

    # Print for debugging purposes
    print(food_groups_list)

    # Return the food groups as JSON
    # return jsonify(food_groups_list)
    return jsonify({'food_groups_list': food_groups_list, 'is_admin': is_admin})

def extract_video_id(url):
    """Extract YouTube video ID from different formats"""

    # ✅ Handle "youtu.be/VIDEO_ID" short links
    if "youtu.be" in url:
        return url.split("/")[-1].split("?")[0]  # Extract after last "/"

    # ✅ Handle "youtube.com/watch?v=VIDEO_ID" full links
    parsed_url = urlparse(url)
    if 'youtube.com' in parsed_url.netloc and 'v' in parse_qs(parsed_url.query):
        return parse_qs(parsed_url.query)['v'][0]

    return None  # If no match found

def get_videos():
    user = 'hbnnarzullayev'
    password = 'Sersarson7'
    host = 'hbnnarzullayev.mysql.pythonanywhere-services.com'
    mysql_db = 'hbnnarzullayev$flask3'
    # Connect to the database
    dbs = msd.connect(host=host, user=user, password=password, database=mysql_db)
    cursor = dbs.cursor()
    # conn = MySQLdb.connect("hbnnarzullayev.mysql.pythonanywhere-services.com","hbnnarzullayev","Sersarson7","hbnnarzullayev$flask3" )
    # cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT video_url, name FROM youtube_food_videos where link_type is null ORDER BY RAND() LIMIT 5;")
    videos = cursor.fetchall()
    cursor.close()
    dbs.close()
    video_list = [{'video_url': video[0], 'name': video[1]} for video in videos]
    return video_list

@app.route('/suggested_videos', methods=['GET'])
def suggested_videos():
    videos = get_videos()
    print(videos)
    return jsonify([
        {
            'title': video['name'],  # ✅ Correct key
            'videoId': extract_video_id(video['video_url']),  # ✅ Extract Video ID from URL
            'thumbnail': f"https://img.youtube.com/vi/{extract_video_id(video['video_url'])}/hqdefault.jpg"
        }
        for video in videos
    ])

def get_gold_prices():
    print("Fetching gold prices...")
    response = requests.get(API_URL, headers={'X-Api-Key': API_KEY})

    if response.status_code == requests.codes.ok:
        data = response.json()
        print("Received Data:", data)  # Debugging uchun

        # JSON tuzilmasini tekshiramiz
        if "price" in data:
            return data["price"]
        else:
            print("Warning: 'price' key not found in API response.")
            return None
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None

# Oddiy tahlil va tavsiya
def analyze_gold_price(prices):
    print("analyze_gold_price")
    values = prices[-10:]  # Oxirgi 10 ta kurs
    df = pd.Series(values)
    trend = df.diff().mean()  # O'rtacha o'zgarish

    if trend > 0:
        action = "Sotib olish tavsiya etiladi"
        take_profit = max(values) * 1.02  # 2% foyda
        stop_loss = min(values) * 0.98   # 2% zarar
        print(f"{action} - {take_profit} - {stop_loss}")
    elif trend < 0:
        action = "Sotish tavsiya etiladi"
        take_profit = min(values) * 0.98  # 2% foyda
        stop_loss = max(values) * 1.02   # 2% zarar
    else:
        action = "Neutral"
        take_profit = None
        stop_loss = None

    return action, take_profit, stop_loss

@app.route('/analiz', methods=['GET', 'POST'])
def analiz():
    recommendation = None
    take_profit = None
    stop_loss = None
    gold_price = get_gold_price()
    historical_prices = get_historical_gold_prices()

    if request.method == 'POST' and historical_prices:
        recommendation, take_profit, stop_loss = analyze_gold_price(historical_prices)

    return render_template('analiz.html', gold_price=gold_price, recommendation=recommendation, take_profit=take_profit, stop_loss=stop_loss)

def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'xls', 'xlsx', 'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload_files_tosave', methods=['GET', 'POST'])
def upload_files_tosave():
    if session.get("nma") and session['loggedin'] == True:
        UPLOAD_FOLDER = '/home/hbnnarzullayev/mysite/uploads/saved_files/'
        app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

        # Ruxsat etilgan fayl turlari
        ALLOWED_EXTENSIONS = {'xls', 'xlsx', 'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}
        if request.method == 'POST':
            parol = request.form['password']
            # Fayl yuklanganligini tekshirish
            if 'file' not in request.files:
                return 'Fayl topilmadi'
            file = request.files['file']

            # Fayl nomi bo'sh emasligini tekshirish
            if file.filename == '':
                return 'Fayl tanlanmadi'

            # Fayl ruxsat etilgan turda ekanligini tekshirish va saqlash
            if file and allowed_file(file.filename):
                file_pkey = generate_random_pkey()
                now = datetime.now(tz=timezone(timedelta(hours=5)))
                user = 'hbnnarzullayev'
                password = 'Sersarson7'
                host = 'hbnnarzullayev.mysql.pythonanywhere-services.com'
                mysql_db = 'hbnnarzullayev$flask3'
                # Connect to the database
                dbs = msd.connect(host=host, user=user, password=password, database=mysql_db)
                cursor = dbs.cursor()
                s_filename = file_pkey + '-' + file.filename
                cursor.execute('''INSERT INTO saved_files (file_name, username, date_time, pkey, password) VALUES (%s, %s, %s, %s, %s)''',
                           [s_filename, session.get("nma"), now.strftime('%Y-%m-%d %H:%M:%S'), file_pkey, parol])
                dbs.commit()
                cursor.close()
                dbs.close()
                file_path = f"/home/hbnnarzullayev/mysite/uploads/saved_files/"+s_filename
                file.save(file_path)
                return 'Fayl muvaffaqiyatli yuklandi!'
        username = session.get("nma")
        files = get_uploaded_files(username)
        return render_template('upload_files_tosave.html', files=files)
    return render_template('index.html')

def get_uploaded_files(username):
    user = 'hbnnarzullayev'
    password = 'Sersarson7'
    host = 'hbnnarzullayev.mysql.pythonanywhere-services.com'
    mysql_db = 'hbnnarzullayev$flask3'

    # MySQL ga ulanish
    dbs = msd.connect(host=host, user=user, password=password, database=mysql_db)
    cursor = dbs.cursor()

    # To'g'ri parametrlangan SQL so'rovi
    if username == 'Admin':
        cursor.execute("SELECT ID, file_name, username, date_time, pkey FROM saved_files")
        files = cursor.fetchall()
    else:
        cursor.execute("SELECT ID, file_name, username, date_time, pkey FROM saved_files WHERE username = %s", (username,))
        files = cursor.fetchall()

    cursor.close()
    dbs.close()  # Ulanishni yopish
    return files

@app.route('/delete/<int:file_id>', methods=['POST'])
def delete_file(file_id):
    user = 'hbnnarzullayev'
    password = 'Sersarson7'
    host = 'hbnnarzullayev.mysql.pythonanywhere-services.com'
    mysql_db = 'hbnnarzullayev$flask3'
    # Connect to the database
    dbs = msd.connect(host=host, user=user, password=password, database=mysql_db)
    cursor = dbs.cursor()
    UPLOAD_FOLDER = '/home/hbnnarzullayev/mysite/uploads/saved_files/'

    # Fayl nomini bazadan olish
    cursor.execute("SELECT file_name FROM saved_files WHERE ID = %s", (file_id,))
    file = cursor.fetchone()
    if file:
        file_name = file[0]
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file_name)

        # Serverdan faylni o'chirish
        if os.path.exists(file_path):
            os.remove(file_path)

        # Bazadan fayl yozuvini o'chirish
        cursor.execute("DELETE FROM saved_files WHERE ID = %s", (file_id,))
        dbs.commit()

    cursor.close()
    return redirect(url_for('upload_files_tosave'))

@app.route('/download/<int:file_id>')
def download_file(file_id):
    print("file_id:")
    print(file_id)
    username = session.get("nma")
    UPLOAD_FOLDER = '/home/hbnnarzullayev/mysite/uploads/saved_files/'
    if not username:
        return "Foydalanuvchi login qilmagan!", 403
    user = 'hbnnarzullayev'
    password = 'Sersarson7'
    host = 'hbnnarzullayev.mysql.pythonanywhere-services.com'
    mysql_db = 'hbnnarzullayev$flask3'
    # Connect to the database
    dbs = msd.connect(host=host, user=user, password=password, database=mysql_db)
    cursor = dbs.cursor()
    cursor.execute("SELECT file_name FROM saved_files WHERE ID = %s", (file_id,))
    file = cursor.fetchone()
    if file is None:
        cursor.execute("SELECT file_name FROM messages WHERE ID = %s", (file_id,))
        file = cursor.fetchone()
        print(file)
    dbs.close()

    if file:
        file_name = file[0]
        print(file_name)
        return send_from_directory('/home/hbnnarzullayev/mysite/uploads/saved_files/', file_name, as_attachment=True)
    else:
        return "Fayl topilmadi yoki sizga tegishli emas!", 404
