import pandas as pd
import numpy as np
import re
import requests
import MySQLdb.cursors
from flask_login import LoginManager
from flask import Flask, redirect, url_for, render_template, request, session
from flask import Flask,render_template, request
from flask_mysqldb import MySQL
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)

app.config['MYSQL_HOST'] = 'hbnnarzullayev.mysql.pythonanywhere-services.com'
app.config['MYSQL_USER'] = 'hbnnarzullayev'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'hbnnarzullayev@flask3'
mysql=MySQL(app)

#new_data_frame = pd.read_csv('stock_prices.csv')

@app.route("/", methods=['GET', 'POST'])
def home():
    city = "London"  # Replace with your desired city or make it dynamic
    temperature, description = get_weather(city)
    id:home
    if request.method == 'GET':
        pass

    if request.method == 'POST' and request.content == "True":
        pass

    return render_template("index.html", content="True", temperature=temperature, description=description)

def get_weather(city):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid=2994f73aeemsh232bd3c479702b3p1d9077jsnbca2588c950e"
    response = requests.get(url)
    data = response.json()
    #return data
    try:
        temperature = data["main"]["temp"]
        description = data["weather"][0]["description"]
    except KeyError as e:
        # Handle the KeyError appropriately (e.g., provide a default value or error message)
        temperature = None
        description = "Weather data not available"

    return temperature, description

@app.route("/signup", methods=["POST", "GET"])
def signup():
    result = 0
    n = 2
    for i in range(1, n + 1):
        result += i
        # this ^^ is the shorthand for
        # result = result + i
    msg = ''
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['pwd']
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM users2 WHERE username = %s and password = %s and email = %s', (username,password,email,))
        user = cursor.fetchall()
        print(user)
        if user:
            session['username'] = username
            session['pwd'] = password
            msg = 'Account already exists!'
        elif not re.match(r'[^@]+@[^@]+\.[^@]+', email):
            msg = 'Invalid email address!'
        elif not re.match(r'[A-Za-z0-9]+', username):
            msg = 'Username must contain only characters and numbers!'
        elif not username or not password or not email:
            msg = 'Please fill out the form!'
        else:
            # Account doesnt exists and the form data is valid, now insert new account into accounts table
            cursor.execute(''' INSERT INTO users2 VALUES(%s,%s,%s,%s)''',[username,password,email,result])
            mysql.connection.commit()
            msg = 'You have successfully registered!'
    else:
            msg = 'Incorrect username / password !'
    return render_template('signup.html', msg=msg)



@app.route("/login", methods=["POST", "GET"])
def login():
    error = ''
    if request.method == 'POST':
        username = request.form['nma']
        password = request.form['pwda']
        cur = mysql.connection.cursor(MySQLdb.cursors.DictCursor)

        cur = mysql.connection.cursor()
        cur.execute("SELECT * FROM users2 WHERE username = %s and password = %s", (username,password,))
        user = cur.fetchall()
        cur.close()
        print(user)
        if user:
            session['loggedin'] = True
            session['nma'] = username
            session['pwda'] = password
            return redirect('/profile')
        else:
            error = 'Invalid username or password'
            return render_template('login.html', error=error)

    return render_template('login.html')

@app.route('/logout')
def logout():
    msg = ''
    session.pop('loggedin', None)
    session.pop('id', None)
    session.pop('username', None)
    # Check if user is loggedin
    if 'loggedin' in session:
        # We need all the account info for the user so we can display it on the profile page
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('DELETE FROM users2 where username = %s and password = %s', (session['nma'],session['pwda']))
        user = cursor.fetchone()
        # Show the profile page with account info
        return render_template('profile.html', user=user)
    else:
            msg = 'not logged in'
    # User is not loggedin redirect to login page
    return render_template('index.html', msg=msg)

@app.route("/test_form", methods=["POST", "GET"])
def test_form():
    msg = ''
    if request.method == "POST" and request.form["nma"] == "admin" and request.form["pwda"] != "":
        #user = request.form["nm"]
        return redirect(url_for("profile"))
    else:
        return render_template("test_form.html")

@app.route("/profile")
def profile():
    # Check if user is loggedin
    if 'loggedin' in session:
        # We need all the account info for the user so we can display it on the profile page
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM users2 WHERE username = %s and password = %s', (session['nma'],session['pwda']))
        user = cursor.fetchone()
        # Show the profile page with account info
        return render_template('profile.html', user=user)
    # User is not loggedin redirect to login page
    return redirect(url_for('login'))

@app.route("/yangiliklar1")
def yangiliklar1():
    if request.method == "POST" and request.form["nma"] == "admin":
        #user = request.form["nm"]
        return redirect(url_for("yangiliklar1"))
    else:
        return render_template("yangiliklar1.html")


@app.route("/portal", methods=["POST", "GET"])
def portal():
    if request.method == "POST" and request.form["nma"] == "admin":
        #user = request.form["nm"]
        return redirect(url_for("profile"))
    else:
        return render_template("portal.html")


@app.route("/<usr>")
def user(usr):
    return f"<h1>{usr}</h1>"

