import pandas as pd
from flask_cors import *
# #import numpy as np
# import random
import re
import hashlib
import time
import schedule
import threading
from flask_mysqldb import MySQL
#from app import app
#import urllib.request
#import requests
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
from flask import Flask, redirect, render_template, request, session, url_for, send_from_directory,flash, send_file, g, jsonify
from functools import wraps
#from flask import Flask,render_template, request
#from flask_mysqldb import MySQL
from flask_mail import *
from flask_sqlalchemy import SQLAlchemy
#from pydal import DAL, Field
#from flask import send_from_directory
from flask_socketio import SocketIO
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

# Redis connection setup
redis_host = 'redis-11369.c1.asia-northeast1-1.gce.cloud.redislabs.com'
redis_port = 11369
redis_password = 'g2Td26z6j4UbQKLBe6gZ75eh7zya3tSf'

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
socketio = SocketIO(app)

# app.config['MAIL_SERVER'] = 'smtp.gmail.com'
# app.config['MAIL_PORT'] = 997
# app.config['MAIL_USE_TLS'] = True
# app.config['MAIL_USERNAME'] = 'hbncompanyofficials@gmail.com'  # Your Gmail address
# app.config['MAIL_PASSWORD'] = 'Sersarson7$'  # Your Gmail password or app-specific password

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

def execute_query_to_dataframe(query):
    app.secret_key = "super secret key"
    user = 'hbnnarzullayev'
    password = 'Sersarson7'
    host = 'hbnnarzullayev.mysql.pythonanywhere-services.com'
    mysql = 'hbnnarzullayev$flask3'
    dbs = msd.connect(host,user,password, mysql)
    cursor = dbs.cursor()
    a=cursor.execute("""DECLARE
  cnt INT := 0;
  v_summa NUMBER;
BEGIN
  WHILE cnt < 5 LOOP
    dbms_output.put_line('Inside simulated FOR LOOP on TechOnTheNet.com');
    cnt := cnt + 1;

    FOR rec IN (SELECT t.tin,
                       t.na2_code,
                       t.date_srok,
                       t.nachislen_n,
                       t.umenshen_n
                  FROM nla_copy t
                 WHERE t.date_srok < DATE '2013-11-10'
                   AND t.na2_code IN (SELECT DISTINCT na2_code
                                        FROM nla_copy
                                       WHERE na2_code NOT IN ('46', '101', '191', '199'))
                 ORDER BY na2_code ASC, DATE_SROK DESC
                 OFFSET 0 ROWS
                 FETCH NEXT (SELECT COUNT(DISTINCT(na2_code))
                               FROM nla_copy
                              WHERE na2_code NOT IN ('46', '101', '191', '199')) ROWS ONLY)
    LOOP
      IF rec.nachislen_n > 0 THEN
          v_summa := v_summa + rec.nachislen_n;
          update nla_copy set saldo_tek_n=v_summa;
      END IF;
    END LOOP;
  END LOOP;
  dbms_output.put_line('Total summa: ' || v_summa);
END;""")
    print (a)

    return df