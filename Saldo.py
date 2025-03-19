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
            cursor.execute("SELECT date_srok from nla where date_srok='2020-01-01'  and tin=%s group by date_srok",(tin,))
            simple_year = cursor.fetchone()[0]
            cursor.execute("SELECT DISTINCT date_srok FROM nla where date_srok>=%s and tin=%s GROUP BY date_srok ORDER BY date_srok asc",(min_date_srok, tin, ))
            dates = [row[0] for row in cursor.fetchall()]
            # Fetch distinct na2_codes
            cursor.execute("SELECT DISTINCT na2_code FROM nla where tin=%s GROUP BY na2_code ORDER BY na2_code", (tin, ))
            na2_codes = [row[0] for row in cursor.fetchall()]
            cursor.execute("select date_srok from start  where tin=%s group by date_srok", (tin, ))
            pr_dates = [row[0] for row in cursor.fetchall()]
            for current_date in dates:
                cursor.execute("update nla set pr_penya=0 where date_srok>%s and tin=%s", (current_date,tin,))
                cursor.execute("SELECT SUM(saldo_all) FROM nla WHERE date_srok = %s  and tin=%s group by date_srok", (current_date, tin, ))
                saldo_all_codes = cursor.fetchone()[0]
                cursor.execute("UPDATE nla SET saldo_all_codes = %s WHERE last_date_srok = %s and tin=%s", (saldo_all_codes, current_date, tin, ))
                cursor.execute("select percentage from ulush where data=%s", (current_date,))
                foiz = cursor.fetchone()[0]
                cursor.execute("select ulush from ulush where data=%s", (current_date,))
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
                                cursor.execute("select percentage from ulush where data=%s", (nlb_date1,))
                                foiz = cursor.fetchone()[0]
                                cursor.execute("select last_date_srok from nlb where date_srok=%s and tin=%s group by last_date_srok", (nlb_date1, tin, ))
                                last_date_srok_p = cursor.fetchone()
                                cursor.execute("select ulush from ulush where data=ifnull(%s,%s)", (nlb_date1, last_date_srok_p, ))
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
                                    cursor.execute("UPDATE nlb SET pr_penya = if((ifnull(pr_saldo_all,0))>0,((ifnull(pr_saldo_all,0))/%s)*pr_penya_all,0) where last_date_srok=%s and tin=%s", (pr_ned_sum, nlb_date1, tin, ))
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
                                cursor.execute("select percentage from ulush where data=%s", (nlb_date,))
                                foiz = cursor.fetchone()[0]
                                cursor.execute("select last_date_srok from nlb where date_srok=%s and tin=%s group by last_date_srok", (nlb_date, tin, ))
                                last_date_srok_p = cursor.fetchone()
                                cursor.execute("select ulush from ulush where data=ifnull(%s,%s)", (nlb_date, last_date_srok_p, ))
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
                                    cursor.execute("UPDATE nlb SET pr_penya = if((ifnull(pr_saldo_all,0))>0,((ifnull(pr_saldo_all,0))/%s)*pr_penya_all,0) where last_date_srok=%s and tin=%s", (pr_ned_sum, nlb_date, tin, ))
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
                # 2020dan keyin
                elif current_date > simple_year:
                    if current_date in pr_dates:
                        cursor.execute("select max(last_date_srok) from nlb where pr_saldo_all_codes<>0 and tin=%s", (tin, ))
                        max_last_pr_date = cursor.fetchone()[0]
                        cursor.execute("select max(last_date_srok) from nlb where tin=%s", (tin, ))
                        max_last_pr_dates = cursor.fetchone()[0]
                        cursor.execute("SELECT DISTINCT na2_code FROM nla where tin=%s and last_date_srok=%s GROUP BY na2_code ORDER BY na2_code", (tin, current_date, ))
                        na2_codes = [row[0] for row in cursor.fetchall()]
                        cursor.execute("delete from nlb where tin=%s", (tin, ))
                        cursor.execute("select min(date) from start where date_srok=%s and tin=%s",(current_date,tin, ))
                        min_pr_date = cursor.fetchone()[0]
                        cursor.execute("insert into nlb (tin, na2_code, date_srok, saldo_all, shtraf, oldingi_kun_saldo, saldo_all_codes,sum_p, saldo_tek_p, last_date_srok) select tin, na2_code, date_srok, saldo_all, ifnull(shtraf,0), oldingi_kun_saldo, saldo_all_codes, sum_p, saldo_tek_p, last_date_srok from nla where date_srok<=%s and last_date_srok>=%s  and tin=%s order by date_srok, na2_code;", (current_date, min_pr_date, tin, ))
                        cursor.execute("select last_date_srok from nlb where tin=%s group by last_date_srok", (tin, ))
                        nlb_dates1 = [row[0] for row in cursor.fetchall()]
                        for nlb_date1 in nlb_dates1:
                            if nlb_date1<=simple_year:
                                cursor.execute("SELECT DISTINCT na2_code FROM nla where tin=%s GROUP BY na2_code ORDER BY na2_code", (tin, ))
                                na2_codes1 = [row[0] for row in cursor.fetchall()]
                                cursor.execute("select percentage from ulush where data=%s", (nlb_date1,))
                                foiz = cursor.fetchone()[0]
                                cursor.execute("select last_date_srok from nlb where date_srok=%s and tin=%s group by last_date_srok", (nlb_date1, tin, ))
                                last_date_srok_p = cursor.fetchone()
                                cursor.execute("select ulush from ulush where data=ifnull(%s,%s)", (nlb_date1, last_date_srok_p, ))
                                ulush = cursor.fetchone()[0]
                                cursor.execute("select sum(shtraf) from nlb where date_srok=%s and tin=%s", (nlb_date1, tin, ))
                                shtraf_all = cursor.fetchone()[0]
                                cursor.execute("SELECT sum(ifnull(nachislen_n-umenshen_n,0)) as sums_all FROM pereraschet WHERE date_real<= %s and date_srok>=%s and date_srok<=%s and date_srok>date_real and tin=%s", (nlb_date, nlb_date, current_date, tin, ))
                                pr_sum_all = cursor.fetchone()[0]
                                cursor.execute("UPDATE nlb SET pr_saldo_all_codes=(ifnull(%s,0)-ifnull(saldo_all_codes,0)+ifnull(%s,0)) where last_date_srok=%s and tin=%s", (pr_sum_all, shtraf_all, nlb_date1, tin, ))
                                cursor.execute("select max(last_date_srok) from nlb where tin=%s", (tin, ))
                                for na2_code1 in na2_codes1:
                                    now = datetime.now(tz=timezone(timedelta(hours=5)))
                                    cursor.execute("select DATEDIFF(date_srok,last_date_srok) as farq from nlb where last_date_srok=%s and na2_code=%s and tin=%s", (nlb_date1,na2_code1, tin, ))
                                    farq = cursor.fetchone()
                                    cursor.execute("SELECT sum(nachislen_n-umenshen_n) as sums FROM pereraschet WHERE  date_srok<=%s and date_real<= %s and date_srok>=%s and na2_code=%s and date_srok>date_real and tin=%s", (current_date, nlb_date, nlb_date, na2_code1, tin, ))
                                    pr_sum = cursor.fetchone()[0]
                                    cursor.execute("UPDATE nlb SET pr_saldo_all=(ifnull(%s,0)+ifnull(shtraf,0)-oldingi_kun_saldo) where last_date_srok=%s and na2_code=%s and tin=%s", (pr_sum, nlb_date1, na2_code1, tin, ))
                                    cursor.execute("SELECT sum(pr_saldo_all) as sums FROM nlb WHERE pr_saldo_all>0 and last_date_srok=%s and tin=%s", (nlb_date1, tin, ))
                                    pr_ned_sum = cursor.fetchone()[0]
                                    cursor.execute("UPDATE nlb SET pr_penya_all = if(pr_saldo_all_codes>0,(pr_saldo_all_codes*(%s)*(%s)*(%s)),0) where last_date_srok=%s and na2_code=%s and tin=%s", (foiz,ulush,farq,nlb_date1, na2_code1, tin, ))
                                    # cursor.execute("UPDATE nlb SET pr_penya_all = if((saldo_all_codes-ifnull(%s,0))<0,((-1)*(saldo_all_codes-ifnull(%s,0))*(%s)*(%s)*(%s)),0) where last_date_srok=%s and na2_code=%s", (pr_sum_all, pr_sum_all, foiz,ulush,farq,nlb_date, na2_code1,))
                                    cursor.execute("UPDATE nlb SET pr_penya = if((ifnull(pr_saldo_all,0))>0,((ifnull(pr_saldo_all,0))/%s)*pr_penya_all,0) where last_date_srok=%s and tin=%s", (pr_ned_sum, nlb_date1, tin, ))
                                    cursor.execute("delete from pr_progress where pr_date!=%s and tin=%s", (nlb_date1, tin, ))
                                    cursor.execute("INSERT INTO pr_progress (tin, pr_date, na2_code, datetime) VALUES (%s, %s, %s, %s)", (tin, nlb_date1, na2_code1,now.strftime('%Y-%m-%d %H:%M:%S')))

                                dbs.commit()
                            else:
                                cursor.execute("select percentage from ulush where data=%s", (nlb_date1,))
                                foiz = cursor.fetchone()[0]
                                cursor.execute("select last_date_srok from nlb where date_srok=%s and tin=%s group by last_date_srok", (nlb_date1, tin, ))
                                last_date_srok_p = cursor.fetchone()
                                cursor.execute("select ulush from ulush where data=ifnull(%s,%s)", (nlb_date1, last_date_srok_p, ))
                                ulush = cursor.fetchone()[0]
                                cursor.execute("select sum(shtraf) from nlb where date_srok=%s and tin=%s", (nlb_date1, tin, ))
                                shtraf_all = cursor.fetchone()[0]
                                cursor.execute("SELECT sum(ifnull(nachislen_n-umenshen_n,0)) as sums_all FROM pereraschet WHERE date_real<= %s and date_srok>=%s and date_srok<=%s and date_srok>date_real and tin=%s", (nlb_date1, nlb_date1, current_date, tin, ))
                                pr_sum_all = cursor.fetchone()[0]
                                # cursor.execute("UPDATE nlb SET pr_saldo_all_codes=(ifnull(%s,0)-ifnull(saldo_all_codes,0)+ifnull(%s,0)) where last_date_srok=%s and tin=%s", (pr_sum_all, shtraf_all, nlb_date1, tin, ))
                                cursor.execute("select max(last_date_srok) from nlb where tin=%s", (tin, ))
                                for na2_code1 in na2_codes:
                                    now = datetime.now(tz=timezone(timedelta(hours=5)))
                                    cursor.execute("select DATEDIFF(date_srok,last_date_srok) as farq from nlb where date_srok=%s and na2_code=%s and tin=%s", (nlb_date1,na2_code1, tin, ))
                                    farq = cursor.fetchone()
                                    cursor.execute("SELECT sum(nachislen_n-umenshen_n) as sums FROM pereraschet WHERE  date_srok<=%s and date_real<= %s and date_srok>=%s and na2_code=%s and date_srok>date_real and tin=%s", (current_date, nlb_date1, nlb_date1, na2_code1, tin, ))
                                    pr_sum = cursor.fetchone()[0]
                                    cursor.execute("UPDATE nlb SET pr_saldo_all=(ifnull(%s,0)+ifnull(shtraf,0)-oldingi_kun_saldo) where date_srok=%s and na2_code=%s and tin=%s", (pr_sum, nlb_date1, na2_code1, tin, ))
                                    cursor.execute("SELECT sum(pr_saldo_all) as sums FROM nlb WHERE pr_saldo_all>0 and date_srok=%s and tin=%s", (nlb_date1, tin, ))
                                    pr_ned_sum = cursor.fetchone()[0]
                                    cursor.execute("UPDATE nlb SET pr_penya = if((ifnull(pr_saldo_all,0))>0,((ifnull(pr_saldo_all,0))*(%s)*(%s)*(%s)),0) where date_srok=%s and na2_code=%s and tin=%s", (foiz,ulush,farq,nlb_date1, na2_code1, tin, ))
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
                                cursor.execute("SELECT DISTINCT na2_code FROM nla where tin=%s and date_srok=%s GROUP BY na2_code ORDER BY na2_code", (tin, current_date, ))
                                na2_codes3 = [row[0] for row in cursor.fetchall()]
                                for na2_code1 in na2_codes3:
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
                    cursor.execute("update nla set sum_p=0 where date_srok=%s and tin=%s", (current_date,tin, ))
                    cursor.execute("SELECT DISTINCT na2_code FROM nla where tin=%s and date_srok=%s GROUP BY na2_code ORDER BY na2_code", (tin, current_date, ))
                    na2_codess = [row[0] for row in cursor.fetchall()]
                    for na2_code1 in na2_codess:
                        now = datetime.now(tz=timezone(timedelta(hours=5)))
                        cursor.execute("SELECT last_date_srok FROM nla WHERE date_srok = %s and na2_code=%s and tin=%s", (current_date, na2_code1, tin, ))
                        last_date_od = cursor.fetchone()[0]
                        cursor.execute("SELECT IFNULL(saldo_all, 0) FROM nla WHERE date_srok = %s and na2_code=%s and tin=%s", (last_date_od, na2_code1, tin,))
                        saldo_all = cursor.fetchone()[0]
                        cursor.execute("select sum(shtraf1) as summa from nla where date_srok=%s and shtraf1<>0 and na2_code=%s and tin=%s;", (last_date_od,na2_code1, tin, ))
                        last_shtraf = cursor.fetchone()[0]
                        cursor.execute("SELECT IFNULL(saldo_tek_p, 0) FROM nla WHERE date_srok = %s and na2_code=%s and tin=%s", (last_date_od, na2_code1, tin, ))
                        saldo_tek_p = cursor.fetchone()[0]
                        cursor.execute("UPDATE nla SET oldingi_kun_saldo = %s WHERE date_srok = %s AND na2_code = %s and tin=%s", (saldo_all, current_date, na2_code1, tin,))
                        cursor.execute("select DATEDIFF(date_srok,last_date_srok) as farq from nla where date_srok=%s and na2_code=%s and tin=%s", (current_date,na2_code1,tin,))
                        farq = cursor.fetchone()[0]
                        cursor.execute("UPDATE nla SET shtraf1 = case when (-1)*(%s+uploch_n+umenshen_n-nachislen_n-vozvrat-uploch_p)>0 and (-1)*(%s+uploch_n+umenshen_n-nachislen_n-vozvrat-uploch_p)>=(%s+shtraf) then (%s+shtraf) when (-1)*(%s+uploch_n+umenshen_n-nachislen_n-vozvrat-uploch_p)>0 and (-1)*(%s+uploch_n+umenshen_n-nachislen_n-vozvrat-uploch_p)<(%s+shtraf) then (-1)*(%s+uploch_n+umenshen_n-nachislen_n-vozvrat-uploch_p) else 0 end where date_srok=%s and na2_code=%s and tin=%s", (saldo_all, saldo_all, last_shtraf, last_shtraf, saldo_all, saldo_all, last_shtraf, saldo_all, current_date, na2_code1, tin, ))
                        cursor.execute("select sum(shtraf1) as sum_shtraf from nla where date_srok=%s and na2_code=%s and tin=%s", (last_date_od,na2_code1,tin, ))
                        sum_shtraf = cursor.fetchone()[0]
                        cursor.execute("select sum(shtraf1) as sum_shtraf from nla where date_srok=%s and tin=%s", (last_date_od, tin, ))
                        sum_shtraf_all = cursor.fetchone()[0]
                        cursor.execute("UPDATE nla SET oldingi_kun_n = if(oldingi_kun_saldo<0,(-1*oldingi_kun_saldo),0) where date_srok=%s and na2_code=%s and tin=%s", (current_date,na2_code1,tin, ))
                        cursor.execute("select sum(shtraf1) as sum_shtraf from nla where date_srok=%s and na2_code=%s and tin=%s", (last_date_od,na2_code1,tin, ))
                        sum_shtraf = cursor.fetchone()[0]
                        cursor.execute("SELECT IFNULL((-1*(oldingi_kun_saldo+%s)*(%s)*(%s)*(%s)), 0) FROM nla WHERE date_srok = %s and ((-1)*oldingi_kun_n+%s)<0 and na2_code=%s and tin=%s", (sum_shtraf, foiz,ulush,farq,current_date, sum_shtraf, na2_code1,tin,))
                        sum_p1 = cursor.fetchone()
                        if sum_p1 is not None:
                            sum_p = sum_p1[0]
                        else:
                            sum_p = 0
                        cursor.execute("UPDATE nla SET sum_p = if(((-1)*oldingi_kun_n+%s)<0,(-1*(oldingi_kun_saldo+%s)*(%s)*(%s)*(%s)+ifnull(pr_penya,0)),ifnull(pr_penya,0)) where date_srok=%s and na2_code=%s and tin=%s", (sum_shtraf, sum_shtraf, foiz,ulush,farq,current_date, na2_code1,tin,))
                        cursor.execute("UPDATE nla SET uploch_p = 0 WHERE date_srok = %s AND na2_code = %s and tin=%s", (current_date, na2_code1, tin))
                        # cursor.execute("UPDATE nla SET uploch_p = if((%s+uploch_n+umenshen_n-nachislen_n-vozvrat)>=saldo_tek_p,saldo_tek_p,(%s+uploch_n+umenshen_n-nachislen_n-vozvrat)), saldo_tek_p=if(uploch_p=%s+sum_p,0,%s+sum_p-uploch_p) WHERE (%s+uploch_n+umenshen_n-nachislen_n-vozvrat-uploch_p)>=0 and last_date_srok = %s AND na2_code = %s and tin=%s", (saldo_all, saldo_all, saldo_tek_p, saldo_tek_p, saldo_all, current_date, na2_code1, tin, ))
                        # cursor.execute("UPDATE nla SET uploch_p = if((%s+uploch_n+umenshen_n-nachislen_n-vozvrat)>=saldo_tek_p,saldo_tek_p,(%s+uploch_n+umenshen_n-nachislen_n-vozvrat)), saldo_tek_p=if(uploch_p=%s+sum_p,0,%s+sum_p-uploch_p) WHERE (%s+uploch_n+umenshen_n-nachislen_n-vozvrat-uploch_p)>=0 and date_srok = %s AND na2_code = %s", (saldo_all, saldo_all, saldo_tek_p, saldo_tek_p, saldo_all, current_date, na2_code1))
                        cursor.execute("UPDATE nla SET saldo_tek_p = (%s+%s+pr_penya) WHERE date_srok = %s AND na2_code = %s and tin=%s", (saldo_tek_p, sum_p, current_date, na2_code1, tin,))
                        dbs.commit
                        cursor.execute("UPDATE nla SET uploch_p = case when (%s+uploch_n+umenshen_n-nachislen_n-vozvrat)>=saldo_tek_p and saldo_tek_p>0 then saldo_tek_p when (%s+uploch_n+umenshen_n-nachislen_n-vozvrat)>0 and (%s+uploch_n+umenshen_n-nachislen_n-vozvrat)<saldo_tek_p then (%s+uploch_n+umenshen_n-nachislen_n-vozvrat) when saldo_tek_p<0 then saldo_tek_p else 0 end, saldo_tek_p=if(uploch_p>=%s+sum_p,0,%s+sum_p-uploch_p) WHERE (%s+uploch_n+umenshen_n-nachislen_n-vozvrat-uploch_p)>=0 and date_srok = %s AND na2_code = %s and tin=%s", (saldo_all, saldo_all, saldo_all, saldo_all, saldo_tek_p, saldo_tek_p, saldo_all, current_date, na2_code1, tin,))
                        # cursor.execute("UPDATE nla SET uploch_p = if((%s+uploch_n+umenshen_n-nachislen_n-vozvrat)>=saldo_tek_p,saldo_tek_p,(%s+uploch_n+umenshen_n-nachislen_n-vozvrat)), saldo_tek_p=if(uploch_p=(%s+%s+pr_penya),0,(%s+sum_p+pr_penya-ifnull(uploch_p,0))) WHERE (%s+uploch_n+umenshen_n-nachislen_n-vozvrat-uploch_p)>=0 and date_srok = %s AND na2_code = %s", (saldo_all, saldo_all, saldo_tek_p, sum_p, saldo_tek_p, saldo_all, current_date, na2_code1))
                        cursor.execute("UPDATE nla SET saldo_all = if(date_srok=last_date_srok,(uploch_n+umenshen_n-nachislen_n-vozvrat-uploch_p),(%s+uploch_n+umenshen_n-nachislen_n-vozvrat-uploch_p)) WHERE date_srok = %s AND na2_code = %s and tin=%s", (saldo_all, current_date, na2_code1, tin, ))
                        cursor.execute("delete from loop_progress where last_date_srok!=%s and na2_code!=%s and tin=%s", (current_date, na2_code1, tin,))
                        cursor.execute("INSERT INTO loop_progress (tin, last_date_srok, na2_code, datetime) VALUES (%s, %s, %s, %s)", (tin, current_date, na2_code1,now.strftime('%Y-%m-%d %H:%M:%S')))
                    dbs.commit()
            time.sleep(10)

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