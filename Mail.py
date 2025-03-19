from flask import *
from flask_mail import *
from random import *
app=Flask(__name__)
mail=mail(app)
app.configp["MAIL_SERVER"]='smtp.gmail.com'
app.config["MAIL_PORT"]=465
app.config["MAIL_USERNAME"]='hbncompanyofficials@gmail.com'
app.config['MAIL_PASSWORD']=''
app.config['MAIL_USE_TLS']=False
app.config['MAIL_USE_SSL']=True
mail=Mail(app)
otp=randint(000000,999999)
@route('/')
def index():
    return render_template("index.html")
@app.route('/verify', methods=["POST"])
def verify():
    email=request.form["email"]
    msg=Message('OTP', sender='hbncompanyofficials@gmail.com', recipents=[email])
    msg.body=str(otp)
    mail.send(msg)
    return render_template("verify.html")
@app.route('/validate', methods=["POST"])
def validate():
    user_otp=request.form['otp']
    if otp==int(user_otp):
        return "Succes"
    else:
        return "Fail"