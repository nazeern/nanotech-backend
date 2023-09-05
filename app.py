from flask import Flask
from funcs import *

app = Flask(__name__)

@app.route("/")
def hello_world():
    return test_me()