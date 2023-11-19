from flask import Flask
from flask_bootstrap import Bootstrap4

app = Flask(__name__)
bootstrap = Bootstrap4(app)

from . import routes