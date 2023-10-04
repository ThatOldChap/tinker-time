from app import app
from flask import render_template
from random import random

@app.route('/')
@app.route('/index')
def index():

    return render_template('layout.html')


def generateWinLossRecord(num_trades, win_rate_percent):

    # Make Trade.WIN Enum

    win_threshold = win_rate_percent / 100
    trade_record = []

    for i in num_trades:
        if random() < win_threshold:
            trade_record.append()

