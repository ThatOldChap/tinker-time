from app import app
from flask import render_template
from random import random
import enum, pprint
import matplotlib.pyplot as plt
import numpy as np


starting_balance = 100000
risk_per_trade_dollars = 1000
risk_to_reward_ratio = 3

@app.route('/')
@app.route('/index')
def index():

    return render_template('layout.html')


def generateTradeRecord(starting_balance, risk_to_reward_ratio, risk_per_trade_dollars, num_trades, win_rate_percent):

    class TradeRecord(enum.Enum):
        WIN = "Win"
        LOSS = "Loss"

    win_value = risk_to_reward_ratio * risk_per_trade_dollars
    loss_value = risk_per_trade_dollars
    win_threshold = win_rate_percent / 100
    current_balance = starting_balance
    cumulative_pnl = 0

    record_header = ['Trade #', 'Win/Loss', 'R/R', 'P&L', 'Cumulatitive P&L', 'Balance']
    start_record = [0, '-', '-', '-', '-', starting_balance]    
    trade_record = [record_header]
    trade_record.append(start_record)

    for i in range(num_trades):

        trade_num = i +1

        if random() < win_threshold:
            cumulative_pnl += win_value
            current_balance += win_value
            trade_record.append([trade_num, TradeRecord.WIN.value, f'+{risk_to_reward_ratio}R', f'${win_value}', f'${cumulative_pnl}', current_balance])

        else:
            cumulative_pnl -= loss_value
            current_balance -= loss_value
            trade_record.append([trade_num, TradeRecord.LOSS.value, '-1R', f'-${loss_value}', f'${cumulative_pnl}', current_balance])
    
    return trade_record


def generateChartData(trade_record):

    x_values = []
    y_values = []

    # Remove the header row before processing
    trade_record.pop(0)

    for trade in trade_record:
        x_values.append(trade[0])
        y_values.append(trade[5])
    
    return [x_values, y_values]


def generateChart(x_values, y_values):

    x_points = np.array(x_values)
    y_points = np.array(y_values)

    plt.plot(x_points, y_points)
    plt.show()


