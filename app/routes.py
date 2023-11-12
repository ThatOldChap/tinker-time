import enum
import io
import pprint, urllib, base64
from random import random
import matplotlib.pyplot as plt
import numpy as np
from flask import render_template, request, jsonify

from app import app

# Default Values
starting_balance = 100000
base_risk = 1000
risk_to_reward_ratio = 2
win_rate = 50
num_trades = 25
num_simulations = 1


@app.route('/')
@app.route('/index')
def index():

    plot = updatePlot(starting_balance, risk_to_reward_ratio, base_risk, num_trades, win_rate, num_simulations)

    return render_template('layout.html', plot = plot)


@app.route('/update_chart', methods=['POST'])
def update_chart():

    # Extract the request's form dictionary
    data = request.form.to_dict()
    print(data)
    for key, value in data.items():        

        if key == 'startingBalance':     
            starting_balance = int(value)
        if key == 'baseRisk':            
            base_risk = float(value)
        if key == 'riskToRewardRatio':           
            risk_to_reward_ratio = float(value)
        if key == 'winRate':            
            win_rate = int(value)
        if key == 'numTrades':   
            num_trades = int(value)
        if key == 'numSimulations':
            num_simulations = int(value)

    plot = updatePlot(starting_balance, risk_to_reward_ratio, base_risk, num_trades, win_rate, num_simulations)

    # Load the chart into the json response to update the image using jQuery
    response = {
        'message': f'Chart has successfully been updated',
        'plot': plot
    }

    return jsonify(response)


def updatePlot(starting_balance, risk_to_reward_ratio, base_risk, num_trades, win_rate, num_simulations):

    plot = None

    for i in range(num_simulations):
        record = generateTradeRecord(starting_balance, risk_to_reward_ratio, base_risk, num_trades, win_rate)
        data = generateChartData(record)

        if i == 0:
            plot = generateChart(data[0], data[1], True)
        else:
            plot = generateChart(data[0], data[1], False)

    return plot


def generateTradeRecord(starting_balance, risk_to_reward_ratio, base_risk, num_trades, win_rate):

    class TradeRecord(enum.Enum):
        WIN = "Win"
        LOSS = "Loss"

    win_value = risk_to_reward_ratio * base_risk
    loss_value = base_risk
    win_threshold = win_rate / 100
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


def generateChart(x_values, y_values, clear_plot):

    # Clear the chart to start with new data each time
    if clear_plot:
        plt.clf()

    x_points = np.array(x_values)
    y_points = np.array(y_values)

    plt.plot(x_points, y_points)
    plt.title('Equity Curve')
    plt.xlabel('# of Trades')
    plt.ylabel('Account Balance')

    img = io.BytesIO()
    plt.savefig(img, format = 'png')
    img.seek(0)
    plot = urllib.parse.quote(base64.b64encode(img.read()).decode())

    return plot