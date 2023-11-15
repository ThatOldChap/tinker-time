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

# Debugging
def get_pp():
    return pprint.PrettyPrinter()


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

# TODO: Add a setting for starting level of risk (lowest or highest)
def generateTradeRecord(starting_balance, risk_to_reward_ratio, base_risk, num_trades, win_rate,
                        loss_mgt, win_mgt, max_losses, max_wins, num_risk_levels):
    
    def max_losses_hit(num_consecutive_losses, max_losses):
        result = num_consecutive_losses >= max_losses
        if result:
            print(f'Max number of consecutive wins hit: {num_consecutive_losses}/{max_losses}')
        return result

    def max_wins_hit(num_consecutive_wins, max_wins):
        result = num_consecutive_wins >= max_wins
        if result: 
            print(f'Max number of consecutive wins hit: {num_consecutive_wins}/{max_wins}')
        return result
    
    def get_lowest_risk_level(num_risk_levels):
        if num_risk_levels == 2:
            return RiskLevel.HALF
        elif num_risk_levels == 3:
            return RiskLevel.QUARTER
    
    # Function to reduce risk by a level (cut in half) based on the num of levels selected
    def reduce_risk(current_risk_level, lowest_risk_level):

        if current_risk_level == RiskLevel.FULL:
            print('Reducing risk from Full to Half.')
            return RiskLevel.HALF
        
        if current_risk_level == RiskLevel.HALF and lowest_risk_level.value < RiskLevel.HALF.value:
            print('Reducing risk from Half to Quarter.')
            return RiskLevel.QUARTER
        
        else:
            # Return the same risk level if current_risk_level is the lowest_risk_level
            return current_risk_level
        
    # Function to increase risk by a level (double) based on the num of levels selected
    def increase_risk(current_risk_level):

        if current_risk_level == RiskLevel.FULL:
            print('Keep maintining risk at Full.')
            return RiskLevel.FULL

        elif current_risk_level == RiskLevel.HALF:
            print('Increasing risk from Half to Full.')
            return RiskLevel.FULL
        
        elif current_risk_level == RiskLevel.QUARTER:
            print('Increasing risk from Quarter to Half.')
            return RiskLevel.HALF

    # Pre-Loop Calcs
    win_threshold = win_rate / 100
    base_win_value = risk_to_reward_ratio * base_risk
    adj_win_RR = risk_to_reward_ratio
    adj_loss_RR = -1

    # Initializations
    current_balance = starting_balance 
    adj_win_value = base_win_value

    num_consecutive_losses = 0
    num_consecutive_wins = 0

    current_risk_level = RiskLevel.FULL
    new_risk_level = RiskLevel.FULL
    lowest_risk_level = get_lowest_risk_level(num_risk_levels)

    prev_trade_result = None
    current_trade_result = None

    # Table Formatting
    record_header = ['Trade #', 'Win/Loss', 'R/R', 'P&L ($)', 'Balance ($)']
    start_record = [0, '-', '-', '-', starting_balance]    
    trade_record = [record_header]
    trade_record.append(start_record)

    for i in range(num_trades):

        # Start at trade #1
        trade_num = i +1

        # Trade is a Win
        if random() < win_threshold:
            current_trade_result = TradeRecord.WIN
            print(f'-----Trade {trade_num}: Win -----')

            # Initialize the prev_trade_result for comparison after the 1st trade
            if i == 0:
                prev_trade_result = TradeRecord.WIN
                num_consecutive_wins = 1
            
            else:
                # Reset the loss counter when a win is taken
                if prev_trade_result == TradeRecord.LOSS:
                    num_consecutive_losses = 0

                    # Increase risk by a level if the previous trade was a Loss
                    new_risk_level = increase_risk(current_risk_level)
                
                # Increase risk level if previously reduced with a WinMgt strategy
                elif prev_trade_result == TradeRecord.WIN and current_risk_level.value < 1:
                    new_risk_level = increase_risk(current_risk_level)

                num_consecutive_wins += 1

            # No Win Management strategy used so keep the same risk
            if win_mgt == WinMgt.NONE:
                new_risk_level = current_risk_level
                print('No WinMgt strategy used, maintain current risk.')

            # Evaluate reducing risk if the max num of wins has been hit
            elif max_wins_hit(num_consecutive_wins, max_wins):
                
                if win_mgt == WinMgt.REDUCE_RISK_AFTER_MAX_NUM_WINS:                    
                    new_risk_level = reduce_risk(current_risk_level, lowest_risk_level)

                elif win_mgt == WinMgt.LOWEST_RISK_AFTER_MAX_NUM_WINS:
                    new_risk_level = lowest_risk_level
                    print(f'Risk has been reduced to the lowest level: {new_risk_level}')

                # Reset win counter to 0 if the risk has been reduced
                num_consecutive_wins = 0

            # Calculates the trade results in $ and R/R adjusting for the current risk level
            adj_win_RR = risk_to_reward_ratio * current_risk_level.value
            adj_win_value = base_win_value * current_risk_level.value
            current_balance += adj_win_value

            # Log the trade in the record
            trade_record.append([trade_num, current_trade_result.value, f'+{adj_win_RR}R', f'+{adj_win_value}', current_balance])

        # Trade is a Loss
        else:
            current_trade_result = TradeRecord.LOSS
            print(f'-----Trade {trade_num}: Loss -----')

            # Initialize the prev_trade_result for comparison after the 1st trade
            if i == 0:
                prev_trade_result = TradeRecord.LOSS
                num_consecutive_losses = 1
            
            else:
                # Reset the win counter when a loss is taken
                if prev_trade_result == TradeRecord.WIN:
                    num_consecutive_wins = 0

                num_consecutive_losses += 1

            # No Loss Management strategy used so keep the same risk
            if loss_mgt == LossMgt.NONE:
                new_risk_level = current_risk_level
                print('No Loss strategy used, maintain current risk.')

            # Evaluate reducing risk if the max num of losses has been hit
            elif max_losses_hit(num_consecutive_losses, max_losses):

                if loss_mgt == LossMgt.REDUCE_RISK_AFTER_MAX_NUM_LOSSES:
                    new_risk_level = reduce_risk(current_risk_level, num_risk_levels)

                elif loss_mgt == LossMgt.LOWEST_RISK_AFTER_MAX_NUM_LOSSES:
                    new_risk_level = lowest_risk_level
                    print(f'Risk has been reduced to the lowest level: {new_risk_level}')

            # Default to the same risk if no conditions are met
            else:
                new_risk_level = current_risk_level

            # Calculates the trade results in $ and R/R adjusting for the current risk level
            adj_loss_RR = -1 * current_risk_level.value
            adj_loss_value = base_risk * current_risk_level.value
            current_balance -= adj_loss_value
            
            # Log the trade in the record
            trade_record.append([trade_num, current_trade_result.value, f'{adj_loss_RR}R', f'-{adj_loss_value}', current_balance])
    
        # Updates lookback variables
        current_risk_level = new_risk_level
        prev_trade_result = current_trade_result

    return trade_record


class LossMgt(enum.Enum):
    NONE = 0
    REDUCE_RISK_AFTER_MAX_NUM_LOSSES = 1
    LOWEST_RISK_AFTER_MAX_NUM_LOSSES = 2

class WinMgt(enum.Enum):
    NONE = 0
    REDUCE_RISK_AFTER_MAX_NUM_WINS = 1
    LOWEST_RISK_AFTER_MAX_NUM_WINS = 2
    
class RiskLevel(enum.Enum):
    FULL = 1
    HALF = 0.5
    QUARTER = 0.25

class TradeRecord(enum.Enum):
    WIN = "Win"
    LOSS = "Loss"

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