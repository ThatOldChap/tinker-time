import enum
import io, logging, sys
import pprint, urllib, base64
from random import random
import matplotlib.pyplot as plt
import numpy as np
from flask import render_template, request, jsonify

from app import app

# Helper enums
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

# Default Values
starting_balance = 100000
base_risk = 1000
risk_to_reward_ratio = 2
win_rate = 50
num_trades = 25
num_simulations = 1
loss_mgt = LossMgt.NONE
win_mgt = WinMgt.NONE       
max_losses = 1              # Max # of losses starts with 1
max_wins = 2                # Max # of wins < 2 results in never being able to increase risk with b2b wins
num_risk_levels = 2
compare_baseline = False

# Debugging
def get_pp():
    return pprint.PrettyPrinter(width="200")

# Setup a logger
#logging.basicConfig(stream=sys.stderr, level=logging.INFO)
logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
logging.getLogger('matplotlib.font_manager').disabled = True


# Initializations and Tracking Variables
last_record = None
last_altered_record = None
pp = get_pp()


@app.route('/')
@app.route('/index')
def index():

    plot = updatePlot(starting_balance, risk_to_reward_ratio, base_risk, num_trades, win_rate, num_simulations, \
                      loss_mgt, win_mgt, max_losses, max_wins, num_risk_levels, compare_baseline)

    return render_template('layout.html', plot = plot)


@app.route('/update_chart', methods=['POST'])
def update_chart():

    # Extract the request's form dictionary
    data = request.form.to_dict()
    logging.debug(pp.pprint(data))
    for key, value in data.items():        

        if key == 'startingBalance':     
            starting_balance = int(value)
        if key == 'riskToRewardRatio':           
            risk_to_reward_ratio = float(value)
        if key == 'baseRisk':            
            base_risk = float(value)
        if key == 'numTrades':   
            num_trades = int(value)
        if key == 'winRate':            
            win_rate = int(value)
        if key == 'numSimulations':
            num_simulations = int(value)
        if key == 'lossMgt':
            loss_mgt = LossMgt(int(value))
        if key == 'winMgt':
            win_mgt = WinMgt(int(value))
        if key == 'maxLosses':
            max_losses = int(value)
        if key == 'maxWins':
            max_wins = int(value)
        if key == 'numRiskLevels':
            num_risk_levels = int(value)
        if key == 'compareBaseline':
            if value == 'false':
                compare_baseline = False
            elif value == 'true':
                compare_baseline = True
            

    plot = updatePlot(starting_balance, risk_to_reward_ratio, base_risk, num_trades, win_rate, num_simulations, \
                      loss_mgt, win_mgt, max_losses, max_wins, num_risk_levels, compare_baseline)

    # Load the chart into the json response to update the image using jQuery
    response = {
        'message': f'Chart has successfully been updated',
        'plot': plot
    }

    return jsonify(response)


def updatePlot(starting_balance, risk_to_reward_ratio, base_risk, num_trades, win_rate, num_simulations, \
               loss_mgt, win_mgt, max_losses, max_wins, num_risk_levels, compare_baseline):

    plot = None

    for i in range(num_simulations):
        record = generateTradeRecord(starting_balance, risk_to_reward_ratio, base_risk, num_trades, win_rate, \
                                    loss_mgt, win_mgt, max_losses, max_wins, num_risk_levels)
        data = generateChartData(record, compare_baseline)

        if i == 0:
            plot = generateBaselineChart(data[0], data[1], True)
        else:
            plot = generateBaselineChart(data[0], data[1], False)

    return plot

# TODO: Add a setting for accounting for b/e trades so that the risk is unadjusted
def generateTradeRecord(starting_balance, risk_to_reward_ratio, base_risk, num_trades, win_rate,
                        loss_mgt, win_mgt, max_losses, max_wins, num_risk_levels):
      
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
    record_header = ['Trade #', 'Win/Loss', 'Risk (R)', 'P&L (R)', 'P&L ($)', 'Balance ($)']
    start_record = [0, '---', '---', '---', '---', starting_balance]    
    trade_record = [record_header]
    trade_record.append(start_record)

    for i in range(num_trades):

        # Start at trade #1
        trade_num = i +1

        # Trade is a TradeRecord.WIN
        if random() < win_threshold:
            current_trade_result = TradeRecord.WIN
            logging.debug(f'-----Trade {trade_num}: Win -----')

            # Initialize the prev_trade_result for comparison after the 1st trade
            if i == 0:
                prev_trade_result = TradeRecord.WIN
                num_consecutive_wins = 1
            
            else:
                # Reset the loss counter when a win is taken
                if prev_trade_result == TradeRecord.LOSS:
                    num_consecutive_losses = 0

                # Increase risk level every time a single win is achieved
                new_risk_level = increase_risk(current_risk_level)
                num_consecutive_wins += 1

            # Evaluate reducing risk if the max num of wins has been hit
            if max_wins_hit(num_consecutive_wins, max_wins):
                
                if win_mgt == WinMgt.REDUCE_RISK_AFTER_MAX_NUM_WINS:                    
                    new_risk_level = reduce_risk(current_risk_level, lowest_risk_level)

                elif win_mgt == WinMgt.LOWEST_RISK_AFTER_MAX_NUM_WINS:
                    new_risk_level = lowest_risk_level
                    logging.debug(f'Risk has been reduced to the lowest level: {new_risk_level}')

                # Reset win counter to 0 if the risk has been reduced
                # TODO: Review whether to start the counter on max wins at full risk
                num_consecutive_wins = 0

            # Calculates the trade results in $ and R/R adjusting for the current risk level
            adj_win_RR = risk_to_reward_ratio * current_risk_level.value
            adj_win_value = base_win_value * current_risk_level.value
            current_balance += adj_win_value

            # Log the trade in the record
            trade_record.append([trade_num, current_trade_result.value, f'{current_risk_level.value}R', \
                                 f'+{adj_win_RR}R', f'+{adj_win_value}', current_balance])

        # Trade is a TradeRecord.LOSS
        else:
            current_trade_result = TradeRecord.LOSS
            logging.debug(f'-----Trade {trade_num}: Loss -----')

            # Initialize the prev_trade_result for comparison after the 1st trade
            if i == 0:
                prev_trade_result = TradeRecord.LOSS
                num_consecutive_losses = 1
            
            else:
                # Reset the win counter when a loss is taken
                if prev_trade_result == TradeRecord.WIN:
                    num_consecutive_wins = 0

                num_consecutive_losses += 1

            # Evaluate reducing risk if the max num of losses has been hit
            if max_losses_hit(num_consecutive_losses, max_losses):

                if loss_mgt == LossMgt.REDUCE_RISK_AFTER_MAX_NUM_LOSSES:
                    new_risk_level = reduce_risk(current_risk_level, lowest_risk_level)

                elif loss_mgt == LossMgt.LOWEST_RISK_AFTER_MAX_NUM_LOSSES:
                    new_risk_level = lowest_risk_level
                    logging.debug(f'Risk has been reduced to the lowest level: {new_risk_level}')

            # Default to the same risk if no conditions are met
            else:
                new_risk_level = current_risk_level

            # Calculates the trade results in $ and R/R adjusting for the current risk level
            adj_loss_RR = -1 * current_risk_level.value
            adj_loss_value = base_risk * current_risk_level.value
            current_balance -= adj_loss_value
            
            # Log the trade in the record
            trade_record.append([trade_num, current_trade_result.value, f'{current_risk_level.value}R', \
                                 f'{adj_loss_RR}R', f'-{adj_loss_value}', current_balance])
    
        # Updates lookback variables
        current_risk_level = new_risk_level
        prev_trade_result = current_trade_result

    # Store the last record for any futher alterations
    last_record = trade_record
    logging.debug(pp.pprint(trade_record))

    return trade_record

def alterTradeRecord(existing_record, starting_balance, risk_to_reward_ratio, base_risk, loss_mgt, win_mgt, \
                     max_losses, max_wins, num_risk_levels):
    
    # Pre-Loop Calcs
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

    # Extract the first two header rows and remove them before looping through
    altered_record = [existing_record[0]]
    altered_record.append(existing_record[1])
    header = existing_record.pop(0)         # Remove header
    first = existing_record.pop(0)          # Remove trade 0

    # Loop through the existing trades rather than generate new ones
    for trade in existing_record:

        # Extract useful data from the existing record
        trade_num = trade[0]
        current_trade_result = TradeRecord(trade[1])

        # Trade is a TradeRecord.WIN
        if current_trade_result == TradeRecord.WIN:
            logging.debug(f'-----Trade {trade_num}: Win -----')

            # Initialize the prev_trade_result for comparison after the 1st trade
            if trade_num == 1:
                prev_trade_result = TradeRecord.WIN
                num_consecutive_wins = 1
        
            else:
                # Reset the loss counter when a win is taken
                if prev_trade_result == TradeRecord.LOSS:
                    num_consecutive_losses = 0

                # Increase risk level every time a single win is achieved
                new_risk_level = increase_risk(current_risk_level)
                num_consecutive_wins += 1

            # Evaluate reducing risk if the max num of wins has been hit
            if max_wins_hit(num_consecutive_wins, max_wins):
                
                if win_mgt == WinMgt.REDUCE_RISK_AFTER_MAX_NUM_WINS:                    
                    new_risk_level = reduce_risk(current_risk_level, lowest_risk_level)

                elif win_mgt == WinMgt.LOWEST_RISK_AFTER_MAX_NUM_WINS:
                    new_risk_level = lowest_risk_level
                    logging.debug(f'Risk has been reduced to the lowest level: {new_risk_level}')

                # Reset win counter to 0 if the risk has been reduced
                # TODO: Review whether to start the counter on max wins at full risk
                num_consecutive_wins = 0
            
            # Calculates the trade results in $ and R/R adjusting for the current risk level
            adj_win_RR = risk_to_reward_ratio * current_risk_level.value
            adj_win_value = base_win_value * current_risk_level.value
            current_balance += adj_win_value

            # Log the altered trade in the new record
            altered_record.append([trade_num, current_trade_result.value, f'{current_risk_level.value}R', \
                                 f'+{adj_win_RR}R', f'+{adj_win_value}', current_balance])

        # Trade is a TradeRecord.LOSS
        else:
            logging.debug(f'-----Trade {trade_num}: Loss -----')

            # Initialize the prev_trade_result for comparison after the 1st trade
            if trade_num == 1:
                prev_trade_result = TradeRecord.LOSS
                num_consecutive_losses = 1
            
            else:
                # Reset the win counter when a loss is taken
                if prev_trade_result == TradeRecord.WIN:
                    num_consecutive_wins = 0

                num_consecutive_losses += 1
            
            # Evaluate reducing risk if the max num of losses has been hit
            if max_losses_hit(num_consecutive_losses, max_losses):

                if loss_mgt == LossMgt.REDUCE_RISK_AFTER_MAX_NUM_LOSSES:
                    new_risk_level = reduce_risk(current_risk_level, lowest_risk_level)

                elif loss_mgt == LossMgt.LOWEST_RISK_AFTER_MAX_NUM_LOSSES:
                    new_risk_level = lowest_risk_level
                    logging.debug(f'Risk has been reduced to the lowest level: {new_risk_level}')

            # Default to the same risk if no conditions are met
            else:
                new_risk_level = current_risk_level

            # Calculates the trade results in $ and R/R adjusting for the current risk level
            adj_loss_RR = -1 * current_risk_level.value
            adj_loss_value = base_risk * current_risk_level.value
            current_balance -= adj_loss_value

            # Log the trade in the record
            altered_record.append([trade_num, current_trade_result.value, f'{current_risk_level.value}R', \
                                 f'{adj_loss_RR}R', f'-{adj_loss_value}', current_balance])
        
        # Updates lookback variables
        current_risk_level = new_risk_level
        prev_trade_result = current_trade_result
    
    # Store the last record for any futher alterations
    last_altered_record = altered_record
    logging.debug(pp.pprint(altered_record))

    # Add the header rows back into the existing record
    existing_record.insert(0, first)
    existing_record.insert(0, header)

    return altered_record


def merge_records(main_record, altered_record):

    # Format the new table
    header = ['Trade #', 'Win/Loss', 'Main Risk (R)', 'Main P&L (R)', 'Main P&L ($)', 'Main Balance ($)', '|', \
              'Adj Risk (R)', 'Adj P&L (R)', 'Adj P&L ($)', 'Adj Balance ($)']
    merged_record = [header]

    # Remove the header rows from each record
    main_record.pop(0)
    altered_record.pop(0)

    for r1, r2 in zip(main_record, altered_record):
        merged_record.append([r1[0], r1[1], r1[2], r1[3], r1[4], r1[5], '|', \
                              r2[2], r2[3], r2[4], r2[5]])
    
    return merged_record


def generateChartData(trade_record, compare_baseline):

    x_values = []
    y1_values = []
    y2_values = []

    # Remove the header row before processing
    trade_record.pop(0)

    for trade in trade_record:
        x_values.append(trade[0])
        y1_values.append(trade[5])
        if compare_baseline:
            y2_values.append(trade[10])
    
    return [x_values, y1_values, y2_values]


def generateBaselineChart(x_values, y_values, clear_plot):

    # Clear the chart to start with new data each time
    if clear_plot:
        plt.clf()

    x_points = np.array(x_values)
    y_points = np.array(y_values)

    plt.plot(x_points, y_points)
    plt.title('Equity Curve')
    plt.xlabel('# of Trades')
    plt.ylabel('Account Balance')

    plot = get_plot_img(plt)

    return plot


def generateCompareChart(x_values, y_values, clear_plot):
    pass



def get_plot_img(plt):
    img = io.BytesIO()
    plt.savefig(img, format = 'png')
    img.seek(0)
    plot = urllib.parse.quote(base64.b64encode(img.read()).decode())
    return plot


def max_losses_hit(num_consecutive_losses, max_losses):
    result = num_consecutive_losses >= max_losses
    if result:
        logging.debug(f'Max number of consecutive losses hit: {num_consecutive_losses}/{max_losses}')
    return result


def max_wins_hit(num_consecutive_wins, max_wins):
    result = num_consecutive_wins >= max_wins
    if result: 
        logging.debug(f'Max number of consecutive wins hit: {num_consecutive_wins}/{max_wins}')
    return result


def get_lowest_risk_level(num_risk_levels):
    if num_risk_levels == 2:
        return RiskLevel.HALF
    elif num_risk_levels == 3:
        return RiskLevel.QUARTER


# Function to reduce risk by a level (cut in half) based on the num of levels selected
def reduce_risk(current_risk_level, lowest_risk_level):

    if current_risk_level == RiskLevel.FULL:
        logging.debug('Reducing risk from Full to Half.')
        return RiskLevel.HALF
    
    if current_risk_level == RiskLevel.HALF and lowest_risk_level.value < RiskLevel.HALF.value:
        logging.debug('Reducing risk from Half to Quarter.')
        return RiskLevel.QUARTER
    
    else:
        # Return the same risk level if current_risk_level is the lowest_risk_level
        return current_risk_level
    
# Function to increase risk by a level (double) based on the num of levels selected
def increase_risk(current_risk_level):

    if current_risk_level == RiskLevel.FULL:
        logging.debug('Keep maintining risk at Full.')
        return RiskLevel.FULL

    elif current_risk_level == RiskLevel.HALF:
        logging.debug('Increasing risk from Half to Full.')
        return RiskLevel.FULL
    
    elif current_risk_level == RiskLevel.QUARTER:
        logging.debug('Increasing risk from Quarter to Half.')
        return RiskLevel.HALF