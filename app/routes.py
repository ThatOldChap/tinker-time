import enum, os
import io, logging, sys, openpyxl
import pprint, urllib, base64
from pathlib import Path
from random import random
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
import numpy as np
from flask import render_template, request, jsonify, send_from_directory, flash, url_for
from matplotlib import rcParams
from datetime import datetime
from time import strptime

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

# Initializing Parameters
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
risk_mgt_enabled = False

# Setup the directory to store any temp files
# basedir = '/tinker-time/app'
basedir = os.path.abspath(".")
tmpdir = basedir + '/app/tmp/'
print(f'basedir = {os.path.abspath(".")}')
print(f'tmpdir = {tmpdir}')
print(f'cwd = {Path.cwd()}')


# Sets the bounds for the plot so no labels are cutoff
rcParams.update({'figure.autolayout': True})

# Debugging
def get_pp():
    return pprint.PrettyPrinter(width="200")

# Setup a logger
#logging.basicConfig(stream=sys.stderr, level=logging.INFO)
logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
logging.getLogger('matplotlib.font_manager').disabled = True
pp = get_pp()


# Cache Variables
last_record = None
last_altered_record = None
last_merged_record = None
last_record_list = None
last_plot = None
last_compare_baseline = False


@app.route('/')
@app.route('/index')
def index():

    global last_plot, last_record, last_altered_record

    baseline = generateTradeRecord(starting_balance, risk_to_reward_ratio, base_risk, num_trades, win_rate, \
                                    loss_mgt, win_mgt, max_losses, max_wins, num_risk_levels)
    plot = generateBaselineChart(baseline)
    
    # Cache the appropriate variables
    last_plot = plot
    last_record = baseline
    last_altered_record = last_record
    
    return render_template('layout.html', plot = plot)


@app.route('/update_chart', methods=['POST'])
def update_chart():

    global last_plot, last_compare_baseline, last_record, last_altered_record, last_record_list
    baseline_has_updated = False    # This can probably be deleted

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
        if key == 'baselineHasUpdated':
            if value == 'false':
                baseline_has_updated = False
            elif value == 'true':
                baseline_has_updated = True

    # Logic checks
    risk_mgt_enabled = not (loss_mgt == LossMgt.NONE and win_mgt == WinMgt.NONE)
    compare_enabled = compare_baseline and not last_compare_baseline
    compare_disabled = not compare_baseline and last_compare_baseline

    if compare_enabled:
        # Create compare chart with new baseline and using the original baseline as the altered
        if risk_mgt_enabled:
            message = 'Compare switch has been enabled with a RiskMgt strategy being used'
            baseline = alterTradeRecord(last_record, starting_balance, risk_to_reward_ratio, base_risk, \
                                        LossMgt.NONE, WinMgt.NONE, max_losses, max_wins, num_risk_levels)
            #compared = last_record
            compared = alterTradeRecord(baseline, starting_balance, risk_to_reward_ratio, base_risk, loss_mgt, \
                                        win_mgt, max_losses, max_wins, num_risk_levels)
            plot = generateCompareChart(baseline, compared, compare_baseline)

            # Cache the appropriate records
            last_record = baseline
            last_altered_record = compared

        # Use the same baseline chart
        elif not risk_mgt_enabled:
            message = 'Compare switch has been enabled with no RiskMgt strategy being used'
            plot = last_plot
    
    elif compare_disabled:
        # Use prev altered record as new baseline data and create new baseline chart with a single series
        if risk_mgt_enabled:
            message = 'Compare switch has been disabled with a RiskMgt strategy still being used'
            baseline = last_altered_record
            plot = generateBaselineChart(baseline)

            # Cache the appropriate records
            last_record = baseline

        # No compare chart is active so existing chart should be the same baseline chart
        elif not risk_mgt_enabled:
            message = 'Compare switch has been disabled with no RiskMgt strategy being used'
            plot = last_plot
    
    elif compare_baseline:
        if baseline_has_updated:
            # If no risk_mgt strategy is used, keep generating new baseline charts with a single series
            if not risk_mgt_enabled:
                message = 'Compare Baseline is enabled with no RiskMgt strategy being used'
                baseline = generateTradeRecord(starting_balance, risk_to_reward_ratio, base_risk, num_trades, win_rate, \
                                                loss_mgt, win_mgt, max_losses, max_wins, num_risk_levels)
                plot = generateBaselineChart(baseline)

                # Cache the appropriate records
                last_record = baseline
            
            # Create compare chart with new baseline and new altered data
            elif risk_mgt_enabled:
                message = 'Compare Baseline is enabled with a RiskMgt strategy being used'
                baseline = generateTradeRecord(starting_balance, risk_to_reward_ratio, base_risk, num_trades, win_rate, \
                                                LossMgt.NONE, WinMgt.NONE, max_losses, max_wins, num_risk_levels)
                compared = alterTradeRecord(baseline, starting_balance, risk_to_reward_ratio, base_risk, loss_mgt, \
                                            win_mgt, max_losses, max_wins, num_risk_levels)
                plot = generateCompareChart(baseline, compared, compare_baseline)

                # Cache the appropriate records
                last_record = baseline
                last_altered_record = compared
            
        elif not baseline_has_updated:
            # Update compare chart with same baseline, new altered data
            if risk_mgt_enabled:
                message = 'Compare Baseline is enabled and a risk parameter has updated, and a RiskMgt strategy is being used'
                baseline = last_record
                compared = alterTradeRecord(baseline, starting_balance, risk_to_reward_ratio, base_risk, loss_mgt, \
                                            win_mgt, max_losses, max_wins, num_risk_levels)
                plot = generateCompareChart(baseline, compared, compare_baseline)

                # Cache the appropriate records
                last_record = baseline
                last_altered_record = compared
            
            # Create new baseline chart with same record
            elif not risk_mgt_enabled:
                message = 'Compared Baseline is enabled, and RiskMgt strategies have been disabled'
                baseline = last_record
                plot = generateBaselineChart(baseline)

                # Cache the appropriate records
                last_record = baseline

    elif not compare_baseline:
        # Revert to the baseline and re-apply the adjusted RiskMgt parameters to get the new chart
        if not baseline_has_updated and (num_simulations == 1):
            message = 'Compare Baseline is disabled, risk parameter adjusted with num_simulations = 1'  
            baseline = alterTradeRecord(last_record, starting_balance, risk_to_reward_ratio, base_risk, LossMgt.NONE, \
                                            WinMgt.NONE, max_losses, max_wins, num_risk_levels)
            new_baseline = alterTradeRecord(last_record, starting_balance, risk_to_reward_ratio, base_risk, loss_mgt, \
                                            win_mgt, max_losses, max_wins, num_risk_levels)
            plot = generateBaselineChart(new_baseline)
            
            # Cache the appropriate records
            last_record = new_baseline                

        # Create a new baseline chart if any Strategy parameters are updated
        #elif baseline_has_updated:
        else:
            message = 'Compare Baseline is disabled, strategy parameter adjusted'        
            baseline_list = []
            for i in range(num_simulations):
                baseline = generateTradeRecord(starting_balance, risk_to_reward_ratio, base_risk, num_trades, win_rate, \
                                                loss_mgt, win_mgt, max_losses, max_wins, num_risk_levels)
                baseline_list.append(baseline)

            plot = generateMultiBaselineChart(baseline_list, compare_baseline)

            # Cache the appropriate records
            last_record = baseline_list[-1]
            last_record_list = baseline_list

    
    else:
        message = 'Undetermined case'
        
    # Cache tracking variables 
    last_plot = plot
    last_compare_baseline = compare_baseline
    
    # Load the chart into the json response to update the image using jQuery
    logging.debug(message)
    response = {
        'message': message,
        'plot': plot
    }

    return jsonify(response)


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

    # Update the cached last_record with the new record
    """ global last_record
    last_record = trade_record """

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

    # Loop through the existing trades rather than generate new ones
    for trade in existing_record:

        # Skip the header row and Trade 0
        if trade[0] == 'Trade #' or trade[0] == 0:
            continue

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
    
    logging.debug(pp.pprint(altered_record))

    return altered_record


def merge_records(main_record, altered_record):

    global last_merged_record

    # Format the new table
    header = ['Trade #', 'Win/Loss', 'Main Risk (R)', 'Main P&L (R)', 'Main P&L ($)', 'Main Balance ($)', '|', \
              'Adj Risk (R)', 'Adj P&L (R)', 'Adj P&L ($)', 'Adj Balance ($)']
    merged_record = [header]

    for r1, r2 in zip(main_record, altered_record):
        # Skip the header row
        if r1[0] == 'Trade #':
            continue
        merged_record.append([r1[0], r1[1], r1[2], r1[3], r1[4], r1[5], '|', \
                              r2[2], r2[3], r2[4], r2[5]])
        
    # Cache the last merged record to prep for a downloaded file
    last_merged_record = merged_record
    logging.debug(pp.pprint(merged_record))
    
    return merged_record


def generateBaselineChart(baseline_record):

    global last_record

    # Get the data for the plot
    data = generateChartData(baseline_record, compare_baseline)

    # Clear the plot of any past series and create a new baseline chart
    plot = plotBaselineChart(data[0], data[1], True)

    return plot


def generateMultiBaselineChart(baseline_record_list, compare_baseline):

    global last_record
    
    for record in baseline_record_list:
        data = generateChartData(record, compare_baseline)

        # Clear the plot of any past series to start with a fresh chart
        if record == baseline_record_list[0]:
            plot = plotBaselineChart(data[0], data[1], True)
        else:
            plot = plotBaselineChart(data[0], data[1], False)

    return plot


def generateCompareChart(baseline_record, compared_record, compare_baseline):

    global last_record, last_altered_record

    # Get the data for the plot
    merged = merge_records(baseline_record, compared_record)
    data = generateChartData(merged, compare_baseline)

    # Clear the plot of any past compared series, and re-add the baseline and the new compared series
    plot = plotCompareChart(data[0], data[1], data[2], True)

    return plot


def generateChartData(trade_record, compare_baseline):

    x_values = []
    y1_values = []
    y2_values = []

    for trade in trade_record:

        # Skip Header row
        if trade[0] == "Trade #":
            continue

        x_values.append(trade[0])
        y1_values.append(trade[5])
        if compare_baseline:
            y2_values.append(trade[10])
    
    return [x_values, y1_values, y2_values]


def plotBaselineChart(x_values, y_values, clear_plot):

    # Clear the chart to start with new data each time
    if clear_plot:
        plt.clf()

    x_points = np.array(x_values)
    y_points = np.array(y_values)

    # Setup the format for the chart axis and plot it
    y_format = tkr.FuncFormatter(yAxisFormatter)
    ax = plt.subplot(111)
    ax.plot(x_points, y_points)
    ax.yaxis.set_major_formatter(y_format)

    # Format the chart details
    plt.title('Equity Curve')
    plt.xlabel('# of Trades')
    plt.ylabel('Account Balance ($)')

    plot = get_plot_img(plt)

    return plot


def plotCompareChart(x_values, y1_values, y2_values, clear_plot):

    # Clear the chart to start with new data each time
    if clear_plot:
        plt.clf()

    x_points = np.array(x_values)
    y1_points = np.array(y1_values)
    y2_points = np.array(y2_values)

    # Setup the format for the chart axis and plot it
    y_format = tkr.FuncFormatter(yAxisFormatter)
    ax = plt.subplot(111)
    ax.plot(x_points, y1_points, y2_points, color='tab:blue')
    ax.yaxis.set_major_formatter(y_format)

    plt.plot(x_points, y1_points, label='Baseline')
    plt.plot(x_points, y2_points, label='w/ Risk Mgt')
    plt.title('Equity Curve - Baseline vs. Risk Management')
    plt.legend(loc="upper left")
    plt.xlabel('# of Trades')
    plt.ylabel('Account Balance')

    plot = get_plot_img(plt)

    return plot

def yAxisFormatter(x, pos):
   s = '{:0,d}'.format(int(x))
   return s


def get_plot_img(plt):
    img = io.BytesIO()
    plt.savefig(img, format = 'png', dpi=200)
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
    

@app.route('/download_record', methods=['GET', 'POST'])
def download_record():

    global last_record, last_record_list, last_merged_record, last_compare_baseline

    # Delete the last file(s) that were sent before generating a new one
    for filename in os.listdir(tmpdir):
        if os.path.isfile(os.path.join(tmpdir, filename)):
            logging.debug(f'Deleting old file: {filename}')
            os.remove(os.path.join(tmpdir, filename))
    
    def add_baseline_trade_row(sheet, row_num, trade):
        sheet[f'A{row_num}'] = trade[0]
        sheet[f'B{row_num}'] = trade[1]
        sheet[f'C{row_num}'] = trade[2]
        sheet[f'D{row_num}'] = trade[3]
        sheet[f'E{row_num}'] = trade[4]
        sheet[f'F{row_num}'] = trade[5]

        return row_num + 1
    
    def add_merged_trade_row(sheet, row_num, trade):
        sheet[f'A{row_num}'] = trade[0]
        sheet[f'B{row_num}'] = trade[1]
        sheet[f'C{row_num}'] = trade[2]
        sheet[f'D{row_num}'] = trade[3]
        sheet[f'E{row_num}'] = trade[4]
        sheet[f'F{row_num}'] = trade[5]
        # Skip the "|" that was used just for the console printouts
        sheet[f'G{row_num}'] = trade[7]
        sheet[f'H{row_num}'] = trade[8]
        sheet[f'I{row_num}'] = trade[9]
        sheet[f'J{row_num}'] = trade[10]

        return row_num + 1
    
    # Setup the workbook
    wb = openpyxl.Workbook()
    sheet = wb.active
    sheet.title = 'Trade Record'
    sheet = wb[sheet.title]

    row_num = 1

    if last_compare_baseline:
        logging.debug(pp.pprint(last_merged_record))
        for trade in last_merged_record:
            row_num = add_merged_trade_row(sheet, row_num, trade)
    else:
        logging.debug(pp.pprint(last_record))
        for trade in last_record:
            row_num = add_baseline_trade_row(sheet, row_num, trade)
        
    # Save the file to the tmp directory
    filename = 'trade_record_' + datetime.now().strftime("%m-%d-%Y_%H%M%S") + '.xlsx'
    logging.debug(f'Saving {tmpdir + filename}...')
    wb.save(tmpdir + filename)
    wb.close()

    # Send the file to the user
    try:
        return send_from_directory(directory=tmpdir, path=filename, as_attachment=True, max_age=0)
    except FileNotFoundError:
        pass
   