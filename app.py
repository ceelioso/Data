import warnings
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import init_notebook_mode
from flask import Flask, Response, current_app, flash, redirect, render_template, request, session, url_for
from datetime import datetime
import time
import json
import re
import os
import MySQLdb
import dash
import dash_bootstrap_components as dbc
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
from werkzeug.security import generate_password_hash, check_password_hash
from io import BytesIO
import poetry
from random import random
import poetry

# Suppress warnings
warnings.filterwarnings("ignore")

# Initialize Flask app
app_flask = Flask(__name__)
app_flask.secret_key = b'_5#y2L"F4Q8z\n\xec]/'

# Define database connection function
def connect_db():
    return MySQLdb.connect(host='localhost', user='your_username', passwd='your_password', db='your_db')

# Define login route
@app_flask.route('/login', methods=['POST'])
def login():
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form:
        username = request.form['username']
        password = request.form['password']
        db = connect_db()
        cursor = db.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM accounts WHERE username = %s', (username,))
        account = cursor.fetchone()
        if account and check_password_hash(account['password'], password):
            session['loggedin'] = True
            session['id'] = account['id']
            session['username'] = account['username']
            return 'Logged in successfully!'
        else:
            flash('Incorrect username/password!', 'error')
    return render_template('index.html')

# Define register route
@app_flask.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form and 'email' in request.form:
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']
        if not (username and password and email):
            flash('Please fill out all fields!', 'error')
        else:
            db = connect_db()
            cursor = db.cursor(MySQLdb.cursors.DictCursor)
            cursor.execute('SELECT * FROM accounts WHERE username = %s', (username,))
            account = cursor.fetchone()
            if account:
                flash('Account already exists!', 'error')
            elif not re.match(r'[^@]+@[^@]+\.[^@]+', email):
                flash('Invalid email address!', 'error')
            else:
                hashed_password = generate_password_hash(password)
                cursor.execute('INSERT INTO accounts (username, password, email) VALUES (%s, %s, %s)', (username, hashed_password, email))
                db.commit()
                flash('You have successfully registered!', 'success')
                return redirect(url_for('login'))
    return render_template('register.html')

# Define profile route
@app_flask.route('/profile')
def profile():
    if 'loggedin' in session:
        db = connect_db()
        cursor = db.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM accounts WHERE id = %s', (session['id'],))
        account = cursor.fetchone()
        return render_template('profile.html', account=account)
    return redirect(url_for('login'))

# Define home route
@app_flask.route('/home')
def home():
    if 'loggedin' in session:
        return render_template('home.html', username=session['username'])
    return redirect(url_for('login'))

# Define logout route
@app_flask.route('/logout')
def logout():
    session.pop('loggedin', None)
    session.pop('id', None)
    session.pop('username', None)
    return redirect(url_for('login'))


# Initialize Dash app
app_dash = dash.Dash(__name__)

# Define app layout
app_dash.layout = html.Div([
    dcc.Dropdown(
        id='dropdown',
        options=[{'label': i, 'value': i} for i in ['Option 1', 'Option 2', 'Option 3']],
        value='Option 1'
    ),
    html.Div(id='output-container')
])

# Define callback functions
@app_dash.callback(
    Output('output-container', 'children'),
    [Input('dropdown', 'value')]
)
def update_output(value):
    return 'You have selected "{}"'.format(value)

# Main function
def run_apps():
    # Run Flask and Dash apps
    app_flask.run(debug=True)
    app_dash.run_server(debug=True)

# Entry point of the script
if __name__ == "__main__":
    # Call the run_apps function
    run_apps()

# Initialize Plotly notebook mode
init_notebook_mode(connected=True)

# Define Plotly template
temp = {
    "layout": {
        "font": {
            "family": "Franklin Gothic",
            "size": 12
        },
        "width": 800
    },
    "colors": px.colors.qualitative.Plotly
}

train_df = pd.read_csv(r"data/train_data.csv", parse_dates=['Date'])
stock_list = pd.read_csv(r"data/temp_data.csv")

# Display data information
print("The training data begins on {} and ends on {}.\n".format(train_df.Date.min(), train_df.Date.max()))
print(train_df.describe().style.format('{:,.2f}'))

# Initialize Dash app
# Initialize Dash app
app_dash = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

def callback_graph(selected_dropdown_value):
    # Check if the selected value is a valid column name in train_df
    if selected_dropdown_value in train_df.columns:
        # Create a figure based on user input
        fig = px.line(train_df, x="Date", y=selected_dropdown_value, color_discrete_sequence=temp["colors"])
        fig.update_layout(**temp["layout"])
        return fig
    else:
        # If the selected value is not a valid column name, you can handle the error or choose a default column
        # Here, I'm choosing the 'Close' column as a default option
        fig = px.line(train_df, x="Date", y="Close", color_discrete_sequence=temp["colors"])
        fig.update_layout(**temp["layout"])
        return fig

def main_app():
    # Run Flask and Dash apps
    app_flask.run(debug=True)
    app_dash.run_server(debug=True)

# Entry point of the script
if __name__ == "__main__":
    # Call the main_app function
    main_app()

# Initialize Plotly notebook mode
init_notebook_mode(connected=True)

# Define Plotly template
temp = {
    "layout": {
        "font": {
            "family": "Franklin Gothic",
            "size": 12
        },
        "width": 800
    },
    "colors": px.colors.qualitative.Plotly
}

# Display data information
print("The training data begins on {} and ends on {}.\n".format(train_df.Date.min(), train_df.Date.max()))
print(train_df.describe().style.format('{:,.2f}'))


# Initialize Dash app
app = dash.Dash(__name__, 
                external_scripts=['https://cdn.plot.ly/plotly-2.30.0.min.js'],
                suppress_callback_exceptions=True,
                external_stylesheets=[dbc.themes.BOOTSTRAP])

# Define app layout
app.layout = dbc.Container([
    dcc.Dropdown(
        id='dropdown',
        options=[{'label': i, 'value': i} for i in train_df.columns],
        value='Target'  # Set the default value of the dropdown
    ),
    dcc.Graph(id='example-graph'),   # Add other components as needed
])

# Define callback functions
@app.callback(
    Output('example-graph', 'figure'),
    [Input('dropdown', 'value')]
)
def update_graph(selected_dropdown_value):
    # Create a figure based on user input
    fig = px.line(train_df, x="Date", y=selected_dropdown_value, color_discrete_sequence=temp["colors"])
    fig.update_layout(**temp["layout"])
    return fig

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)

# Import necessary modules
import time

# Example code snippet with a function that simulates an error
def example_function():
    # Simulate an error by dividing by zero
    x = 1 / 0

# Main function
def main():
    try:
        # Call the function causing the error
        example_function()
    except Exception as e:
        # Print the exception traceback
        print("Exception occurred:", e)

        # Add a breakpoint here if you want to inspect variables

        # Add a delay to allow time to inspect variables before the program exits
        time.sleep(10)

# Entry point of the script
if __name__ == "__main__":
    # Call the main function
    main()



# Display columns present in train_df
print("Columns present in train_df:", train_df.columns)

# Check if the 'Name' column exists in the DataFrame
if 'Name' in train_df.columns:
    # Group by the 'Name' column and calculate the mean of the 'Target' column
    stock = train_df.groupby('Name')['Target'].mean().mul(100)

    # Print or use the 'stock' data as needed
    print(stock)

    # Proceed with further processing or visualization  
else:
    # Alternatively, you can use a different approach if 'Name' column is not found
    print("Error: 'Name' column not found in train_df.")

# Check if the '17SectorName' column exists in the DataFrame  
if '17SectorName' in train_df.columns:
    train_df['SectorName'] = train_df['17SectorName'].str.rstrip().str.lower().str.capitalize()
    train_df['Name'] = [i.rstrip().lower().capitalize() for i in train_df['Name']]
    train_df['Year'] = train_df['Date'].dt.year
    train_df['result_df'] = pd.to_datetime(train_df['result_df'])
else:
    print("Error: '17SectorName' column not found in train_df.")

# Display the result DataFrame
result_df = pd.DataFrame()

# Iterate over unique years  
for year, group in train_df.groupby(train_df['Date'].dt.year):
    df_temp = train_df[train_df['Date'].dt.year == year]

    # Update the column name to 'SectorName'
    try:
        avg_return = df_temp.groupby('SectorName')['Target'].mean() 
    except KeyError:
        avg_return = df_temp.groupby('17SectorName')['Target'].mean()
        result_df = result_df.append(avg_return, ignore_index=True) # type: ignore


print(result_df)
print(df_temp.columns)

# Iterate over columns of the result DataFrame
for i, col in enumerate(result_df.columns):
    print(f"Column {i+1}: {col}")

# Further processing...

# Now you can use the 'stock' variable further in your code
stock = train_df.groupby('Name')['Target'].mean().mul(100)
stock_low = stock.nsmallest(7)[::-1].rename("Return")
print(stock_low)

fig = make_subplots(rows=1, cols=5, shared_yaxes=True)
for i, col in enumerate(result_df.columns):
    x = train_df[col]
    mask = x <= 0
    fig.add_trace(go.Bar(x=x[mask], y=train_df.index[mask], orientation='h', 
                         text=x[mask], texttemplate='%{text:.2f}%', textposition='auto',
                         hovertemplate='Average Return in %{y} Stocks = %{x:.4f}%',
                         marker=dict(color='red', opacity=0.7), name=col[-4:]), 
                  row=1, col=i+1)
    fig.add_trace(go.Bar(x=x[~mask], y=result_df.index[~mask], orientation='h', 
                         text=x[~mask], texttemplate='%{text:.2f}%', textposition='auto', 
                         hovertemplate='Average Return in %{y} Stocks = %{x:.4f}%',
                         marker=dict(color='green', opacity=0.7), name=col[-4:]), 
                  row=1, col=i+1)
    fig.update_xaxes(range=(x.min() - .15, x.max() + .15), title='{} Returns'.format(col[-4:]), 
                     showticklabels=False, row=1, col=i+1)
fig.update_layout(template=temp, title='Yearly Average Stock Returns by Sector', 
                  hovermode='closest', margin=dict(l=250, r=50),
                  height=600, width=1000, showlegend=False)
fig.show()

train_df = train_df[train_df.Date > '2024-12-23']

print("result_df:", train_df.columns)

print("Column Names:", train_df.columns)

x_hist = []

if 'Target' in train_df.columns:
    # Define x_hist
    x_hist = train_df['Target']

    # Create histogram
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=x_hist * 100, bins=dict(start=min(x_hist), end=max(x_hist), size=1)))

    # Update layout and show figure
    fig.update_layout(title_text='Distribution of Target Variable',
                      xaxis_title_text='Target',
                      yaxis_title_text='Frequency')

    fig.show()
else:
    print("Error: 'Target' column not found in the DataFrame.")

print(train_df)
print(train_df.dtypes)

fig.add_trace(go.Histogram(x=x_hist * 100,
                           marker=dict(opacity=0.7, 
                                       line=dict(width=1)),
                           xbins=dict(start=-40, end=40, size=1)))
fig.update_layout(template=temp, title='Target Distribution',
                  xaxis=dict(title='Stock Return', ticksuffix='%'), height=450)
fig.show()

pal = ['hsl(' + str(h) + ',50%' + ',50%)' for h in np.linspace(0, 360, 18)]
fig = go.Figure()
for i, sector in enumerate(result_df.index[::-1]):
    y_data = train_df[train_df['SectorName'] == sector]['Target']
    fig.add_trace(go.Box(y=y_data * 100, name=sector,
                         marker_color=pal[i], showlegend=False))
fig.update_layout(template=temp, title='Target Distribution by Sector',
                  yaxis=dict(title='Stock Return', ticksuffix='%'),
                  margin=dict(b=150), height=750, width=900)
fig.show()

train_date = train_df.Date.unique()
sectors = train_df.columns.unique().tolist()
sectors = []
sectors.insert(0, 'All')
open_avg = train_df.groupby('Date')['Open'].mean()
high_avg = train_df.groupby('Date')['High'].mean()
low_avg = train_df.groupby('Date')['Low'].mean()
close_avg = train_df.groupby('Date')['Close'].mean() 
buttons = []
print(train_df.columns)

# Check if the 'Name' column exists in the DataFrame
if 'Name' in train_df.columns:
    # Group by the 'Name' column and calculate the mean of the 'Target' column
    stock = train_df.groupby('Name')['Target'].mean().mul(100)

    # Print or use the 'stock' data as needed
    print(stock)

    # Proceed with further processing or visualization
else:
    # Alternatively, you can use a different approach if 'Name' column is not found
    print("Error: 'Name' column not found in train_df.")
if sectors:
    # Iterate over sector names
    for sector_name in sectors:
        # Check if 'SectorName' is present in the DataFrame
        if 'SectorName' in train_df.columns:
            # Filter the DataFrame based on the current sector_name
            sector_df = train_df[train_df['SectorName'] == sector_name]

            # Check if there are any rows in the filtered DataFrame
            if not sector_df.empty:
                # Perform your desired operations on sector_df
                open_avg = sector_df.groupby('Date')['Open'].mean()
            else:
                print(f"No data found for sector: {sector_name}")
        else:
            print("Error: 'SectorName' column not found in the DataFrame.")
else:
    print("Error: 'sectors' list is empty.")

sectors = ["Sector1", "Sector2", "Sector3"]  # Replace with your actual sectors

# Assuming train_df is your DataFrame
train_df = pd.DataFrame({"Date": ["2022-01-01", "2022-01-01", "2022-01-02"],
                         "Open": [100, 110, 95],
                         "SectorName": ["Sector1", "Sector2", "Sector1"]})

# Make sure to reset the index for correct indexing
train_df.reset_index(drop=True, inplace=True)

# Iterate through sectors
for sector in sectors:
    # Filter the DataFrame based on the current sector
    sector_data = train_df[train_df['SectorName'] == sector]

    # Calculate the mean open value for each date
    open_avg = sector_data.groupby('Date')['Open'].mean()

    # Print or use the open_avg data as needed
    print(f"Open average for {sector}:\n{open_avg}\n")

train_df = pd.DataFrame({"Date": ["2024-01-31", "2024-01-31", "2024-12-31"],
                         "Open": [100, 110, 95],
                         "High": [105, 115, 100],  # Adjust column names as needed
                         "SectorName": ["Sector1", "Sector2", "Sector1"]})

# Make sure to reset the index for correct indexing
train_df.reset_index(drop=True, inplace=True)

# Iterate through sectors
fig = go.Figure()

fig = go.Figure()

stock = train_df.groupby('Name')['Target'].mean().mul(100)

for sector in sectors:
    # Filter DataFrame for the current sector
    sector_df = train_df[train_df['SectorName'] == sector]

    # Check if 'Low' column exists in the DataFrame
    if 'Low' in sector_df.columns:
        # Calculate the average opening price for the current sector
        open_avg = sector_df.groupby('Date')['Open'].mean()
        high_avg = sector_df.groupby('Date')['High'].mean()
        low_avg = sector_df.groupby('Date')['Low'].mean()
        close_avg = sector_df.groupby('Date')['Close'].mean()

        # Print the average opening price for the current sector
        print(f"Average opening price for {sector}: {open_avg.mean()}")

        # Add a Candlestick trace for the current sector
        fig.add_trace(go.Candlestick(x=open_avg.index, open=open_avg.values,
                                     high=high_avg.values, low=low_avg.values,
                                     close=close_avg.values, name=sector))
    else:
        print(f"Low column not found for {sector}")

# Update the layout to show only the first sector initially
fig.update_layout(
    updatemenus=[dict(
        active=0,
        buttons=[
            dict(label=sector_name, method="update",
                args=[{"visible": [sector_name == np.trace.name]}]) 
            for sector_name in sectors
        ]
    )]
)




# Show the figure
fig.show()

fig.update_xaxes(rangeslider_visible=True,
                 rangeselector=dict(
                     buttons=list([
                         dict(count=3, label="3m", step="month", stepmode="backward"),
                         dict(count=6, label="6m", step="month", stepmode="backward"),
                         dict(step="all")]), xanchor='left',yanchor='bottom', y=1.16, x=.01))
fig.update_layout(template=temp, title='Stock Price Movements by Sector', 
                  hovermode='x unified', showlegend=False, width=1000,
                  updatemenus=[dict(active=0, type="dropdown",
                                    buttons=buttons, xanchor='left',
                                    yanchor='bottom', y=1.01, x=.01)],
                  yaxis=dict(title='Stock Price'))
fig.show()

