from flask_mysqldb import MySQL
import MySQLdb.cursors
import re
import hashlib
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
from plotly.offline import init_notebook_mode
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from flask import Flask, Response, current_app, flash, redirect, render_template, request, session, url_for
from datetime import datetime
import time
import json
import re
import os
import MySQLdb
import dash
from dash import html
import dash_bootstrap_components as dbc
from dash import dcc  # Update import statements
from dash.dependencies import Input, Output
from werkzeug.security import generate_password_hash, check_password_hash
from io import BytesIO


data = {
    "Date": pd.date_range(start="2024-01-01", end="2024-12-31"),
    "Open": [100, 110, 95] * 121,
    "High": [105, 115, 100] * 121,
    "Low": [95, 105, 90] * 121,
    "Close": [105, 115, 100] * 121,
    "SectorName": ["Sector1", "Sector2", "Sector1"] * 121
}


app = Flask(__name__)


# Change this to your secret key (it can be anything, it's for extra protection)
app.secret_key = '139871'

# Enter your database connection details below
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'pythonlogin'

# Intialize MySQL
mysql = MySQL(app)

@app.route('/', methods=['GET', 'POST'])
def login():
    msg = ''
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form:
        username = request.form['username']
        password = request.form['password']
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM accounts WHERE username = %s AND password = %s', (username, password,))
        account = cursor.fetchone()
        if account:
            session['loggedin'] = True
            session['id'] = account['id']
            session['username'] = account['username']
            return 'Logged in successfully!'
        else:
            msg = 'Incorrect username/password!'
    return render_template('index.html', msg=msg)

@app.route('/logout')
def logout():
    session.pop('loggedin', None)
    session.pop('id', None)
    session.pop('username', None)
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    msg = ''
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form and 'email' in request.form:
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM accounts WHERE username = %s', (username,))
        account = cursor.fetchone()
        if account:
            msg = 'Account already exists!'
        elif not re.match(r'[^@]+@[^@]+\.[^@]+', email):
            msg = 'Invalid email address!'
        elif not re.match(r'[A-Za-z0-9]+', username):
            msg = 'Username must contain only characters and numbers!'
        elif not username or not password or not email:
            msg = 'Please fill out the form!'
        else:
            hash = password + app.secret_key
            hash = hashlib.sha1(hash.encode())
            password = hash.hexdigest()
            cursor.execute('INSERT INTO accounts VALUES (NULL, %s, %s, %s)', (username, password, email,))
            mysql.connection.commit()
            msg = 'You have successfully registered!'
    return render_template('register.html', msg=msg)

@app.route('/home')
def home():
    if 'loggedin' in session:
        return render_template('home.html', username=session['username'])
    return redirect(url_for('login'))

@app.route('/profile')
def profile():
    if 'loggedin' in session:
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM accounts WHERE id = %s', (session['id'],))
        account = cursor.fetchone()
        return render_template('profile.html', account=account)
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)

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

def main():
    # Run Flask and Dash apps
    app_dash.run(debug=True)
    app_dash.run_server(debug=True)

# Entry point of the script
if __name__ == "__main__":
    # Call the main function
    main()
# No changes needed

name = "flask-dashboard"
version = "0.1.0"
description = "Financial dashboard with Flask"
authors = ["Flask Coder <flask.awesome@youremail.com>"]
license = "MIT"

# No changes needed
# No changes needed

python = "^3.9"
Flask = "^1.1.2"
pandas = "^1.2.1"
matplotlib = "^3.3.3"

# No changes needed

flake8 = "^3.8.4"

# No changes needed

name = "flask-dashboard"
version = "0.1.0"
description = "Financial dashboard with Flask"
authors = ["Flask Coder <flask.awesome@youremail.com>"]
license = "MIT"

python = "^3.9"
Flask = "^1.1.2"
pandas = "^1.2.1"
matplotlib = "^3.3.3"

flake8 = "^3.8.4"


# No changes needed

requires = ["poetry-core>=1.0.0"]
requires = ["poetry-core>=1.0.0"]


# Run the Dash app
if __name__ == '__main__':
    app_dash.run_server(debug=True)

app = current_app
matplotlib = "^3.3.3"



def plot(prices):

    prices = (
        prices
        .sort_index()
        .apply(np.log)
        .pipe(lambda x: (x - x.mean()) / x.std())
    )
    prices.plot(
        title='Normalized Log Prices',
        color='black',
        alpha=0.5,
        figsize=(10, 6)
    )
    plt.show()


def plotly(prices):
    layout = go.Layout(title='Normalized Log Prices')
    fig = go.Figure(layout=layout)
    for i, col in enumerate(prices.columns):
        fig.add_trace(go.Scatter(x=prices.index, y=prices[col], mode='lines', name=col))
    fig.show()


# Create a Dash application
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

# Load data
df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/gapminderDataFiveYear.csv')

# Create app layout
app.layout = html.Div(children=[
    html.H1(children='Gapminder Data'),

    html.Div(children='''
        Dash: A web application framework for Python.
    '''),

    dcc.Graph(
        id='example-graph',
        figure={
            'data': [
                {'x': df[df['year'] == i]['gdpPercap'], 'y': df[df['year'] == i]['lifeExp'], 'text': df[df['year'] == i]['country'], 'mode': 'markers', 'name': str(i)} for i in df.year.unique()
            ],
            'layout': {
                'title': 'Life Expectancy vs. GDP Per Capita',
                'xaxis': {'title': 'GDP Per Capita'},
                'yaxis': {'title': 'Life Expectancy'},
                'hovermode': 'closest'
            }
        }
    )
])

# Run the application
if __name__ == '__main__':
    app.run_server(debug=True)

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

train_df = pd.read_csv(r"D:/NewFolder/stockmarket/data/train_data.csv", parse_dates=['Date'])
stock_list = pd.read_pickle(r"D:/NewFolder/stockmarket/data/temp_data.csv")

# Display data information
print("The training data begins on {} and ends on {}.\n".format(train_df.Date.min(), train_df.Date.max()))
print(train_df.describe().style.format('{:,.2f}'))


# Initialize Dash app
train_df = pd.DataFrame(data)

# Initialize Dash app
app = dash.Dash(__name__)

# Define app layout
app.layout = html.Div([
    dcc.Graph(id='candlestick-graph')
])

# Define callback function
@app.callback(
    Output('candlestick-graph', 'figure'),
    [Input('dropdown', 'value')]
)
def update_graph(selected_dropdown_value):
    # Assuming train_df is your DataFrame
    fig = px.line(train_df, x="Date", y=selected_dropdown_value)
    return fig

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)


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

app = Flask(__name__, template_folder='templates')

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)