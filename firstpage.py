from matplotlib import pyplot as plt
from plotly.offline import init_notebook_mode
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from flask import Flask, Response, current_app, flash, redirect, render_template, request, session, url_for
from datetime import datetime
import time
import json
import re
import os
import MySQLdb
import dash
import dash_bootstrap_components as dbc
from dash import dcc, html  # Update import statements
from dash.dependencies import Input, Output
from werkzeug.security import generate_password_hash, check_password_hash
from io import BytesIO
import poetry

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Define database connection function
def connect_db():
    return MySQLdb.connect(host='localhost', user='your_username', passwd='your_password', db='your_db')

# Define login route
@app.route('/pythonlogin/', methods=['GET', 'POST'])
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
@app.route('/pythonlogin/register', methods=['GET', 'POST'])
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
@app.route('/pythonlogin/profile')
def profile():
    if 'loggedin' in session:
        db = connect_db()
        cursor = db.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM accounts WHERE id = %s', (session['id'],))
        account = cursor.fetchone()
        return render_template('profile.html', account=account)
    return redirect(url_for('login'))

# Define home route
@app.route('/pythonlogin/home')
def home():
    if 'loggedin' in session:
        return render_template('home.html', username=session['username'])
    return redirect(url_for('login'))

# Define logout route
@app.route('/pythonlogin/logout')
def logout():
    session.pop('loggedin', None)
    session.pop('id', None)
    session.pop('username', None)
    return redirect(url_for('login'))

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

# Load datastock_prices.csv
train_df = pd.read_csv(r"D:/NewFolder/stockmarket/data/train_data.csv", parse_dates=['Date'])
stock_list = pd.read_pickle(r"D:/NewFolder/stockmarket/data/temp_data.csv")

# Display data information
print("The training data begins on {} and ends on {}.\n".format(train_df.Date.min(), train_df.Date.max()))
print(train_df.describe().style.format('{:,.2f}'))

# Initialize Dash app
app_dash = dash.Dash(__name__, 
                external_scripts=['https://cdn.plot.ly/plotly-2.30.0.min.js'],
                suppress_callback_exceptions=True,
                external_stylesheets=[dbc.themes.BOOTSTRAP])

# Define app layout
app_dash.layout = dbc.Container([
    dcc.Dropdown(
        id='dropdown',
        options=[{'label': i, 'value': i} for i in train_df.columns],
        value='Close'  # Set the default value of the dropdown to 'Close' or any other valid column name
    ),
    dcc.Graph(id='example-graph'),   # Add other components as needed
])

# Define callback functions
@app_dash.callback(
    Output('example-graph', 'figure'),
    [Input('dropdown', 'value')]
)
def callback_graph(selected_dropdown_value):
    # Create a figure based on user input
    fig = px.line(train_df, x="Date", y=selected_dropdown_value, color_discrete_sequence=temp["colors"])
    fig.update_layout(**temp["layout"])
    return fig

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

requires = ["poetry-core>=1.0.0"]

# Run the Dash app
if __name__ == '__main__':
    app_dash.run_server(debug=True)
