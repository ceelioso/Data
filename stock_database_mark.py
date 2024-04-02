import colorsys
import warnings, gc
import colorama
import numpy as np
import pandas as pd
import matplotlib.colors
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import init_notebook_mode
from datetime import datetime, timedelta
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error,mean_absolute_error
from lightgbm import LGBMRegressor
from decimal import ROUND_HALF_UP, Decimal
warnings.filterwarnings("ignore")
import plotly.figure_factory as ff
from IPython.display import display
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

init_notebook_mode(connected=True)

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
# Load data 
train_df = pd.read_csv(r"D:/NewFolder/app/stock_prices.csv", parse_dates=['Date'])
stock_list = pd.read_pickle(r"D:/NewFolder/app/stock_lists.pickle")

# Display data information
print("The training data begins on {} and ends on {}.\n".format(train_df.Date.min(), train_df.Date.max()))
display(train_df.describe().style.format('{:,.2f}'))

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from a CSV file and return a Pandas DataFrame.

    Parameters:
        file_path (str): The file path of the CSV file.

    Returns:
        A Pandas DataFrame containing the loaded data.
    """
    return pd.read_csv(file_path)


# Initialize Dash app
app = dash.Dash(__name__,  
                external_scripts=['https://cdn.plot.ly/plotly-2.30.0.min.js'],
                suppress_callback_exceptions=True,
                external_stylesheets=[dbc.themes.BOOTSTRAP])


# Define app layout
# Add dropdown
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
    if selected_dropdown_value in train_df.columns:
        fig = px.line(train_df, x="Date", y=selected_dropdown_value, color_discrete_sequence=temp["colors"])
        fig.update_layout(**temp)
        return fig
    else:
        print("Error: The selected dropdown value is not a valid column name in the DataFrame.")

# Run the app
if __name__ == '__main__':
    app.run_server()


print("Columns present in train_df:", train_df.columns)

# Check if the 'Name' column exists in the DataFrame
if 'Name' in train_df.columns:
  if 'SectorName' in df_temp.columns:
    avg_return = df_temp.groupby('SectorName')['Target'].mean()
  else:
    print("Error: 'SectorName' column not found in df_temp.")
else:
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

  if 'SectorName' in df_temp.columns:
    avg_return = df_temp.groupby('SectorName')['Target'].mean()
else:
    print("Error: 'SectorName' column not found in df_temp.")

if '17SectorName' in df_temp.columns:
  df_temp['SectorName'] = df_temp['17SectorName'].str.rstrip().str.lower().str.capitalize()
else:
  print("Error: '17SectorName' column not found in df_temp.")
  # Update the column name to 'SectorName'
try:
  avg_return = df_temp.groupby('SectorName')['Target'].mean()
except KeyError:
  avg_return = df_temp.groupby('17SectorName')['Target'].mean()
  
  result_df = result_df.append(avg_return, ignore_index=True)


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
    mask = x<=0
    fig.add_trace(go.Bar(x=x[mask], y=train_df.index[mask],orientation='h', 
                         text=x[mask], texttemplate='%{text:.2f}%',textposition='auto',
                         hovertemplate='Average Return in %{y} Stocks = %{x:.4f}%',
                         marker=dict(color='red', opacity=0.7),name=col[-4:]), 
                  row=1, col=i+1)
    fig.add_trace(go.Bar(x=x[~mask], y=result_df.index[~mask],orientation='h', 
                         text=x[~mask], texttemplate='%{text:.2f}%', textposition='auto', 
                         hovertemplate='Average Return in %{y} Stocks = %{x:.4f}%',
                         marker=dict(color='green', opacity=0.7),name=col[-4:]), 
                  row=1, col=i+1)
    fig.update_xaxes(range=(x.min()-.15,x.max()+.15), title='{} Returns'.format(col[-4:]), 
                     showticklabels=False, row=1, col=i+1)
fig.update_layout(template=temp,title='Yearly Average Stock Returns by Sector', 
                  hovermode='closest',margin=dict(l=250,r=50),
                  height=600, width=1000, showlegend=False)
fig.show()


train_df=train_df[train_df.Date>'2024-12-23']

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


fig.add_trace(go.Histogram(x=x_hist*100,
                           marker=dict(color=colorama[0], opacity=0.7, 
                                       line=dict(width=1, color=colorsys[0])),
                           xbins=dict(start=-40,end=40,size=1)))
fig.update_layout(template=temp,title='Target Distribution', 
                  xaxis=dict(title='Stock Return',ticksuffix='%'), height=450)
fig.show()


pal = ['hsl('+str(h)+',50%'+',50%)' for h in np.linspace(0, 360, 18)]
fig = go.Figure()
for i, sector in enumerate(result_df.index[::-1]):
    y_data=train_df[train_df['SectorName']==sector]['Target']
    fig.add_trace(go.Box(y=y_data*100, name=sector,
                         marker_color=pal[i], showlegend=False))
fig.update_layout(template=temp, title='Target Distribution by Sector',
                  yaxis=dict(title='Stock Return',ticksuffix='%'),
                  margin=dict(b=150), height=750, width=900)
fig.show()

train_date=train_df.Date.unique()
sectors = train_df.columns.unique().tolist()
sectors = []
sectors.insert(0, 'All')
open_avg=train_df.groupby('Date')['Open'].mean()
high_avg=train_df.groupby('Date')['High'].mean()
low_avg=train_df.groupby('Date')['Low'].mean()
close_avg=train_df.groupby('Date')['Close'].mean() 
buttons=[]
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
            dict(label=sector, method="update",
                 args=[{"visible": [sector == trace.name for trace in fig.data]}])
            for sector in sectors
        ]
    )]
)

# Show the figure
fig.show()

# Show the figure
fig.show()
fig.update_xaxes(rangeslider_visible=True,
                 rangeselector=dict(
                     buttons=list([
                         dict(count=3, label="3m", step="month", stepmode="backward"),
                         dict(count=6, label="6m", step="month", stepmode="backward"),
                         dict(step="all")]), xanchor='left',yanchor='bottom', y=1.16, x=.01))
fig.update_layout(template=temp,title='Stock Price Movements by Sector', 
                  hovermode='x unified', showlegend=False, width=1000,
                  updatemenus=[dict(active=0, type="dropdown",
                                    buttons=buttons, xanchor='left',
                                    yanchor='bottom', y=1.01, x=.01)],
                  yaxis=dict(title='Stock Price'))
fig.show()
stock_high=stock.nlargest(7).rename("Return")
stock=pd.concat([stock_high, stock_low], axis=0).reset_index()
stock['Sector']='All'
for i in train_df.SectorName.unique():
    sector=train_df[train_df.SectorName==i].groupby('Name')['Target'].mean().mul(100)
    stock_low=sector.nsmallest(7)[::-1].rename("Return")
    stock_high=sector.nlargest(7).rename("Return")
    sector_stock=pd.concat([stock_high, stock_low], axis=0).reset_index()
    sector_stock['Sector']=i
    stock=stock.append(sector_stock,ignore_index=True)
    
fig=go.Figure()
buttons = []
for i, sector in enumerate(stock.Sector.unique()):
    
    x=stock[stock.Sector==sector]['Name']
    y=stock[stock.Sector==sector]['Return']
    mask=y>0
    fig.add_trace(go.Bar(x=x[mask], y=y[mask], text=y[mask], 
                         texttemplate='%{text:.2f}%',
                         textposition='auto',
                         name=sector, visible=(False if i != 0 else True),
                         hovertemplate='%{x} average return: %{y:.3f}%',
                         marker=dict(color='green', opacity=0.7)))
    fig.add_trace(go.Bar(x=x[~mask], y=y[~mask], text=y[~mask], 
                         texttemplate='%{text:.2f}%',
                         textposition='auto',
                         name=sector, visible=(False if i != 0 else True),
                         hovertemplate='%{x} average return: %{y:.3f}%',
                         marker=dict(color='red', opacity=0.7)))
    
    visibility=[False]*2*len(stock.Sector.unique())
    visibility[i*2],visibility[i*2+1]=True,True
    button = dict(label = sector,
                  method = "update",
                  args=[{"visible": visibility}])
    buttons.append(button)

fig.update_layout(title='Stocks with Highest and Lowest Returns by Sector',
                  template=temp, yaxis=dict(title='Average Return', ticksuffix='%'),
                  updatemenus=[dict(active=0, type="dropdown",
                                    buttons=buttons, xanchor='left',
                                    yanchor='bottom', y=1.01, x=.01)], 
                  margin=dict(b=150),showlegend=False,height=700, width=900)
fig.show()



stocks=train_df[train_df.SecuritiesCode.isin([4169,7089,4582,2158,7036])]
df_pivot=stocks.pivot_table(index='Date', columns='Name', values='Close').reset_index()
pal=['rgb'+str(i) for i in sns.color_palette("coolwarm", len(df_pivot))]

fig = ff.create_scatterplotmatrix(df_pivot.iloc[:,1:], diag='histogram', name='')
fig.update_traces(marker=dict(color=pal, opacity=0.9, line_color='white', line_width=.5))
fig.update_layout(template=temp, title='Scatterplots of Highest Performing Stocks', 
                  height=1000, width=1000, showlegend=False)
fig.show()

corr=train_df.groupby('SecuritiesCode')[['Target','Close']].corr().unstack().iloc[:,1]
stocks=corr.nlargest(10).rename("Return").reset_index()
stocks=stocks.merge(train_df[['Name','SecuritiesCode']], on='SecuritiesCode').drop_duplicates()
pal=sns.color_palette("magma_r", 14).as_hex()
rgb=['rgba'+str(matplotlib.colors.to_rgba(i,0.7)) for i in pal]

del X_train, y_train,  X_val, y_val
gc.collect()

print("\nAverage cross-validation Sharpe Ratio: {:.4f}, standard deviation = {:.2f}.".format(np.mean(sharpe_ratio),np.std(sharpe_ratio)))

feat_importance['avg'] = feat_importance.mean(axis=1)
feat_importance = feat_importance.sort_values(by='avg',ascending=True)
pal=sns.color_palette("plasma_r", 29).as_hex()[2:]

fig=go.Figure()
for i in range(len(feat_importance.index)):
    fig.add_shape(dict(type="line", y0=i, y1=i, x0=0, x1=feat_importance['avg'][i], 
                       line_color=pal[::-1][i],opacity=0.7,line_width=4))
fig.add_trace(go.Scatter(x=feat_importance['avg'], y=feat_importance.index, mode='markers', 
                         marker_color=pal[::-1], marker_size=8,
                         hovertemplate='%{y} Importance = %{x:.0f}<extra></extra>'))
fig.update_layout(template=temp,title='Overall Feature Importance', 
                  xaxis=dict(title='Average Importance',zeroline=False),
                  yaxis_showgrid=False, margin=dict(l=120,t=80),
                  height=700, width=800)
fig.show()


df_pivot=train_df.pivot_table(index='Date', columns='SectorName', values='Close').reset_index()
corr=df_pivot.corr().round(2)
mask=np.triu(np.ones_like(corr, dtype=bool))
c_mask = np.where(~mask, corr, 100)
c=[]
for i in c_mask.tolist()[1:]:
    c.append([x for x in i if x != 100])
    
cor=c[::-1]
x=corr.index.tolist()[:-1]
y=corr.columns.tolist()[1:][::-1]
fig=ff.create_annotated_heatmap(z=cor, x=x, y=y, 
                                hovertemplate='Correlation between %{x} and %{y} stocks = %{z}',
                                colorscale='viridis', name='')
fig.update_layout(template=temp, title='Stock Correlation between Sectors',
                  margin=dict(l=250,t=270),height=800,width=900,
                  yaxis=dict(showgrid=False, autorange='reversed'),
                  xaxis=dict(showgrid=False))
fig.show()

def adjust_price(price):
    """
    Args:
        price (pd.DataFrame)  : pd.DataFrame include stock_price
    Returns:
        price DataFrame (pd.DataFrame): stock_price with generated AdjustedClose
    """
    # transform Date column into datetime
    price.loc[: ,"Date"] = pd.to_datetime(price.loc[: ,"Date"], format="%Y-%m-%d")

    def generate_adjusted_close(df):
        """
        Args:
            df (pd.DataFrame)  : stock_price for a single SecuritiesCode
        Returns:
            df (pd.DataFrame): stock_price with AdjustedClose for a single SecuritiesCode
        """
        # sort data to generate CumulativeAdjustmentFactor
        df = df.sort_values("Date", ascending=False)
        # generate CumulativeAdjustmentFactor
        df.loc[:, "CumulativeAdjustmentFactor"] = df["AdjustmentFactor"].cumprod()
        # generate AdjustedClose
        df.loc[:, "AdjustedClose"] = (
            df["CumulativeAdjustmentFactor"] * df["Close"]
        ).map(lambda x: float(
            Decimal(str(x)).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP)
        ))
        # reverse order
        df = df.sort_values("Date")
        # to fill AdjustedClose, replace 0 into np.nan
        df.loc[df["AdjustedClose"] == 0, "AdjustedClose"] = np.nan
        # forward fill AdjustedClose
        df.loc[:, "AdjustedClose"] = df.loc[:, "AdjustedClose"].ffill()
        return df
    
    # generate AdjustedClose
    price = price.sort_values(["SecuritiesCode", "Date"])
    price = price.groupby("SecuritiesCode").apply(generate_adjusted_close).reset_index(drop=True)
    return price

train=train_df.drop('ExpectedDividend',axis=1).fillna(0)
prices=adjust_price(train)



def create_features(df):
    df=df.copy()
    col='AdjustedClose'
    periods=[5,10,20,30,50]
    for period in periods:
        df.loc[:,"Return_{}Day".format(period)] = df.groupby("SecuritiesCode")[col].pct_change(period)
        df.loc[:,"MovingAvg_{}Day".format(period)] = df.groupby("SecuritiesCode")[col].rolling(window=period).mean().values
        df.loc[:,"ExpMovingAvg_{}Day".format(period)] = df.groupby("SecuritiesCode")[col].ewm(span=period,adjust=False).mean().values
        df.loc[:,"Volatility_{}Day".format(period)] = np.log(df[col]).groupby(df["SecuritiesCode"]).diff().rolling(period).std()
    return df

price_features=create_features(df=prices)
price_features.drop(['RowId','SupervisionFlag','AdjustmentFactor','CumulativeAdjustmentFactor','Close'],axis=1,inplace=True)


price_names=price_features.merge(stock_list[['SecuritiesCode','Name','SectorName']], on='SecuritiesCode').set_index('Date')
price_names=price_names[price_names.index>='2020-12-29']
price_names.fillna(0, inplace=True)

features=['MovingAvg','ExpMovingAvg','Return', 'Volatility']
names=['Average', 'Exp. Moving Average', 'Period', 'Volatility']
buttons=[]

fig = make_subplots(rows=2, cols=2, 
                    shared_xaxes=True, 
                    vertical_spacing=0.1,
                    subplot_titles=('Adjusted Close Moving Average',
                                    'Exponential Moving Average',
                                    'Stock Return', 'Stock Volatility'))

for i, sector in enumerate(price_names.SectorName.unique()):
    
    sector_df=price_names[price_names.SectorName==sector]
    periods=[0,10,30,50]
    colors=px.colors.qualitative.Vivid
    dash=['solid','dash', 'longdash', 'dashdot', 'longdashdot']
    row,col=1,1
    
    for j, (feature, name) in enumerate(zip(features, names)):
        if j>=2:
            row,periods=2,[10,30,50]
            colors=px.colors.qualitative.Bold[1:]
        if j%2==0:
            col=1
        else:
            col=2
        
        for k, period in enumerate(periods):
            if (k==0)&(j<2):
                plot_data=sector_df.groupby(sector_df.index)['AdjustedClose'].mean().rename('Adjusted Close')
            elif j>=2:
                plot_data=sector_df.groupby(sector_df.index)['{}_{}Day'.format(feature,period)].mean().mul(100).rename('{}-day {}'.format(period,name))
            else:
                plot_data=sector_df.groupby(sector_df.index)['{}_{}Day'.format(feature,period)].mean().rename('{}-day {}'.format(period,name))
            fig.add_trace(go.Scatter(x=plot_data.index, y=plot_data, mode='lines',
                                     name=plot_data.name, marker_color=colors[k+1],
                                     line=dict(width=2,dash=(dash[k] if j<2 else 'solid')), 
                                     showlegend=(True if (j==0) or (j==2) else False), legendgroup=row,
                                     visible=(False if i != 0 else True)), row=row, col=col)
            
    visibility=[False]*14*len(price_names.SectorName.unique())
    for l in range(i*14, i*14+14):
        visibility[l]=True
    button = dict(label = sector,
                  method = "update",
                  args=[{"visible": visibility}])
    buttons.append(button)

fig.update_layout(title='Stock Price Moving Average, Return,<br>and Volatility by Sector',
                  template=temp, yaxis3_ticksuffix='%', yaxis4_ticksuffix='%',
                  legend_title_text='Period', legend_tracegroupgap=250,
                  updatemenus=[dict(active=0, type="dropdown",
                                    buttons=buttons, xanchor='left',
                                    yanchor='bottom', y=1.105, x=.01)], 
                  hovermode='x unified', height=800,width=1200, margin=dict(t=150))
fig.show()


iris = load_iris()
X = iris.data  # Feature matrix
y = iris.target  # Target variable

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define and train the random forest classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Now that 'rf' is defined, you can use it for prediction
y_pred = rf.predict(X_test)

# Example usage of 'y_valid', assuming it's supposed to represent the true labels
y_valid = y_test

# Now you can use 'y_valid' along with 'y_pred' for evaluation or other purposes

def calc_spread_return_sharpe(df: pd.DataFrame, portfolio_size: int = 200, toprank_weight_ratio: float = 2) -> float:
    """
    Args:
        df (pd.DataFrame): predicted results
        portfolio_size (int): # of equities to buy/sell
        toprank_weight_ratio (float): the relative weight of the most highly ranked stock compared to the least.
    Returns:
        (float): sharpe ratio
    """
    def _calc_spread_return_per_day(df, portfolio_size, toprank_weight_ratio):
        """
        Args:
            df (pd.DataFrame): predicted results
            portfolio_size (int): # of equities to buy/sell
            toprank_weight_ratio (float): the relative weight of the most highly ranked stock compared to the least.
        Returns:
            (float): spread return
        """
        assert df['Rank'].min() == 0
        assert df['Rank'].max() == len(df['Rank']) - 1
        weights = np.linspace(start=toprank_weight_ratio, stop=1, num=portfolio_size)
        purchase = (df.sort_values(by='Rank')['Target'][:portfolio_size] * weights).sum() / weights.mean()
        short = (df.sort_values(by='Rank', ascending=False)['Target'][:portfolio_size] * weights).sum() / weights.mean()
        return purchase - short

    buf = df.groupby('Date').apply(_calc_spread_return_per_day, portfolio_size, toprank_weight_ratio)
    sharpe_ratio = buf.mean() / buf.std()
    return sharpe_ratio

import pandas as pd

def calculate_feature_importances(model, X, feature_names):
    """Calculate feature importances using a trained model.

    Parameters:
        model (any sklearn model): A trained model that has the `feature_importances_` attribute.
        X (pandas.DataFrame): The feature data used to train the model.
        feature_names (list): A list of strings representing the column names of X.

    Returns:
        pandas.DataFrame: A pandas DataFrame with the feature importances.
    """
    importances = model.feature_importances_
    feature_importances = pd.DataFrame(importances, columns=["Importance"]).reset_index()
    feature_importances.rename(columns={"index": "FeatureName"}, inplace=True)
    feature_importances["FeatureName"] = feature_names
    feature_importances = feature_importances.sort_values(by="Importance", ascending=False)
    return feature_importances

# Compute feature importances
importances = calculate_feature_importances(rf, X.iloc[:, :-1], X.columns[:-1])


def plot_sector_returns(df, column):
    """Plots the average return for each sector."""
    sector_df = df.groupby(['SectorName', 'Name'])[column].mean().mul(100).reset_index()
    sector_df = sector_df.sort_values(by=column, ascending=False)
    
    fig = go.Figure(data=[go.Bar(x=sector_df['SectorName'], y=sector_df[column], text=sector_df[column], textposition='auto')])
    fig.update_layout(title=f'Sector Average Returns ({column.capitalize()} Column)',
                      xaxis_title='SectorName',
                      yaxis_title=f'{column.capitalize()} Column')
    fig.show()

# Call the function with the specified return column
plot_sector_returns(train_df, '17SectorName')

ts_fold = TimeSeriesSplit(n_splits=10, gap=10000)
prices=price_features.dropna().sort_values(['Date','SecuritiesCode'])
y=prices['Target'].to_numpy()
X=prices.drop(['Target'],axis=1)

feat_importance=pd.DataFrame()
sharpe_ratio=[]
    
    
for fold, (train_idx, val_idx) in enumerate(ts_fold.split(X, y)):
    
    print("\n========================== Fold {} ==========================".format(fold+1))
    X_train, y_train = X.iloc[train_idx,:], y[train_idx]
    X_valid, y_val = X.iloc[val_idx,:], y[val_idx]
    
    print("Train Date range: {} to {}".format(X_train.Date.min(),X_train.Date.max()))
    print("Valid Date range: {} to {}".format(X_valid.Date.min(),X_valid.Date.max()))
    
    X_train.drop(['Date','SecuritiesCode'], axis=1, inplace=True)
    X_val=X_valid[X_valid.columns[~X_valid.columns.isin(['Date','SecuritiesCode'])]]
    val_dates=X_valid.Date.unique()[1:-1]
    print("\nTrain Shape: {} {}, Valid Shape: {} {}".format(X_train.shape, y_train.shape, X_val.shape, y_val.shape))
    

    
    params = {'n_estimators': 500,
              'num_leaves' : 100,
              'learning_rate': 0.1,
              'colsample_bytree': 0.9,
              'subsample': 0.8,
              'reg_alpha': 0.4,
              'metric': 'mae',
              'random_state': 21}
    
    rank=[]
    X_val_df=X_valid[X_valid.Date.isin(val_dates)]
    for i in X_val_df.Date.unique():
        temp_df = X_val_df[X_val_df.Date == i].drop(['Date','SecuritiesCode'],axis=1)
        temp_df["pred"] = gbm.predict(temp_df)
        temp_df["Rank"] = (temp_df["pred"].rank(method="first", ascending=False)-1).astype(int)
        rank.append(temp_df["Rank"].values)
    
    gbm = LGBMRegressor(**params).fit(X_train, y_train, 
                                      eval_set=[(X_train, y_train), (X_val, y_val)],
                                      verbose=300, 
                                      eval_metric=['mae','mse'])
    y_pred = gbm.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    mae = mean_absolute_error(y_val, y_pred)
    feat_importance["Importance_Fold"+str(fold)]=gbm.feature_importances_
    feat_importance.set_index(X_train.columns, inplace=True)
    
        # Train a model using the training data
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model on the validation set
    y_pred = model.predict(X_valid)
    val_loss = mean_squared_error(y_valid, y_pred)
    print(f"Validation MSE: {val_loss:.4f}")

    # Calculate feature importances
    feature_importances = model.feature_importances_
    for i, f in enumerate(X.columns):
        feat_importance.loc[fold, f] = feature_importances[i]

    # Calculate the sharpe ratio of the model's predictions
    valid_predictions = (y_pred - y_train.mean()) / y_train.std()
    sharpe_ratio.append(valid_predictions.mean() / valid_predictions.std())

feat_importance.mean().sort_values().plot(kind='barh')
print(f"Average Sharpe Ratio: {np.mean(sharpe_ratio):.4f} +/- {np.std(sharpe_ratio):.4f}")
    
    
# Train a model using the training data
X_train_transformed = pd.DataFrame(model.transform(X_train), columns=model.feature_names_in_)
X_valid_transformed = pd.DataFrame(model.transform(X_valid), columns=model.feature_names_in_)

# Train a model using the transformed features
stacking_model = LGBMRegressor(**params).fit(X_train_transformed, y_train)

# Make predictions using the stacking model
y_pred_stacked = stacking_model.predict(X_valid_transformed)

# Evaluate the performance of the stacking model on the validation set
val_loss_stacked = mean_squared_error(y_pred, y_pred_stacked)
print(f"Validation MSE (stacking model): {val_loss_stacked:.4f}")    

stock_rank=pd.Series([x for y in rank for x in y], name="Rank")
df=pd.concat([X_val_df.reset_index(drop=True),stock_rank,
                  prices[prices.Date.isin(val_dates)]['Target'].reset_index(drop=True)], axis=1)
sharpe=calc_spread_return_sharpe(df)
sharpe_ratio.append(sharpe)
print("Valid Sharpe: {}, RMSE: {}, MAE: {}".format(sharpe,rmse,mae))
    
del X_train, y_train,  X_val, y_val
gc.collect()
    
print("\nAverage cross-validation Sharpe Ratio: {:.4f}, standard deviation = {:.2f}.".format(np.mean(sharpe_ratio),np.std(sharpe_ratio)))

feat_importance['avg'] = feat_importance.mean(axis=1)
feat_importance = feat_importance.sort_values(by='avg',ascending=True)
pal=sns.color_palette("plasma_r", 29).as_hex()[2:]

fig=go.Figure()
for i in range(len(feat_importance.index)):
    fig.add_shape(dict(type="line", y0=i, y1=i, x0=0, x1=feat_importance['avg'][i], 
                       line_color=pal[::-1][i],opacity=0.7,line_width=4))
fig.add_trace(go.Scatter(x=feat_importance['avg'], y=feat_importance.index, mode='markers', 
                         marker_color=pal[::-1], marker_size=8,
                         hovertemplate='%{y} Importance = %{x:.0f}<extra></extra>'))
fig.update_layout(template=temp,title='Overall Feature Importance', 
                  xaxis=dict(title='Average Importance',zeroline=False),
                  yaxis_showgrid=False, margin=dict(l=120,t=80),
                  height=700, width=800)
fig.show()