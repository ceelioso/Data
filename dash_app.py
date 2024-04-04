import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from flask_app import train_df, temp
import plotly.express as px


# Initialize Dash app
external_stylesheets = [dbc.themes.BOOTSTRAP]
app_dash = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# Define app layout
app_dash.layout = html.Div([
    dcc.Graph(id='candlestick-graph')
])

# Define callback function
@app_dash.callback(
    Output('candlestick-graph', 'figure'),
    [Input('dropdown', 'value')]
)
def update_graph(selected_dropdown_value):
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
# Run the app
if __name__ == '__main__':
    app_dash.run_server(debug=True)
