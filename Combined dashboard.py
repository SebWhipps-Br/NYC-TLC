import os
import random
import dask.dataframe as dd
import pandas as pd
import dash
from dash import dcc, html, Input, Output
import plotly.express as px

# Path to dataset folder (update if renamed later)
data_folder = r"C:\Users\Mikolaj Szymczak\Desktop\Schoolwork\NYC Taxis\NYC-TLC\clean yellow taxis 2024"

# Function to determine taxi type based on filename
def get_taxi_type(filename):
    if "yellow" in filename:
        return "Yellow"
    elif "green" in filename:
        return "Green"
    elif "fhv" in filename:
        return "FHV"
    elif "hvfhv" in filename:
        return "HVFHV"
    return "Unknown"

# Function to load and preprocess data
def preprocess_data():
    all_files = [f for f in os.listdir(data_folder) if f.endswith(".parquet")]
    if not all_files:
        raise FileNotFoundError("No cleaned taxi data files found in the dataset folder.")

    data_frames = []
    
    for file in all_files:
        taxi_type = get_taxi_type(file)
        file_path = os.path.join(data_folder, file)
        
        # Load parquet file
        df = dd.read_parquet(file_path, engine="pyarrow")
        
        # Add a 'taxi_type' column
        df["taxi_type"] = taxi_type
        
        # Convert datetime columns
        df["tpep_pickup_datetime"] = dd.to_datetime(df["tpep_pickup_datetime"])
        df["tpep_dropoff_datetime"] = dd.to_datetime(df["tpep_dropoff_datetime"])
        
        # Extract additional time-based features
        df["hour"] = df["tpep_pickup_datetime"].dt.hour
        df["day_of_week"] = df["tpep_pickup_datetime"].dt.dayofweek
        df["month"] = df["tpep_pickup_datetime"].dt.month
        
        data_frames.append(df)
    
    # Concatenate all taxi types
    df_combined = dd.concat(data_frames, axis=0)
    
    # Save precomputed dataset
    df_combined.compute().to_parquet("precomputed_taxi_data.parquet")

# Check if preprocessed file exists; otherwise, process data
if not os.path.exists("precomputed_taxi_data.parquet"):
    preprocess_data()

# Load precomputed data
df_combined = pd.read_parquet("precomputed_taxi_data.parquet")

# Initialize Dash App
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("NYC Taxi Dashboard"),
    
    # Dropdown to select cab type
    html.Label("Select Cab Type:"),
    dcc.Dropdown(
        id="cab_type_filter",
        options=[{"label": t, "value": t} for t in df_combined["taxi_type"].unique()],
        value="Yellow",
        clearable=False
    ),

    # Graph 1: Average Fare per Hour by Drop-off
    html.Div([
        html.Label("Select Hour:"),
        dcc.Dropdown(
            id="hour_filter",
            options=[{"label": f"{h}:00", "value": h} for h in range(24)],
            value=12,
            clearable=False
        ),
        dcc.Graph(id="fare_chart")
    ]),

    # Graph 2: Heatmap of Trip Counts by Day-Hour
    html.Div([
        html.Label("Select Month:"),
        dcc.Dropdown(
            id="month_filter",
            options=[{"label": f"Month {m}", "value": m} for m in df_combined["month"].unique()],
            value=1,
            clearable=False
        ),
        dcc.Graph(id="heatmap_chart")
    ]),

    # Graph 3: Hourly Demand by Drop-off
    html.Div([
        html.Label("Select Drop-off Location:"),
        dcc.Dropdown(
            id="dropoff_filter",
            options=[{"label": f"Location {loc}", "value": loc} for loc in df_combined["DOLocationID"].unique()],
            value=df_combined["DOLocationID"].mode()[0],
            clearable=False
        ),
        dcc.Graph(id="demand_chart")
    ])
])

# Callback for Graph 1: Average Fare per Hour by Drop-off
@app.callback(
    Output("fare_chart", "figure"),
    [Input("hour_filter", "value"),
     Input("cab_type_filter", "value")]
)
def update_fare_chart(selected_hour, cab_type):
    df_filtered = df_combined[(df_combined["hour"] == selected_hour) & (df_combined["taxi_type"] == cab_type)]
    df_avg_fare = df_filtered.groupby("DOLocationID")["fare_amount"].mean().reset_index()

    fig = px.bar(df_avg_fare, x="DOLocationID", y="fare_amount",
                 title=f"Average Fare at {selected_hour}:00 for {cab_type} Taxis")
    return fig

# Callback for Graph 2: Heatmap of Trip Counts by Day-Hour
@app.callback(
    Output("heatmap_chart", "figure"),
    [Input("month_filter", "value"),
     Input("cab_type_filter", "value")]
)
def update_heatmap(selected_month, cab_type):
    df_filtered = df_combined[(df_combined["month"] == selected_month) & (df_combined["taxi_type"] == cab_type)]
    df_counts = df_filtered.groupby(["day_of_week", "hour"]).size().reset_index(name="trip_count")

    fig = px.density_heatmap(df_counts, x="hour", y="day_of_week", z="trip_count",
                             title=f"Trip Counts by Day-Hour for {cab_type} Taxis (Month {selected_month})")
    return fig

# Callback for Graph 3: Hourly Demand by Drop-off
@app.callback(
    Output("demand_chart", "figure"),
    [Input("dropoff_filter", "value"),
     Input("cab_type_filter", "value")]
)
def update_demand_chart(selected_dropoff, cab_type):
    df_filtered = df_combined[(df_combined["DOLocationID"] == selected_dropoff) & (df_combined["taxi_type"] == cab_type)]
    df_hourly_demand = df_filtered.groupby("hour").size().reset_index(name="trip_count")

    fig = px.line(df_hourly_demand, x="hour", y="trip_count",
                  title=f"Hourly Demand at Drop-off {selected_dropoff} for {cab_type} Taxis")
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
