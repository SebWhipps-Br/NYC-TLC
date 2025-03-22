import os
import pandas as pd
import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import random

# --- Setup base path ---
base_path = r"C:\Users\Mikolaj Szymczak\OneDrive - University of Bristol\NYC-TLC"

taxi_folders = {
    "Yellow": "yellow taxis",
    "Green": "green taxis",
    "FHV": "fhvs",
    "HVFHV": "high volume fhvs"
}

# --- Initialize Dash ---
app = dash.Dash(__name__)

# --- Available cab types based on folders that exist and contain parquet files ---
taxi_types = []
fare_supported_types = ["Yellow", "Green"]
distance_supported_types = []

for k, v in taxi_folders.items():
    folder_path = os.path.join(base_path, v)
    if os.path.exists(folder_path):
        has_parquet = any(f.endswith('.parquet') and os.path.getsize(os.path.join(folder_path, f)) > 0 for f in os.listdir(folder_path))
        if has_parquet:
            taxi_types.append(k)
            if k in fare_supported_types:
                distance_supported_types.append(k)

# --- Function to load a sample dynamically based on cab type ---
def load_sample_data(cab_type, n=500):
    if cab_type is None:
        print("[WARNING] cab_type is None")
        return pd.DataFrame()

    folder = os.path.join(base_path, taxi_folders[cab_type])
    all_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.parquet') and os.path.getsize(os.path.join(folder, f)) > 0]

    if not all_files:
        print(f"[WARNING] No files found for {cab_type}")
        return pd.DataFrame()

    selected_file = random.choice(all_files)
    print(f"[INFO] Loading data from {selected_file}")

    try:
        df = pd.read_parquet(selected_file)
        print(f"[DEBUG] Loaded shape: {df.shape}")
        print(f"[DEBUG] Columns: {df.columns.tolist()}")

        # Handle column name variations
        if "tpep_pickup_datetime" in df.columns:
            df["pickup"] = pd.to_datetime(df["tpep_pickup_datetime"])
        elif "pickup_datetime" in df.columns:
            df["pickup"] = pd.to_datetime(df["pickup_datetime"])
        else:
            raise KeyError("Missing pickup datetime column")

        if "tpep_dropoff_datetime" in df.columns:
            df["dropoff"] = pd.to_datetime(df["tpep_dropoff_datetime"])
        elif "dropoff_datetime" in df.columns:
            df["dropoff"] = pd.to_datetime(df["dropoff_datetime"])
        else:
            raise KeyError("Missing dropoff datetime column")

        df["journey_time"] = (df["dropoff"] - df["pickup"]).dt.total_seconds() / 60
        df["hour"] = df["pickup"].dt.hour
        return df.sample(n=min(n, len(df)))
    except Exception as e:
        print(f"[ERROR] Failed to load data from {selected_file}: {e}")
        return pd.DataFrame()


# --- App layout ---
app.layout = html.Div([
    html.H1("NYC Taxi Dashboard (Efficient Sample-Based)"),

    html.H2("Average Journey Time Between Locations"),
    dcc.Dropdown(id="cab_type_filter_1", options=[{"label": t, "value": t} for t in taxi_types], value=taxi_types[0] if taxi_types else None, clearable=False),
    dcc.Graph(id="avg_journey_time"),

    html.H2("Average Distance vs Journey Time Between Same Locations"),
    dcc.Dropdown(id="cab_type_filter_2", options=[{"label": t, "value": t} for t in distance_supported_types], value=distance_supported_types[0] if distance_supported_types else None, clearable=False),
    dcc.Graph(id="distance_vs_time"),

    html.H2("Trip Frequency Between Locations"),
    dcc.Dropdown(id="cab_type_filter_3", options=[{"label": t, "value": t} for t in taxi_types], value=taxi_types[0] if taxi_types else None, clearable=False),
    dcc.Graph(id="trip_frequency"),

    html.H2("Journey Time by Hour of Day"),
    dcc.Dropdown(id="cab_type_filter_4", options=[{"label": t, "value": t} for t in taxi_types], value=taxi_types[0] if taxi_types else None, clearable=False),
    dcc.Graph(id="journey_time_by_hour"),

    html.H2("Average Fare by Hour"),
    dcc.Dropdown(id="cab_type_filter_5", options=[{"label": t, "value": t} for t in fare_supported_types], value=fare_supported_types[0] if fare_supported_types else None, clearable=False),
    dcc.Graph(id="fare_by_hour")
])


# --- Graph Callbacks ---
@app.callback(
    Output("avg_journey_time", "figure"),
    Input("cab_type_filter_1", "value")
)
def update_avg_journey(cab_type):
    df = load_sample_data(cab_type)
    if df.empty:
        return px.scatter(title="No Data Available")
    df_avg = df.groupby(["PULocationID", "DOLocationID"], as_index=False)["journey_time"].mean()
    return px.scatter(df_avg, x="PULocationID", y="DOLocationID", size="journey_time", color="journey_time",
                      labels={"PULocationID": "Pick-up location", "DOLocationID": "Drop-off location", "journey_time": "Avg. time (min)"},
                      title="Average Journey Time Between Locations")


@app.callback(
    Output("distance_vs_time", "figure"),
    Input("cab_type_filter_2", "value")
)
def update_distance_vs_time(cab_type):
    df = load_sample_data(cab_type)
    if df.empty:
        return px.scatter(title="No Data Available")
    df_avg = df.groupby(["trip_distance"], as_index=False)["journey_time"].mean()
    return px.scatter(df_avg, x="trip_distance", y="journey_time", color="trip_distance",
                      labels={"trip_distance": "Trip Distance (miles)", "journey_time": "Journey Time (min)"},
                      title="Distance vs Journey Time Between Same Locations")


@app.callback(
    Output("trip_frequency", "figure"),
    Input("cab_type_filter_3", "value")
)
def update_trip_frequency(cab_type):
    df = load_sample_data(cab_type)
    if df.empty:
        return px.density_heatmap(title="No Data Available")
    df_count = df.groupby(["PULocationID", "DOLocationID"], as_index=False).size().rename(columns={"size": "trip_count"})
    return px.density_heatmap(df_count, x="PULocationID", y="DOLocationID", z="trip_count",
                              labels={"PULocationID": "Pick-up location", "DOLocationID": "Drop-off location", "trip_count": "Trip Count"},
                              title="Trip Frequency Between Locations")


@app.callback(
    Output("journey_time_by_hour", "figure"),
    Input("cab_type_filter_4", "value")
)
def update_journey_time_by_hour(cab_type):
    df = load_sample_data(cab_type)
    if df.empty:
        return px.line(title="No Data Available")
    df_avg = df.groupby("hour", as_index=False)["journey_time"].mean()
    return px.line(df_avg, x="hour", y="journey_time",
                   labels={"hour": "Hour of Day", "journey_time": "Journey Time (min)"},
                   title="Journey Time by Hour of Day")


@app.callback(
    Output("fare_by_hour", "figure"),
    Input("cab_type_filter_5", "value")
)
def update_fare_by_hour(cab_type):
    df = load_sample_data(cab_type)
    if df.empty or "fare_amount" not in df.columns:
        return px.line(title="No Fare Data Available")
    df_avg = df.groupby("hour", as_index=False)["fare_amount"].mean()
    return px.line(df_avg, x="hour", y="fare_amount",
                   labels={"hour": "Hour of Day", "fare_amount": "Average Fare ($)"},
                   title="Average Fare by Hour")


# --- Run App ---
if __name__ == '__main__':
    app.run_server(debug=True)
