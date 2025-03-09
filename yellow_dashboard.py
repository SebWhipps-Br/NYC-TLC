import os
import random
import dask.dataframe as dd
import pandas as pd
import dash
from dash import dcc, html, Input, Output
import plotly.express as px

# Path to clean data folder
data_folder = r"C:\Users\Mikolaj Szymczak\Desktop\Schoolwork\NYC Taxis\NYC-TLC\clean yellow taxis 2024"

# Function to get available columns in files
def get_available_columns():
    all_files = [f for f in os.listdir(data_folder) if f.startswith("cleaned_yellow_tripdata_2024") and f.endswith(".parquet")]
    if not all_files:
        raise FileNotFoundError("No Yellow Cab files found in the dataset folder.")
    
    sample_file = os.path.join(data_folder, all_files[0])  # Use full file path
    df_sample = pd.read_parquet(sample_file)
    return df_sample.columns


# Function to preprocess data (handling inconsistent columns)
def preprocess_data():
    # Expected columns
    expected_columns = ["tpep_pickup_datetime", "PULocationID", "fare_amount"]

    # Detect available columns in the dataset
    available_columns = get_available_columns()
    
    # Filter only available columns
    selected_columns = [col for col in expected_columns if col in available_columns]

    # Ensure required columns are present
    missing_columns = [col for col in expected_columns if col not in available_columns]
    if missing_columns:
        raise KeyError(f"Missing required columns: {missing_columns}. Found: {available_columns}")

    # Load only Yellow Cab data files
    df = dd.read_parquet(
        os.path.join(data_folder, "cleaned_yellow_tripdata_2024-*.parquet"),  # Ensure correct file pattern
        columns=selected_columns,
        engine="pyarrow"
    )

    # Convert datetime column if it exists
    if "tpep_pickup_datetime" in df.columns:
        df["tpep_pickup_datetime"] = dd.to_datetime(df["tpep_pickup_datetime"])

        # Extract hour for analysis
        df["hour"] = df["tpep_pickup_datetime"].dt.hour

    # Group and aggregate fare amounts by hour and location
    df_grouped = df.groupby(["PULocationID", "hour"])[["fare_amount"]].mean().compute().reset_index()

    # Save precomputed results
    df_grouped.to_parquet("precomputed_fares.parquet")


    # Convert datetime column
    df["tpep_pickup_datetime"] = dd.to_datetime(df["tpep_pickup_datetime"])

    # Aggregate data
    df["hour"] = df["tpep_pickup_datetime"].dt.hour
    df_grouped = df.groupby(["PULocationID", "hour"])[["fare_amount"]].mean().compute().reset_index()

    # Save precomputed results
    df_grouped.to_parquet("precomputed_fares.parquet")

# Check if precomputed file exists, otherwise process data
if not os.path.exists("precomputed_fares.parquet"):
    preprocess_data()

# Load precomputed data
df_grouped = pd.read_parquet("precomputed_fares.parquet").reset_index()

def load_sample_data():
    all_files = [f for f in os.listdir(data_folder) if f.startswith("yellow_tripdata_2024") and f.endswith(".parquet")]
    if not all_files:
        raise FileNotFoundError("No Yellow Cab files found in the dataset folder.")
    random_file = random.choice(all_files)
    df_sample = pd.read_parquet(os.path.join(data_folder, random_file))
    return df_sample.sample(5000)  # Reduce size for fast filtering

# Initialize Dash App
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("NYC Taxi Dashboard"),
    dcc.Dropdown(
        id="time_filter",
        options=[{"label": f"{hour}:00", "value": hour} for hour in range(24)],
        value=12,
        clearable=False
    ),
    dcc.Graph(id="fare_chart"),
    dcc.Graph(id="scatter_chart")
])

@app.callback(
    Output("fare_chart", "figure"),
    Input("time_filter", "value")
)
def update_chart(selected_hour):
    df_filtered = df_grouped[df_grouped["hour"] == selected_hour]
    fig = px.bar(df_filtered, x="PULocationID", y="fare_amount", title=f"Average Fare at {selected_hour}:00")
    return fig

@app.callback(
    Output("scatter_chart", "figure"),
    Input("time_filter", "value")
)
def update_scatter(selected_hour):
    df_sample = load_sample_data()
    df_sample["tpep_pickup_datetime"] = pd.to_datetime(df_sample["tpep_pickup_datetime"])
    df_sample["hour"] = df_sample["tpep_pickup_datetime"].dt.hour
    df_filtered = df_sample[df_sample["hour"] == selected_hour]
    fig = px.scatter(df_filtered, x="PULocationID", y="fare_amount", 
                     title=f"Real-Time Sample: Fare Distribution at {selected_hour}:00")
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
