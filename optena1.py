import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet

# Set the page configuration
st.set_page_config(page_title="OPTENA: Energy Optimization Systems", page_icon="favicon.ico", layout="wide")

# App Header
st.markdown(
    "<h1 style='text-align: center; color: #4CAF50; font-size: 5em;'>OPTENA</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<h2 style='text-align: center; color: #555;'>Data Center Energy Optimization Simulator & Forecaster</h2>",
    unsafe_allow_html=True
)

# Function to load data
def load_data(file, file_type="csv"):
    if file_type == "csv":
        data = pd.read_csv(file)
    elif file_type == "h5":
        data = pd.read_hdf(file)
    else:
        raise ValueError("Unsupported file type. Please upload a CSV or HDF5 file.")
    
    if 'Timestamp' in data.columns:
        data['Timestamp'] = pd.to_datetime(data['Timestamp'])
        data.set_index('Timestamp', inplace=True)
    return data

# Standardize column names
def standardize_columns(data):
    required_columns = {
        'Renewable Availability (%)': ['Renewable Availability', 'Renewables (%)'],
        'Workload Energy Consumption (kWh)': ['Workload Consumption', 'Energy (kWh)'],
        'Energy Price ($/kWh)': ['Energy Price', 'Price ($)'],
    }

    for standard_col, possible_names in required_columns.items():
        for col in possible_names:
            if col in data.columns:
                data.rename(columns={col: standard_col}, inplace=True)
                break

    missing_columns = [col for col in required_columns.keys() if col not in data.columns]
    if missing_columns:
        st.error(f"Missing required columns: {', '.join(missing_columns)}")
        st.stop()

    return data

# Forecasting with Prophet
def forecast_prophet(data, columns, periods=365*24):
    forecasts = {}
    for column in columns:
        prophet_data = data.reset_index()[['Timestamp', column]].rename(columns={'Timestamp': 'ds', column: 'y'})
        model = Prophet()
        model.fit(prophet_data)
        future = model.make_future_dataframe(periods=periods, freq='H')
        forecast = model.predict(future)
        forecasts[column] = forecast[['ds', 'yhat']]
    return forecasts

# Simulation function
def run_simulation(data, renewable_threshold, energy_price_per_kwh, emission_factor_non_renewable, emission_factor_renewable):
    data['Baseline Emissions'] = (
        data['Workload Energy Consumption (kWh)'] * emission_factor_non_renewable * (1 - data['Renewable Availability (%)'] / 100) +
        data['Workload Energy Consumption (kWh)'] * emission_factor_renewable * (data['Renewable Availability (%)'] / 100)
    )

    data['Optimized Energy'] = np.where(
        data['Renewable Availability (%)'] >= renewable_threshold * 100,
        data['Workload Energy Consumption (kWh)'],
        data['Workload Energy Consumption (kWh)'] * 0.9
    )

    data['Optimized Emissions'] = (
        data['Optimized Energy'] * emission_factor_non_renewable * (1 - data['Renewable Availability (%)'] / 100) +
        data['Optimized Energy'] * emission_factor_renewable * (data['Renewable Availability (%)'] / 100)
    )

    results = {
        'optimized_energy': round(data['Optimized Energy'].sum(), 2),
        'optimized_cost': round((data['Optimized Energy'] * energy_price_per_kwh).sum(), 2),
        'optimized_emissions': round(data['Optimized Emissions'].sum(), 2),
        'energy_savings': round(data['Workload Energy Consumption (kWh)'].sum() - data['Optimized Energy'].sum(), 2),
        'cost_savings': round((data['Workload Energy Consumption (kWh)'] * energy_price_per_kwh).sum() - \
                        (data['Optimized Energy'] * energy_price_per_kwh).sum(), 2),
        'emissions_savings': round(data['Baseline Emissions'].sum() - data['Optimized Emissions'].sum(), 2),
    }
    return results

# Initialize data to None at the start
data = None

# Unified UI Workflow
st.title("Energy Optimization and Workload Distribution")

# Step 1: Data Upload
uploaded_file = st.file_uploader("Upload your data file (CSV or HDF5)", type=["csv", "h5"])
data_for_processing = None
if uploaded_file:
    file_type = "csv" if uploaded_file.name.endswith('.csv') else "h5"
    data = load_data(uploaded_file, file_type)
    data = standardize_columns(data)
    st.write("Uploaded Data Preview:", data.head())

# Step 2: Parameter Configuration
st.subheader("Configure Parameters")
energy_price_per_kwh = st.slider("Energy Price per kWh ($)", min_value=0.05, max_value=1.00, value=0.1, step=0.01)
renewable_threshold = st.slider("Renewable Energy Threshold (%)", min_value=0, max_value=100, value=70, step=5) / 100
emission_factor_non_renewable = st.slider("Emission Factor (Non-Renewables)", min_value=0.1, max_value=1.0, value=0.5, step=0.05)
emission_factor_renewable = st.slider("Emission Factor (Renewables)", min_value=0.0, max_value=0.5, value=0.02, step=0.01)

# Step 3: Forecasting
use_forecasted_data = st.radio("Use Forecasted Data?", ["No", "Yes"], index=0)
if use_forecasted_data == "Yes" and st.button("Generate Forecasts"):
    with st.spinner("Generating forecasts..."):
        columns_to_forecast = ['Renewable Availability (%)', 'Workload Energy Consumption (kWh)', 'Energy Price ($/kWh)']
        forecasts = forecast_prophet(data, columns_to_forecast)
        forecasted_data = pd.DataFrame({
            'Timestamp': forecasts['Workload Energy Consumption (kWh)']['ds'],
            'Workload Energy Consumption (kWh)': forecasts['Workload Energy Consumption (kWh)']['yhat'],
            'Renewable Availability (%)': forecasts['Renewable Availability (%)']['yhat'],
            'Energy Price ($/kWh)': forecasts['Energy Price ($/kWh)']['yhat']
        })
        forecasted_data.set_index('Timestamp', inplace=True)
        st.write("Forecast Results:", forecasted_data.head())

        # Display charts for forecasts
        for column, forecast in forecasts.items():
            st.write(f"Forecast for {column}:")
            fig, ax = plt.subplots(figsize=(4, 2))
            ax.plot(forecast['ds'], forecast['yhat'], label=f"{column} Forecast")
            ax.set_title(f"{column} Forecast")
            ax.set_xlabel("Time")
            ax.set_ylabel(column)
            ax.legend()
            st.pyplot(fig)

        st.success("You are now using forecasted metrics for the simulation.")
        data_for_processing = forecasted_data
else:
    data_for_processing = data
    if use_forecasted_data == "No":
        st.info("You are using historical data for the simulation.")

# Step 4: Run Optimization
if st.button("Run Optimization"):
    if data_for_processing is not None:
        with st.spinner("Running optimization..."):
            results = run_simulation(
                data_for_processing,
                renewable_threshold,
                energy_price_per_kwh,
                emission_factor_non_renewable,
                emission_factor_renewable
            )
        st.write("Optimization Results:")
        col1, col2, col3 = st.columns(3)
        col1.metric("Optimized Energy (kWh)", f"{results['optimized_energy']}", delta=f"{results['energy_savings']} kWh")
        col2.metric("Optimized Cost ($)", f"{results['optimized_cost']}", delta=f"${results['cost_savings']}")
        col3.metric("Optimized CO2 Emissions (kg)", f"{results['optimized_emissions']}", delta=f"{results['emissions_savings']} kg")

        if use_forecasted_data == "Yes":
            st.success("These results are based on forecasted metrics.")
        else:
            st.success("These results are based on historical data.")

        # Optimization performance charts
        if 'Optimized Energy' in data_for_processing.columns:
            st.write("Optimization Performance Over Time:")
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(data_for_processing.index, data_for_processing['Workload Energy Consumption (kWh)'], label="Original Consumption", alpha=0.7)
            ax.plot(data_for_processing.index, data_for_processing['Optimized Energy'], label="Optimized Consumption", alpha=0.7)
            ax.set_title("Energy Consumption: Original vs Optimized")
            ax.set_xlabel("Time")
            ax.set_ylabel("Energy (kWh)")
            ax.legend()
            st.pyplot(fig)
    else:
        st.error("No data available for processing. Please upload data.")

# Step 5: Recommendations
if data_for_processing is not None:
    st.subheader("OPTENA Recommendations")

    # Time-specific recommendations
    data_for_processing['Hour'] = data_for_processing.index.hour
    data_for_processing['Day'] = data_for_processing.index.day_name()

    hourly_availability = data_for_processing.groupby('Hour')['Renewable Availability (%)'].mean()
    daily_prices = data_for_processing.groupby('Day')['Energy Price ($/kWh)'].mean()

    optimal_hour = hourly_availability.idxmax()
    optimal_day = daily_prices.idxmin()

    st.write(f"- Best hour to schedule workloads: {optimal_hour}:00 with highest average renewable availability.")
    st.write(f"- Best day to minimize costs: {optimal_day} with lowest average energy price.")

    # Other recommendations
    st.write(f"- Shift workloads to maximize renewable availability above {renewable_threshold * 100:.1f}%.")
    st.write("- Reduce workload during high energy price periods.")
    st.write("- Optimize server usage to decrease overall energy draw.")