import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
From prophet import Prophet

# Set the page configuration
st.set_page_config(page_title="OPTENA: Energy Optimization Systems", page_icon="favicon.ico", layout="wide")

# App Header
st.markdown(
    "<h1 style='text-align: center; color: #4CAF50; font-size: 15em;'>OPTENA</h1>",
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

# Baseline calculation function
def calculate_baseline(data, energy_price_per_kwh, emission_factor_non_renewable):
    total_energy = data['Workload Energy Consumption (kWh)'].sum()
    total_cost = total_energy * energy_price_per_kwh
    total_emissions = total_energy * emission_factor_non_renewable
    return {
        'total_energy': total_energy,
        'total_cost': total_cost,
        'total_emissions': total_emissions
    }

# Simulation function for optimization
def run_simulation(data, renewable_threshold, energy_price_per_kwh, emission_factor_non_renewable, emission_factor_renewable):
    # Adjust threshold if it's below data's minimum
    if renewable_threshold < data['Renewable Availability (%)'].min():
        st.warning("Renewable threshold is lower than the data's minimum renewable availability. Using minimum value as threshold.")
        renewable_threshold = data['Renewable Availability (%)'].min()

    # Step 1: Calculate baseline emissions
    data['Baseline Emissions'] = (
        data['Workload Energy Consumption (kWh)'] * emission_factor_non_renewable * (1 - data['Renewable Availability (%)'] / 100) +
        data['Workload Energy Consumption (kWh)'] * emission_factor_renewable * (data['Renewable Availability (%)'] / 100)
    )

    # Step 2: Optimize by favoring renewable hours
    data['Optimized Energy'] = np.where(
        data['Renewable Availability (%)'] >= renewable_threshold * 100,
        data['Workload Energy Consumption (kWh)'],
        data['Workload Energy Consumption (kWh)'] * 0.9  # Example optimization factor
    )

    # Recalculate emissions after optimization
    data['Optimized Emissions'] = (
        data['Optimized Energy'] * emission_factor_non_renewable * (1 - data['Renewable Availability (%)'] / 100) +
        data['Optimized Energy'] * emission_factor_renewable * (data['Renewable Availability (%)'] / 100)
    )

    # Calculate total values after optimization
    results = {
        'optimized_energy': data['Optimized Energy'].sum(),
        'optimized_cost': (data['Optimized Energy'] * energy_price_per_kwh).sum(),
        'optimized_emissions': data['Optimized Emissions'].sum(),
        'energy_savings': data['Workload Energy Consumption (kWh)'].sum() - data['Optimized Energy'].sum(),
        'cost_savings': (data['Workload Energy Consumption (kWh)'] * energy_price_per_kwh).sum() - 
                        (data['Optimized Energy'] * energy_price_per_kwh).sum(),
        'emissions_savings': data['Baseline Emissions'].sum() - data['Optimized Emissions'].sum(),
    }
    
    return results

# Forecasting function using ARIMA
#def forecast_arima(series, periods=365*24):
    #model = ARIMA(series, order=(5, 1, 0))
    #model_fit = model.fit()
    #forecast = model_fit.forecast(steps=periods)
    
    #return forecast

# Sidebar Inputs: File Uploader (supports both .csv and .h5)
uploaded_file = st.sidebar.file_uploader("Upload your data file", type=["csv", "h5"])
if uploaded_file:
    file_type = "csv" if uploaded_file.name.endswith('.csv') else "h5"
    data = load_data(uploaded_file, file_type)
else:
    st.sidebar.write("Using default synthetic dataset.")
    data = load_data('synthetic_data_center_data.csv', file_type="csv")

# Standardize columns and validate the dataset
data = standardize_columns(data)

# Display loaded data preview
st.write("Loaded Data Preview:", data.head())

# Sidebar sliders dynamically adjusted to dataset values
energy_price_min = round(data['Energy Price ($/kWh)'].min(), 2)
energy_price_max = round(data['Energy Price ($/kWh)'].max(), 2)
renewable_min = round(data['Renewable Availability (%)'].min(), 1)
renewable_max = round(data['Renewable Availability (%)'].max(), 1)

energy_price_per_kwh = st.sidebar.slider("Energy Price per kWh ($)", min_value=energy_price_min,
                                         max_value=energy_price_max,
                                         value=round((energy_price_min + energy_price_max) / 2, 2), step=0.01)

emission_factor_non_renewable = st.sidebar.slider("Emission Factor for Non-Renewables (kg CO? per kWh)", 
                                                  min_value=0.1, max_value=1.0, value=0.5, step=0.05)

emission_factor_renewable = st.sidebar.slider("Emission Factor for Renewables (kg CO? per kWh)", 
                                              min_value=0.0, max_value=0.5, value=0.02, step=0.01)

renewable_threshold = st.sidebar.slider("Renewable Energy Availability Threshold (%)", 
                                        min_value=renewable_min,
                                        max_value=renewable_max,
                                        value=(renewable_min + renewable_max) / 2,
                                        step=1.0) / 100

# Calculate baseline metrics
baseline_results = calculate_baseline(data, energy_price_per_kwh, emission_factor_non_renewable)

st.header("Baseline Metrics")
col1, col2, col3 = st.columns(3)
col1.metric("Total Energy Consumption (kWh)", f"{baseline_results['total_energy']:.2f}")
col2.metric("Total Cost ($)", f"{baseline_results['total_cost']:.2f}")
col3.metric("Total CO? Emissions (kg)", f"{baseline_results['total_emissions']:.2f}")

# Run simulation and display optimized metrics
if st.sidebar.button('Run Simulation'):
    
    with st.spinner('Running simulation...'):
        simulation_results = run_simulation(data, renewable_threshold,
                                            energy_price_per_kwh,
                                            emission_factor_non_renewable,
                                            emission_factor_renewable)

        # Display optimized results side by side with baseline results
        st.header("Optimized Metrics")
        
        col1.metric("Optimized Energy Consumption (kWh)", f"{simulation_results['optimized_energy']:.2f}",
                    delta=f"{simulation_results['energy_savings']:.2f} kWh saved")
        
        col2.metric("Optimized Cost ($)", f"{simulation_results['optimized_cost']:.2f}",
                    delta=f"${simulation_results['cost_savings']:.2f} saved")
        
        col3.metric("Optimized CO? Emissions (kg)", f"{simulation_results['optimized_emissions']:.2f}",
                    delta=f"{simulation_results['emissions_savings']:.2f} kg CO? reduced")

# Forecast future metrics using ARIMA model
#if st.sidebar.button('Forecast Future Metrics'):
    
    #with st.spinner('Forecasting future metrics...'):
        
        # Forecast energy consumption using ARIMA model for one #year ahead (365*24 hours)
        #energy_forecast = forecast_arima(data['Workload Energy #Consumption (kWh)'], periods=365*24)
        
        #cost_forecast = energy_forecast * energy_price_per_kwh
        #emissions_forecast = energy_forecast * #emission_factor_non_renewable

# Forecasting function using Prophet
def forecast_prophet(data, column, periods=365*24):
    """
    Forecast future values using Prophet.

    Parameters:
        data (pd.DataFrame): DataFrame with 'Timestamp' and the column to forecast.
        column (str): Name of the column to forecast.
        periods (int): Number of future periods (hours) to forecast.

    Returns:
        pd.DataFrame: DataFrame with the forecasted values.
    """
    # Prepare data for Prophet
    prophet_data = data.reset_index()[['Timestamp', column]].rename(columns={'Timestamp': 'ds', column: 'y'})
    
    # Initialize and fit the Prophet model
    model = Prophet()
    model.fit(prophet_data)
    
    # Create a DataFrame for future periods
    future = model.make_future_dataframe(periods=periods, freq='H')
    
    # Generate forecast
    forecast = model.predict(future)
    return forecast
        
        # Plot forecasted results for energy consumption, cost and #emissions
        
        #plt.figure(figsize=(10, 6))
        #plt.plot(energy_forecast.index, energy_forecast.values, #label='Forecasted Energy')
        #plt.title('Forecasted Energy Consumption')
        #plt.xlabel('Time')
        #plt.ylabel('Energy Consumption (kWh)')
        #plt.legend()
        #st.pyplot(plt)