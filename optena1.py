import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet

# Set the page configuration
st.set_page_config(page_title="OPTENA: Energy Optimization Systems", page_icon="favicon.ico", layout="wide")

# App Header
st.markdown(
    "<h1 style='text-align: center; color: #4CAF50; font-size: 10em;'>OPTENA</h1>",
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
    if renewable_threshold < data['Renewable Availability (%)'].min():
        st.warning("Renewable threshold is lower than the data's minimum renewable availability. Using minimum value as threshold.")
        renewable_threshold = data['Renewable Availability (%)'].min()

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
        'optimized_energy': data['Optimized Energy'].sum(),
        'optimized_cost': (data['Optimized Energy'] * energy_price_per_kwh).sum(),
        'optimized_emissions': data['Optimized Emissions'].sum(),
        'energy_savings': data['Workload Energy Consumption (kWh)'].sum() - data['Optimized Energy'].sum(),
        'cost_savings': (data['Workload Energy Consumption (kWh)'] * energy_price_per_kwh).sum() - 
                        (data['Optimized Energy'] * energy_price_per_kwh).sum(),
        'emissions_savings': data['Baseline Emissions'].sum() - data['Optimized Emissions'].sum(),
    }
    return results

# Main Interface
uploaded_file = st.sidebar.file_uploader("Upload your data file", type=["csv", "h5"])

# Tabs for features
tabs = st.tabs(["Energy Optimization", "Predictive Maintenance", "Workload Distribution"])

# Sidebar Parameters Header
st.sidebar.header("Feature Parameters")

# Placeholder for data
data = None

if uploaded_file:
    file_type = "csv" if uploaded_file.name.endswith('.csv') else "h5"
    data = load_data(uploaded_file, file_type)
    data = standardize_columns(data)

# Tab 1: Energy Optimization
with tabs[0]:
    st.subheader("Energy Optimization")
    
    if data is not None:
        st.write("Loaded Data Preview:", data.head())

        # Sidebar controls for Energy Optimization
        energy_price_per_kwh = st.sidebar.slider("Energy Price per kWh ($)", min_value=0.05, max_value=1.00, value=0.1, step=0.01)
        emission_factor_non_renewable = st.sidebar.slider("Emission Factor for Non-Renewables (kg CO2 per kWh)", 
                                                          min_value=0.1, max_value=1.0, value=0.5, step=0.05)
        emission_factor_renewable = st.sidebar.slider("Emission Factor for Renewables (kg CO2 per kWh)", 
                                                      min_value=0.0, max_value=0.5, value=0.02, step=0.01)
        renewable_threshold = st.sidebar.slider("Renewable Energy Availability Threshold (%)", 
                                                min_value=0, max_value=100, value=70, step=5) / 100

        # Baseline Metrics
        baseline_results = calculate_baseline(data, energy_price_per_kwh, emission_factor_non_renewable)
        st.header("Baseline Metrics")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Energy Consumption (kWh)", f"{baseline_results['total_energy']:.2f}")
        col2.metric("Total Cost ($)", f"{baseline_results['total_cost']:.2f}")
        col3.metric("Total CO2 Emissions (kg)", f"{baseline_results['total_emissions']:.2f}")

        # Simulation Button and Results
        if st.sidebar.button('Run Simulation'):
            simulation_results = run_simulation(data, renewable_threshold, energy_price_per_kwh, emission_factor_non_renewable, emission_factor_renewable)
            st.header("Optimized Metrics")
            col1.metric("Optimized Energy Consumption (kWh)", f"{simulation_results['optimized_energy']:.2f}")
            col2.metric("Optimized Cost ($)", f"{simulation_results['optimized_cost']:.2f}")
            col3.metric("Optimized CO2 Emissions (kg)", f"{simulation_results['optimized_emissions']:.2f}")

    else:
        st.warning("Please upload a valid data file to proceed.")

# Tab 2: Predictive Maintenance
with tabs[1]:
    st.subheader("Predictive Maintenance")
    st.write("This feature is under development. Please check back later!")

# Tab 3: Workload Distribution
with tabs[2]:
    st.subheader("Workload Distribution")
    st.write("This feature is under development. Please check back later!")