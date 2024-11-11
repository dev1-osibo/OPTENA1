# optena_app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Set the page configuration to include the title and favicon
st.set_page_config(page_title="OPTENA", page_icon="favicon.ico")

# Set the page configuration
st.set_page_config(page_title="OPTENA: Data Center Energy Optimization", layout="wide")

# App Header
st.markdown(
    "<h1 style='text-align: center; color: #4CAF50; font-size: 10em;'>OPTENA</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<h2 style='text-align: center; color: #555;'>Data Center Energy Optimization Simulator</h2>",
    unsafe_allow_html=True
)
#st.markdown("""
#<div style='text-align: center; color: #666; font-size: 1.1em; #max-width: 800px; margin: 0 auto;'>
#This app optimizes data center workloads by shifting them to #periods of higher renewable energy availability, reducing costs #and emissions. 
#Adjust parameters and upload data to explore the impact of #optimization in real-time.
#</div>
#""",
#unsafe_allow_html=True
#)

# Function to load data
def load_data(file, file_type="csv"):
    if file_type == "csv":
        data = pd.read_csv(file)
    elif file_type == "h5":
        data = pd.read_hdf(file)
    else:
        raise ValueError("Unsupported file type. Please upload a CSV or HDF5 file.")
    
    # Ensure Timestamp is parsed as datetime
    if 'Timestamp' in data.columns:
        data['Timestamp'] = pd.to_datetime(data['Timestamp'])
        data.set_index('Timestamp', inplace=True)
    
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
    # Step 1: Calculate baseline emissions
    data['Baseline Emissions'] = (
        data['Workload Energy Consumption (kWh)'] * emission_factor_non_renewable * 
        (1 - data['Renewable Availability (%)'] / 100) + 
        data['Workload Energy Consumption (kWh)'] * emission_factor_renewable * 
        (data['Renewable Availability (%)'] / 100)
    )

    # Step 2: Optimize by favoring renewable hours
    data['Optimized Energy'] = np.where(
        data['Renewable Availability (%)'] >= renewable_threshold * 100, 
        data['Workload Energy Consumption (kWh)'], 
        data['Workload Energy Consumption (kWh)'] * 0.9  # Reduce energy usage if renewables are unavailable
    )

    # Recalculate emissions after optimization
    data['Optimized Emissions'] = (
        data['Optimized Energy'] * emission_factor_non_renewable * 
        (1 - data['Renewable Availability (%)'] / 100) + 
        data['Optimized Energy'] * emission_factor_renewable * 
        (data['Renewable Availability (%)'] / 100)
    )

    # Calculate total values
    results = {
        'optimized_energy': data['Optimized Energy'].sum(),
        'optimized_cost': (data['Optimized Energy'] * energy_price_per_kwh).sum(),
        'optimized_emissions': data['Optimized Emissions'].sum(),
        'energy_savings': data['Workload Energy Consumption (kWh)'].sum() - data['Optimized Energy'].sum(),
        'cost_savings': (data['Workload Energy Consumption (kWh)'] * energy_price_per_kwh).sum() - (data['Optimized Energy'] * energy_price_per_kwh).sum(),
        'emissions_savings': data['Baseline Emissions'].sum() - data['Optimized Emissions'].sum(),
    }
    return results

# Sidebar Inputs: File Uploader (supports both .csv and .h5)
uploaded_file = st.sidebar.file_uploader("Upload your data file", type=["csv", "h5"])

# Load data based on file type
if uploaded_file:
    file_type = "csv" if uploaded_file.name.endswith('.csv') else "h5"
    data = load_data(uploaded_file, file_type)
else:
    st.sidebar.write("Using default synthetic dataset.")
    data = load_data('synthetic_data_center_data.csv', file_type="csv")

# Display loaded data preview
st.write("Loaded Data Preview:", data.head())

# Sidebar sliders for adjustable parameters
energy_price_per_kwh = st.sidebar.slider("Energy Price per kWh ($)", min_value=0.05, max_value=1.00, value=0.1, step=0.01)
emission_factor_non_renewable = st.sidebar.slider("Emission Factor for Non-Renewable (kg CO₂ per kWh)", min_value=0.1, max_value=1.0, value=0.5, step=0.05)
emission_factor_renewable = st.sidebar.slider("Emission Factor for Renewable (kg CO₂ per kWh)", min_value=0.0, max_value=0.5, value=0.02, step=0.01)
renewable_threshold = st.sidebar.slider("Renewable Energy Availability Threshold (%)", min_value=0, max_value=100, value=70, step=5) / 100


# Tips Section
st.sidebar.markdown("### Tips")
st.sidebar.markdown(
    """
    **For Maximum Cost Savings**:
    - Use a **high energy price** and a **high renewable threshold** to leverage times when renewable energy is more available, which reduces overall consumption during low-renewable periods.
    
    **For Maximum Emission Reductions**:
    - Set a **high emission factor for non-renewables** and a **low emission factor for renewables**. Combine this with a **high renewable threshold** to shift more energy usage to renewable-heavy periods, maximizing CO? savings.
    
    **Balanced Cost and Emission Reduction**:
    - Adjust the **renewable threshold** based on current emission factors. If emission factors are high for both renewables and non-renewables, a **moderate threshold** can help achieve balanced reductions in both cost and emissions.
    """
)

# Calculate and display baseline metrics
baseline_results = calculate_baseline(data, energy_price_per_kwh, emission_factor_non_renewable)
st.header("Baseline Metrics")
col1, col2, col3 = st.columns(3)
col1.metric("Total Energy Consumption (kWh)", f"{baseline_results['total_energy']:.2f}")
col2.metric("Total Cost ($)", f"{baseline_results['total_cost']:.2f}")
col3.metric("Total CO₂ Emissions (kg)", f"{baseline_results['total_emissions']:.2f}")

# Run the simulation and display optimized metrics
if st.sidebar.button('Run Simulation'):
    with st.spinner('Running simulation...'):
        simulation_results = run_simulation(
            data, 
            renewable_threshold, 
            energy_price_per_kwh, 
            emission_factor_non_renewable, 
            emission_factor_renewable
        )

    # Display optimized results side by side in a consistent layout
    st.header("Optimized Metrics")
    col1, col2, col3 = st.columns(3)

    # Energy Savings
    energy_savings = simulation_results['energy_savings']
    energy_color = "inverse" if energy_savings > 0 else "normal"
    col1.metric(
        "Optimized Energy Consumption (kWh)", 
        f"{simulation_results['optimized_energy']:.2f}", 
        delta=f"{abs(energy_savings):.2f} kWh saved" if energy_savings > 0 else f"{abs(energy_savings):.2f} kWh increased",
        delta_color=energy_color
    )

    # Cost Savings
    cost_savings = simulation_results['cost_savings']
    cost_color = "inverse" if cost_savings > 0 else "normal"
    col2.metric(
        "Optimized Cost ($)", 
        f"{simulation_results['optimized_cost']:.2f}", 
        delta=f"${abs(cost_savings):.2f} saved" if cost_savings > 0 else f"${abs(cost_savings):.2f} increased",
        delta_color=cost_color
    )

    # Emissions Savings
    emissions_savings = simulation_results['emissions_savings']
    emissions_color = "inverse" if emissions_savings > 0 else "normal"
    col3.metric(
        "Optimized CO₂ Emissions (kg)", 
        f"{simulation_results['optimized_emissions']:.2f}", 
        delta=f"{abs(emissions_savings):.2f} kg CO₂ reduced" if emissions_savings > 0 else f"{abs(emissions_savings):.2f} kg CO₂ increased",
        delta_color=emissions_color
    )

    # Display charts comparing baseline and optimized metrics
    #st.subheader("Energy Comparison")
    #st.write("This chart compares baseline vs. optimized energy #consumption over time.")
    #data['Optimized Energy (kWh)'] = #simulation_results['optimized_energy']
    #st.line_chart(data[['Workload Energy Consumption (kWh)', #'Optimized Energy (kWh)']])
