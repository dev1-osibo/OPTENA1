# optena_app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Set the page configuration (only call this once)
st.set_page_config(page_title="OPTENA: Energy Optimization Systems", page_icon="favicon.ico", layout="wide")

# App Header
st.markdown(
    "<h1 style='text-align: center; color: #4CAF50; font-size: 15em;'>OPTENA</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<h2 style='text-align: center; color: #555;'>Data Center Energy Optimization Simulator</h2>",
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
    
    # Ensure Timestamp is parsed as datetime
    if 'Timestamp' in data.columns:
        data['Timestamp'] = pd.to_datetime(data['Timestamp'])
        data.set_index('Timestamp', inplace=True)
    
    return data

# Standardize column names for compatibility
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
        st.error(f"The following required columns are missing: {', '.join(missing_columns)}")
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
        data['Workload Energy Consumption (kWh)'] * emission_factor_non_renewable * 
        (1 - data['Renewable Availability (%)'] / 100) + 
        data['Workload Energy Consumption (kWh)'] * emission_factor_renewable * 
        (data['Renewable Availability (%)'] / 100)
    )

    # Step 2: Optimize by favoring renewable hours
    data['Optimized Energy'] = np.where(
        data['Renewable Availability (%)'] >= renewable_threshold * 100, 
        data['Workload Energy Consumption (kWh)'], 
        data['Workload Energy Consumption (kWh)'] * 0.9
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

# Standardize columns and validate the dataset
data = standardize_columns(data)

# Display loaded data preview
st.write("Loaded Data Preview:", data.head())

# Sidebar sliders dynamically adjusted to data
energy_price_min = data['Energy Price ($/kWh)'].min()
energy_price_max = data['Energy Price ($/kWh)'].max()

renewable_min = data['Renewable Availability (%)'].min()
renewable_max = data['Renewable Availability (%)'].max()

energy_price_per_kwh = st.sidebar.slider("Energy Price per kWh ($)", min_value=round(energy_price_min, 2), max_value=round(energy_price_max, 2), value=round((energy_price_min + energy_price_max) / 2, 2), step=0.01)

emission_factor_non_renewable = st.sidebar.slider("Emission Factor for Non-Renewable (kg CO? per kWh)", min_value=0.1, max_value=1.0, value=0.5, step=0.05)
emission_factor_renewable = st.sidebar.slider("Emission Factor for Renewable (kg CO? per kWh)", min_value=0.0, max_value=0.5, value=0.02, step=0.01)



# Calculate renewable min and max values
renewable_min = data['Renewable Availability (%)'].min()
renewable_max = data['Renewable Availability (%)'].max()

# Debugging: Validate the min and max
st.write("Renewable Min:", renewable_min)
st.write("Renewable Max:", renewable_max)

# Handle invalid or missing data
if renewable_min is None or renewable_max is None or renewable_min > renewable_max:
    st.error("Invalid Renewable Availability data: Check the input dataset.")
    st.stop()

# Provide default values if min == max
if renewable_min == renewable_max:
    renewable_min = 0
    renewable_max = 100

# Define slider with valid range
renewable_threshold = st.sidebar.slider(
    "Renewable Energy Availability Threshold (%)",
    min_value=round(renewable_min, 1),
    max_value=round(renewable_max, 1),
    value=round((renewable_min + renewable_max) / 2, 1),
    step=1.0
) / 100


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
col3.metric("Total CO? Emissions (kg)", f"{baseline_results['total_emissions']:.2f}")

# Run the simulation and display optimized metrics
if st.sidebar.button('Run Simulation'):
    with st.spinner('Running simulation...'):
        simulation_results = run_simulation(data, renewable_threshold, energy_price_per_kwh, emission_factor_non_renewable, emission_factor_renewable)

    # Display optimized results side by side
    st.header("Optimized Metrics")
    col1, col2, col3 = st.columns(3)

    # Energy Savings
    energy_savings = simulation_results['energy_savings']
    energy_color = "inverse" if energy_savings > 0 else "normal"
    col1.metric("Optimized Energy Consumption (kWh)", f"{simulation_results['optimized_energy']:.2f}", delta=f"{abs(energy_savings):.2f} kWh saved", delta_color=energy_color)

    # Cost Savings
    cost_savings = simulation_results['cost_savings']
    cost_color = "inverse" if cost_savings > 0 else "normal"
    col2.metric("Optimized Cost ($)", f"{simulation_results['optimized_cost']:.2f}", delta=f"${abs(cost_savings):.2f} saved", delta_color=cost_color)

    # Emissions Savings
    emissions_savings = simulation_results['emissions_savings']
    emissions_color = "inverse" if emissions_savings > 0 else "normal"
    col3.metric("Optimized CO? Emissions (kg)", f"{simulation_results['optimized_emissions']:.2f}", delta=f"{abs(emissions_savings):.2f} kg CO? reduced", delta_color=emissions_color)

    # Display energy comparison chart
    st.subheader("Energy Comparison")
    plt.figure(figsize=(10, 6))
    plt.plot(data.index, data['Workload Energy Consumption (kWh)'], label='Baseline Energy')
    plt.plot(data.index, data['Optimized Energy'], label='Optimized Energy', linestyle='--')
    plt.xlabel('Time')
    plt.ylabel('Energy Consumption (kWh)')
    plt.title('Baseline vs. Optimized Energy Consumption')
    plt.legend()
    st.pyplot(plt)

# Footer with your name and email
st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        bottom: 0;
        width: 100%;
        text-align: center;
        font-size: small;
        color: grey;
        padding: 10px;
    }
    </style>
    <div class="footer">
        Built by - Babasola Osibo | Email - <a href="mailto:babasolao@optena.app">babasolao@optena.app</a>
    </div>
    """,
    unsafe_allow_html=True
)