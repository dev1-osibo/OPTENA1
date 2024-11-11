# Define the necessary packages
necessary_packages = {'streamlit', 'pandas', 'numpy', 'matplotlib'}

# Read the existing requirements.txt file
with open('requirements.txt', 'r') as file:
    lines = file.readlines()

# Filter only the necessary packages
filtered_lines = [
    line for line in lines if any(pkg in line for pkg in necessary_packages)
]

# Write the filtered packages back to requirements.txt (or create a new file if you want)
with open('requirements.txt', 'w') as file:
    file.writelines(filtered_lines)

print("Filtered requirements.txt has been saved.")
