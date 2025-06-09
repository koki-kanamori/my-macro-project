import pandas as pd
import numpy as np

# Assume we have a hypothetical dataset for illustration purposes
# Replace this with actual data loaded from your preferred source (e.g., CSV, Google Sheet)
# This data represents simplified values for GDP, Capital, and Labor over time.
# In a real scenario, you would need to source proper economic data.
data = {
    'Year': range(1990, 2020),
    'GDP': np.linspace(100, 500, 30),  # Example: GDP grows over time
    'Capital': np.linspace(200, 800, 30), # Example: Capital grows over time
    'Labor': np.linspace(50, 150, 30)   # Example: Labor grows over time
}

df = pd.DataFrame(data)

# Calculate the growth rates (log differences)
df['GDP_growth'] = np.log(df['GDP']).diff() * 100
df['Capital_growth'] = np.log(df['Capital']).diff() * 100
df['Labor_growth'] = np.log(df['Labor']).diff() * 100

# Assume alpha (capital share) and beta (labor share) are constant
# In a real growth accounting exercise, these would be estimated or taken from national accounts data.
alpha = 0.3 # Example value for capital share
beta = 0.7  # Example value for labor share (assuming constant returns to scale, alpha + beta = 1)

# Calculate the contribution of capital and labor to growth
df['Capital_contribution'] = alpha * df['Capital_growth']
df['Labor_contribution'] = beta * df['Labor_growth']

# Calculate Total Factor Productivity (TFP) growth as the residual
df['TFP_growth'] = df['GDP_growth'] - df['Capital_contribution'] - df['Labor_contribution']

# Drop the first row which has NaN values due to differencing
df = df.dropna().reset_index(drop=True)

# Calculate the average growth rates over the sample period (1990-2019, effectively 1991-2019 after differencing)
average_gdp_growth = df['GDP_growth'].mean()
average_capital_growth = df['Capital_growth'].mean()
average_labor_growth = df['Labor_growth'].mean()
average_capital_contribution = df['Capital_contribution'].mean()
average_labor_contribution = df['Labor_contribution'].mean()
average_tfp_growth = df['TFP_growth'].mean()

# Create the table structure
table_data = {
    'Growth component': ['Real GDP growth', 'Contribution of Capital', 'Contribution of Labor', 'Total Factor Productivity'],
    'Average Annual Growth Rate (%)': [
        average_gdp_growth,
        average_capital_contribution,
        average_labor_contribution,
        average_tfp_growth
    ]
}

table_df = pd.DataFrame(table_data)

# Format the table for better presentation
table_df['Average Annual Growth Rate (%)'] = table_df['Average Annual Growth Rate (%)'].map('{:.2f}'.format)

# Print the table
print("Growth Accounting Decomposition (1990-2019)")
print(table_df.to_string(index=False))
