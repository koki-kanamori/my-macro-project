
import numpy as np
!pip install fredapi
!pip install statsmodels

import pandas as pd
from fredapi import Fred
import statsmodels.api as sm
import matplotlib.pyplot as plt

# 1. 国の選択とGDPデータの取得
# Choosing New Zealand and fetching Real Gross Domestic Product
fred = Fred(api_key='c5932312951035b709f0a329cb5ad044') # Replace with your FRED API key
# Using RGDPNANZA666NRUG for New Zealand Annual Real GDP
nz_gdp_data = fred.get_series('RGDPNANZA666NRUG')
# Ensure the data is a pandas Series and sort by index (time)
nz_gdp_data = nz_gdp_data.sort_index()
# Remove potential NaNs
nz_gdp_data = nz_gdp_data.dropna()

# 2. 選んだ国の対数実質GDPにHP-filterをかけ、循環変動成分およびトレンド成分に分解
# Apply log transformation
nz_log_gdp = np.log(nz_gdp_data)
# Apply HP filter (lambda=1600 for quarterly data, but this is annual data so adjust lambda)
# For annual data, lambda is typically 6.25
lamb_annual = 6.25
nz_cycle, nz_trend = sm.tsa.filters.hpfilter(nz_log_gdp, lamb=lamb_annual)

# 3. 日本についても同じように、対数実質GDPにHP-filterをかけて循環変動成分およびトレンド成分に分解
# Fetch Japan's Real Gross Domestic Product (Real GDP in Billion Yen, Seasonally Adjusted Quarterly)
# I will use the JPNRGDPEXP series again as it worked previously.
jp_gdp_data = fred.get_series('JPNRGDPEXP')
# Ensure the data is a pandas Series and sort by index (time)
jp_gdp_data = jp_gdp_data.sort_index()
# Remove potential NaNs
jp_gdp_data = jp_gdp_data.dropna()

# Align the two series based on common dates
# Note: NZ data is annual, JP data is quarterly. This alignment will only keep the years present in both datasets.
common_index = nz_log_gdp.index.intersection(np.log(jp_gdp_data).resample('AS').first().index)
nz_log_gdp_aligned = nz_log_gdp[common_index]
jp_log_gdp_aligned_annual = np.log(jp_gdp_data).resample('AS').first()[common_index]


# Apply HP filter to Japan's log GDP (using annual data aligned with NZ)
jp_cycle, jp_trend = sm.tsa.filters.hpfilter(jp_log_gdp_aligned_annual, lamb=lamb_annual)
# Align Japan's cycle component to the common index as well
jp_cycle = jp_cycle[common_index]


# 4. 選んだ国および日本について循環変動成分の標準偏差を計算して比較するほか、選んだ国と日本の間の循環変動成分の相関係数を計算
nz_cycle_std = nz_cycle.std()
jp_cycle_std = jp_cycle.std()

correlation = nz_cycle.corr(jp_cycle)

print(f"New Zealand Cycle Component Standard Deviation: {nz_cycle_std:.4f}")
print(f"Japan Cycle Component Standard Deviation (Annual): {jp_cycle_std:.4f}")
print(f"Correlation between New Zealand (Annual) and Japan (Annual) Cycle Components: {correlation:.4f}")

# 5. 選んだ国および日本について循環変動成分の時系列データを一つのグラフ上にプロットして比較
plt.figure(figsize=(12, 6))
plt.plot(nz_cycle.index, nz_cycle, label='New Zealand Cycle Component (Annual)')
plt.plot(jp_cycle.index, jp_cycle, label='Japan Cycle Component (Annual)')
plt.title('Real GDP Cycle Components: New Zealand vs. Japan (Annual Data)')
plt.xlabel('Date')
plt.ylabel('Log Deviation from Trend')
plt.legend()
plt.grid(True)
plt.show()
