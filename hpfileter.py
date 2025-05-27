import pandas as pd
from fredapi import Fred
from statsmodels.tsa.filters.hp_filter import hpfilter
import matplotlib.pyplot as plt
import numpy as np

# 1. 国の選択とGDPデータの取得
# Choosing the United States and fetching Real Gross Domestic Product (GDPC1)
fred = Fred(api_key='c5932312951035b709f0a329cb5ad044') # Replace with your FRED API key
gdp_data = fred.get_series('RGDPNANZA666NRUG') # Using GDPC1 for US Quarterly Real GDP

# Ensure the data is a pandas Series and sort by index (time)
gdp_data = gdp_data.sort_index()

# Remove potential NaNs at the end
gdp_data = gdp_data.dropna()

# 2. 対数変換
log_gdp = np.log(gdp_data)

# 3. HPフィルターの適用とλの検討
lambdas = [10, 100, 1600]
gdp_trends = {}
gdp_cycles = {}

for lam in lambdas:
  gdp_trend, gdp_cycle = hpfilter(log_gdp, lamb=lam)
  gdp_trends[lam] = gdp_trend
  gdp_cycles[lam] = gdp_cycle

# 4. 可視化

# グラフ1：元のデータとトレンド成分の比較
plt.figure(figsize=(12, 6))
plt.plot(log_gdp.index, log_gdp, label='Original Log GDP', color='black', linewidth=1.5)
for lam in lambdas:
  plt.plot(gdp_trends[lam].index, gdp_trends[lam], label=f'Trend (λ={lam})', linestyle='--')

plt.title('Comparison of Original Log GDP and HP Filtered Trend Components (New Zealand)')
plt.xlabel('Date')
plt.ylabel('Log GDP')
plt.legend()
plt.grid(True)
plt.show()

# グラフ2：循環成分の比較
plt.figure(figsize=(12, 6))
for lam in lambdas:
  plt.plot(gdp_cycles[lam].index, gdp_cycles[lam], label=f'Cycle (λ={lam})')

plt.title('Comparison of HP Filtered Cycle Components (New Zealand)')
plt.xlabel('Date')
plt.ylabel('Log GDP Cycle')
plt.legend()
plt.grid(True)
plt.show()
