# 1. Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

# 2. Load Dataset
df = pd.read_csv("india_2000_2024_daily_weather.csv")

df.head()

# 3. Explore Dataset
print(df.info())

print(df.describe())

print(df.isnull().sum())

# 4. Data Cleaning
df.drop_duplicates(inplace=True)

df['date'] = pd.to_datetime(df['date'])

df['Year'] = df['date'].dt.year

df['Month'] = df['date'].dt.month

# 5. Select One City
city = "Ahmedabad"

city_df = df[df['city'] == city]

# 6. Yearly Temperature Trend
year_temp = city_df.groupby("Year")["temperature_2m_max"].mean()

plt.figure(figsize=(12,5))

plt.plot(year_temp.index, year_temp.values, marker='o')

plt.title("Average Yearly Temperature")

plt.xlabel("Year")

plt.ylabel("Temperature (°C)")

plt.show()

# 7. Yearly Rainfall Trend
year_rain = city_df.groupby("Year")["rain_sum"].sum()

plt.figure(figsize=(12,5))

plt.plot(year_rain.index, year_rain.values, color="green", marker="o")

plt.title("Yearly Rainfall")

plt.xlabel("Year")

plt.ylabel("Rainfall (mm)")

plt.show()

# 8. Monthly Seasonal Temperature
monthly = city_df.groupby("Month")["temperature_2m_max"].mean()

plt.figure(figsize=(10,5))

sns.lineplot(x=monthly.index, y=monthly.values)

plt.title("Seasonal Temperature Variation")

plt.show()

# 9. Monthly Rainfall
monthly_rain = city_df.groupby("Month")["rain_sum"].mean()

plt.figure(figsize=(10,5))

sns.barplot(x=monthly_rain.index, y=monthly_rain.values)

plt.title("Average Monthly Rainfall")

plt.show()

# 10. Heatmap
pivot = city_df.pivot_table(
    values="temperature_2m_max",
    index="Month",
    columns="Year",
    aggfunc="mean"
)

plt.figure(figsize=(15,6))

sns.heatmap(pivot, cmap="coolwarm")

plt.title("Temperature Heatmap")

plt.show()

# 11. Boxplot(Anamolies)
plt.figure(figsize=(10,5))

sns.boxplot(data=city_df, x="Month", y="temperature_2m_max")

plt.title("Temperature Anomalies")

plt.show()

# 12. Rainfall vs Temperature
plt.figure(figsize=(8,6))

sns.scatterplot(
    data=city_df,
    x="rain_sum",
    y="temperature_2m_max"
)

plt.title("Rainfall vs Temperature")

plt.show()

# 13. Correlation
corr = city_df[['temperature_2m_max','rain_sum']].corr()

sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm"
)

plt.show()

# 14. Summary
print("Average Temperature :", city_df["temperature_2m_max"].mean())

print("Highest Temperature :", city_df["temperature_2m_max"].max())

print("Lowest Temperature :", city_df["temperature_2m_max"].min())

print("Average Rainfall :", city_df["rain_sum"].mean())