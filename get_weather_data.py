#!/usr/bin/env python3
"""
get_la_wind_data.py

This script demonstrates how to fetch historical hourly wind data for Los Angeles
using the Meteostat Python library. It retrieves wind speed and wind direction
for a specified time period and plots the results.

Dependencies:
  - meteostat (pip install meteostat)
  - pandas
  - matplotlib
  - numpy
"""

from datetime import datetime
import matplotlib.pyplot as plt
from meteostat import Point, Hourly

# Define the time period for which you want the wind data.
# Here, we use January 1, 2022 as an example.
start = datetime(2025, 1, 7)
end = datetime(2025, 1, 20)

# Create a Point for Los Angeles (latitude, longitude, elevation in meters optionally)
la_point = Point(34.0494, -118.5236)

# Fetch hourly data for the location and time period.
data = Hourly(la_point, start, end)
data = data.fetch()

# Display the first few rows of the data.
print("Retrieved Data:")
print(data.head())

# The data typically includes columns such as:
#  - 'temp': temperature in °C,
#  - 'dwpt': dew point,
#  - 'rhum': relative humidity,
#  - 'prcp': precipitation,
#  - 'wdir': wind direction (degrees),
#  - 'wspd': wind speed (m/s),
#  - 'pres': pressure, etc.
#
# We are interested in 'wdir' (wind direction) and 'wspd' (wind speed).

# Plot the wind speed and wind direction.
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

ax1.plot(data.index, data["wspd"], label="Wind Speed (m/s)", color="blue")
ax1.set_ylabel("Wind Speed (m/s)")
ax1.legend()
ax1.grid(True)

ax2.plot(data.index, data["wdir"], label="Wind Direction (°)", color="orange")
ax2.set_ylabel("Wind Direction (°)")
ax2.set_xlabel("Time")
ax2.legend()
ax2.grid(True)

plt.suptitle("Los Angeles Hourly Wind Data")
plt.show()
