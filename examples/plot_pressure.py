"""
Example script demonstrating the use of `Pressure` and `Time` classes from the `physical_property` package.
Creates a sinusoidal pressure profile over time and plots it.
"""

import numpy as np
from physical_property import Time, Pressure

# Create a Time instance with 100 points from 0 to 10 seconds
time_values = np.linspace(0, 10, 100)  # 0 to 10 seconds
time = Time(value=time_values, unit="s", name="Elapsed Time")

# Create a Pressure instance with a sinusoidal variation (e.g., 1 to 3 bar)
pressure = Pressure(
    value=2 + np.sin(time_values),  # Mean of 2 bar, amplitude of 1 bar,
    unit="bar",
    name="System Pressure",
)
pressure = pressure.to("psi")  # Convert pressure to psi for plotting
pressure_2 = Pressure(value=2 + np.cos(time_values), unit="psi", name="System Pressure 2")

# Plot Pressure vs. Time
fig = pressure.plot(
    x=time,
    y=pressure_2, # Plot both pressure profiles on the same plot
    title="Pressure Variation Over Time",
    mode="lines"  # Smooth line plot without markers
    
)

# Display the plot
fig.show()

# Optional: Print some info
print(f"Time range: {time.min()} to {time.max()} {time.unit}")
print(f"Pressure range: {pressure.min()} to {pressure.max()} {pressure.unit}")
print(f"Average pressure: {pressure.mean():.2f} {pressure.unit}")