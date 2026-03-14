import pandas as pd
import numpy as np

print("Generating blind live telemetry data...")

# Generate time arrays
t_live = np.linspace(0, 100, 4000)

# Simulate 3000 readings of a healthy bridge, followed by 1000 readings of sudden damage
normal_phase = np.sin(2 * np.pi * 5 * t_live[:3000]) + np.random.normal(0, 0.4, 3000)
damaged_phase = 2.0 * np.sin(2 * np.pi * 4.2 * t_live[3000:]) + 0.8 * np.sin(2 * np.pi * 12 * t_live[3000:]) + np.random.normal(0, 1.2, 1000)

live_readings = np.concatenate([normal_phase, damaged_phase])

# Create DataFrame with ONLY the sensor readings. Notice there is NO 'Status' column here!
df_live = pd.DataFrame({'Sensor_Reading': live_readings})
df_live.to_csv('live_telemetry.csv', index=False)

print("live_telemetry.csv generated successfully! This data is completely unlabeled.")