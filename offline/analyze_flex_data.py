import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

folder = Path.home() / "tremorgate_ws/offline"
csv_path = folder / "flex_sensor_data.csv"
df = pd.read_csv(csv_path)

sensors = ["thumb", "index", "middle", "ring", "little"]
raw_cols = [f"{s}_raw" for s in sensors]
delta_cols = [f"{s}_delta" for s in sensors]

print("\n=== TRE MORGATE FLEX SENSOR ANALYSIS ===")
print(f"Total samples collected: {len(df)}")
print(f"Movement-status counts:\n{df['movement_status'].value_counts()}")
print("\nSensor summary:")
print(df[raw_cols].describe().round(2).to_string())
print("\nMaximum absolute deviation from baseline:")
print(df[delta_cols].abs().max().round(2).to_string())

df["elapsed_seconds"] = (
    pd.to_datetime(df["timestamp"]) -
    pd.to_datetime(df["timestamp"]).iloc[0]
).dt.total_seconds()

plt.figure(figsize=(12, 6))
for sensor in sensors:
    plt.plot(
        df["elapsed_seconds"],
        df[f"{sensor}_raw"],
        label=sensor.capitalize()
    )

plt.xlabel("Elapsed time (seconds)")
plt.ylabel("Arduino ADC reading")
plt.title("TremorGate: Live Five-Finger Flex Sensor Readings")
plt.legend()
plt.grid(True)
plt.tight_layout()

graph_path = folder / "flex_sensor_plot.png"
plt.savefig(graph_path, dpi=150)
print(f"\nGraph saved to: {graph_path}")
plt.show()
