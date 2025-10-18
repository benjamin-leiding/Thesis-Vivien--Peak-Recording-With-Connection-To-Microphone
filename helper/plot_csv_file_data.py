import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("peak_log.csv")

# plot everything
plt.scatter(df["time"], df["amplitude"], c=df["threshold"], cmap="viridis", s=20)
plt.colorbar(label="Threshold")
plt.xlabel("Time (s)")
plt.ylabel("Peak amplitude")
plt.title("Detected Peaks over Time")
plt.show()
