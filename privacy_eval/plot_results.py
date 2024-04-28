import matplotlib.pyplot as plt

# Data for plotting
models = ['70M', '160M', '410M', '2.8B', '6.9B']
methods = ['No-RLHF', 'DPO', 'SFT', 'PPO']
colors = ['blue', 'green', 'red', 'purple']  # Different color for each method

accuracy_data = {
    "No-RLHF": [17.4, 10.0, 34.6, 82.4, 91.7],
    "SFT": [0.7, 1.8, 47.9, 88.7, 91.8],
    "PPO": [0.0, 0.0, 44.8, 88.6, 92.9],
    "DPO": [27.4, 3.5, 10.6, 16.5, 91.1]
}

# Plotting with titles
fig, axs = plt.subplots(1, 1, figsize=(8, 10))

for i, (method, values) in enumerate(accuracy_data.items()):
    axs.plot(models, values, label=method, marker='o', linewidth = 5)
axs.set_title('Privacy Leakage', fontsize = 30)
axs.set_xlabel('Model Size', fontsize = 30)
axs.set_ylabel('Accuracy (%)', fontsize = 30)
axs.tick_params(axis='x', which='major', labelsize=25)
axs.tick_params(axis='y', which='major', labelsize=25)
axs.legend()

plt.tight_layout()
plt.savefig("./privacy_plot.pdf")
plt.show()