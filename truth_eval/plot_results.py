import matplotlib.pyplot as plt

# Data for plotting
models = ['70M', '160M', '410M', '2.8B', '6.9B']
methods = ['No-RLHF', 'SFT', 'DPO', 'PPO']
colors = ['blue', 'green', 'red', 'purple']  # Different color for each method

accuracy_data = {
    "No-RLHF": [4.4, 5.9, 8.5, 36.8, 37.9],
    "SFT": [0.0, 0.0, 2.6, 13.8, 25.6],
    "PPO": [0.0, 0.0, 3.4, 10.2, 24.2],
    "DPO": [2.9, 5.8, 6.6, 34.1, 30.5]
}

# Plotting with titles
fig, axs = plt.subplots(1, 1, figsize=(8, 10))

for i, (method, values) in enumerate(accuracy_data.items()):
    axs.plot(models, values, label=method, marker='o', linewidth = 5)
axs.set_title('Model Truthfulness', fontsize = 30)
axs.set_xlabel('Model Size', fontsize = 30)
axs.set_ylabel('Accuracy (%)', fontsize = 30)
axs.tick_params(axis='x', which='major', labelsize=25)
axs.tick_params(axis='y', which='major', labelsize=25)
axs.legend()

plt.savefig("truthfulness_plot.pdf",  format='pdf', bbox_inches='tight')
plt.show()