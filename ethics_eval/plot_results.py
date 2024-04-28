import matplotlib.pyplot as plt
import numpy as np

# Data for plotting
models = ['70M', '160M', '410M', '2.8B', '6.9B']
methods = ['No-RLHF', 'SFT', 'DPO', 'PPO']
colors = ['blue', 'green', 'red', 'purple']  # Different color for each method

benign_data = {
    "No-RLHF": np.array([0.001, 0.042, 0.214, 0.969, 0.378]),
    "SFT": np.array([0.000, 0.009, 0.116, 0.477, 0.211]),
    "PPO": np.array([0.027, 0.004, 0.093, 0.210, 0.205]),
    "DPO": np.array([0.000, 0.003, 0.701, 0.978, 0.539])
}

adversarial_data = {
    "No-RLHF": np.array([0.001, 0.119, 0.582, 0.982, 0.936]),
    "SFT": np.array([0.000, 0.063, 0.175, 0.699, 0.937]),
    "PPO": np.array([0.102, 0.010, 0.198, 0.461, 0.851]),
    "DPO": np.array([0.000, 0.003, 0.882, 0.984, 0.986])
}

# Plotting with titles
fig, axs = plt.subplots(1, 2, figsize=(20, 10))

for i, (method, values) in enumerate(benign_data.items()):
    axs[0].plot(models, values, label=method, marker='o', linewidth=5)
axs[0].set_title('Benign System Prompt', fontsize = 25)
axs[0].set_xlabel('Model Size', fontsize = 25)
axs[0].set_ylabel('False Positive Rate For Ethics Statements', fontsize = 25)
axs[0].tick_params(axis='x', which='major', labelsize=20)
axs[0].tick_params(axis='y', which='major', labelsize=20)
axs[0].legend(fontsize = 25)

for i, (method, values) in enumerate(adversarial_data.items()):
    axs[1].plot(models, values, label=method, marker='o', linewidth=5)
axs[1].set_title('Adversarial System Prompt', fontsize = 25)
axs[1].set_xlabel('Model Size', fontsize = 25)
#axs[1].set_ylabel('False Positive Rate For Ethics Statements', fontsize = 25)
axs[1].tick_params(axis='x', which='major', labelsize=20)
axs[1].tick_params(axis='y', which='major', labelsize=20)
axs[1].legend(fontsize = 25)

plt.tight_layout()
plt.savefig("./ethics_plots.pdf",  format='pdf', bbox_inches='tight')
plt.show()