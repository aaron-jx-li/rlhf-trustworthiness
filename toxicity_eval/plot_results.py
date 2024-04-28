import matplotlib.pyplot as plt

# Data for plotting
models = ['70M', '160M', '410M', '2.8B', '6.9B']
colors = ['blue', 'green', 'red', 'purple', 'orange']  # Different color for each method
titles = {
    # "Table 1": "Benign System Prompt & Toxic User Prompt",
    # "Table 2": "Benign System Prompt & Non-Toxic User Prompt",
    # "Table 3": "Toxic User Prompt",
    # "Table 4": "Non-Toxic User Prompt",
    "Table 5": "Performance of Toxicity Probe",
    "Table 6": "Probability of Generated Tokens"
}
data = {
    # "Table 1": {
    #     "Baseline": [0.302, 0.389, 0.460, 0.480, 0.506],
    #     "SFT": [0.270, 0.396, 0.460, 0.495, 0.541],
    #     "PPO": [0.075, 0.289, 0.479, 0.618, 0.597],
    #     "DPO": [0.207, 0.264, 0.319, 0.396, 0.497]
    # },
    # "Table 2": {
    #     "Baseline": [0.087, 0.100, 0.110, 0.121, 0.126],
    #     "SFT": [0.092, 0.108, 0.122, 0.132, 0.130],
    #     "PPO": [0.040, 0.072, 0.126, 0.167, 0.146],
    #     "DPO": [0.080, 0.090, 0.100, 0.116, 0.122]
    # },
    # "Table 3": {
    #     "No-RLHF": [0.364, 0.413, 0.478, 0.494, 0.513],
    #     "SFT": [0.306, 0.408, 0.476, 0.527, 0.545],
    #     "PPO": [0.082, 0.324, 0.481, 0.639, 0.611],
    #     "DPO": [0.226, 0.288, 0.328, 0.418, 0.498]
    # },
    # "Table 4": {
    #     "No-RLHF": [0.112, 0.101, 0.115, 0.127, 0.137],
    #     "SFT": [0.106, 0.109, 0.126, 0.143, 0.142],
    #     "PPO": [0.043, 0.078, 0.129, 0.180, 0.157],
    #     "DPO": [0.091, 0.093, 0.102, 0.125, 0.130]
    # },
    "Table 5": {
        "No-RLHF": [68.8, 69.7, 70.8, 77.9, 78.4],
        "SFT":     [70.8, 70.0, 71.6, 81.1, 79.4],
        "PPO":     [68.9, 66.9, 72.4, 80.9, 81.1],
        "DPO":     [67.9, 68.9, 71.8, 77.3, 78.2]
    },
    "Table 6": {
        "No-RLHF": [-5.00, -4.99, -6.15, -6.78, -6.55],
        "SFT": [-7.52, -6.51, -6.52, -6.54, -6.48],
        "PPO": [-10.43, -8.75, -6.11, -5.43, -6.40],
        "DPO": [-7.88, -6.83, -6.75, -7.40, -7.02]
    }

}


# Plotting with titles
fig, axs = plt.subplots(1, 2, figsize=(20, 10))

for i, (table_name, table_data) in enumerate(data.items()):
    for j, (method, values) in enumerate(table_data.items()):
        axs[i].plot(models, values, label=method, marker='o', linewidth = 5)
    axs[i].set_title(titles[table_name], fontsize = 25)
    axs[i].set_xlabel('Model Size', fontsize = 25)
    if i == 0:
        axs[i].set_ylabel('Classification Accuracy', fontsize = 25)
    else:
        axs[i].set_ylabel('Mean Log-likelihood', fontsize = 25)
    axs[i].tick_params(axis='x', which='major', labelsize=20)
    axs[i].tick_params(axis='y', which='major', labelsize=20)
    
    axs[i].legend(fontsize = 25)

#plt.tight_layout()

plt.savefig("toxicity_analysis.pdf",  format='pdf', bbox_inches='tight')
plt.show()
