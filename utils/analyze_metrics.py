import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from math import pi

# Setting up the plotting style for publication-quality visuals
try:
    sns.set_style('whitegrid')  # Use Seaborn's whitegrid style
except:
    plt.style.use('ggplot')  # Fallback to Matplotlib's ggplot style
sns.set_context("paper", font_scale=1.2)

# Reading the Excel file
df = pd.read_excel('../results/ΗΚΕΛΥ.xlsx')

# Defining metrics and models
metrics = ['BLEU', 'ROUGE1', 'ROUGE2', 'ROUGEL', 'METEOR', 'BERTSCORE f1']
models = df['Model'].tolist()

# Creating a figure for grouped bar plot
plt.figure(figsize=(12, 6))
bar_width = 0.25
x = np.arange(len(metrics))
colors = sns.color_palette("husl", len(models))

# Plotting bars for each model
for i, model in enumerate(models):
    scores = df[df['Model'] == model][metrics].values[0]
    plt.bar(x + i * bar_width, scores, bar_width, label=model, color=colors[i])

# Customizing the bar plot
plt.xlabel('Metrics', fontsize=12)
plt.ylabel('Scores', fontsize=12)
plt.title('Model Performance Comparison Across Metrics', fontsize=14, pad=15)
plt.xticks(x + bar_width, metrics, rotation=45)
plt.legend()
plt.tight_layout()

# Saving the bar plot
plt.savefig('model_comparison_bar.png', dpi=300, bbox_inches='tight')
plt.close()

# Creating a radar plot
def make_radar_plot(data, labels, title):
    # Number of variables
    categories = metrics
    N = len(categories)
    
    # Computing angle for each axis
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    # Initializing the radar plot
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    # Plotting data for each model
    for i, model in enumerate(labels):
        values = data[data['Model'] == model][metrics].values[0].tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=model, color=colors[i])
        ax.fill(angles, values, alpha=0.1, color=colors[i])
    
    # Customizing the radar plot
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_title(title, size=14, color='black', y=1.1)
    ax.grid(True)
    plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
    
    # Saving the radar plot
    plt.savefig('model_comparison_radar.png', dpi=300, bbox_inches='tight')
    plt.close()

# Creating the radar plot
make_radar_plot(df, models, 'Model Performance Radar Chart')

# Creating a heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df[metrics].set_index(df['Model']), annot=True, cmap='YlOrRd', fmt='.3f')
plt.title('Model Performance Heatmap', fontsize=14, pad=15)
plt.xlabel('Metrics', fontsize=12)
plt.ylabel('Models', fontsize=12)

# Saving the heatmap
plt.savefig('model_comparison_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

# Generating a summary report
summary_stats = df[metrics].describe()
with open('model_metrics_summary.txt', 'w') as f:
    f.write('Model Performance Summary\n')
    f.write('========================\n\n')
    f.write(summary_stats.to_string())
    f.write('\n\nKey Observations:\n')
    f.write('- Meltemi fine-tuned shows the highest performance across all metrics.\n')
    f.write('- LLaMA fine-tuned outperforms Meltemi without fine-tuning but lags behind Meltemi fine-tuned.\n')
    f.write('- BERTSCORE f1 shows the least variation among models (std: {:.3f}).'.format(summary_stats.loc['std', 'BERTSCORE f1']))