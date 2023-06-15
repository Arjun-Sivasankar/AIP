import re
import os
import plotly.graph_objects as go
import numpy as np
from scipy.interpolate import make_interp_spline
import plotly.subplots as sp
import plotly.offline as offline
from scipy.stats import gaussian_kde

# Create a folder to store the plot files
plot_folder = "Plots"
os.makedirs(plot_folder, exist_ok=True)

# Read the contents of the text file
with open('eval_acc3.txt', 'r') as file:
    contents = file.read()

# Define the pattern to match the evaluation accuracy lines
pattern = r"Model: (\w+).*accuracy for (\d+\.\d+): (\d+\.\d+)"

# Find all matches of the pattern in the contents
matches = re.findall(pattern, contents)

# Create a dictionary to store the accuracies for each model and label
accuracies = {}

# Iterate over the matches and store the accuracies in the dictionary
for match in matches:
    model = match[0]
    label = float(match[1])
    accuracy = float(match[2])

    if model not in accuracies:
        accuracies[model] = {}

    accuracies[model][label] = accuracy

# Print the accuracies
for model, labels in accuracies.items():
    print(f"Model: {model}")
    for label, accuracy in labels.items():
        print(f"Interval length {label}: Accuracy => {accuracy}")
    print("--------------------")

# Get the unique labels
labels = np.unique([label for accuracies in accuracies.values() for label in accuracies.keys()])
num_labels = len(labels)

# Set different colors for each model
colors = ['blue', 'orange', 'green', 'red']

# Create a list to store the traces for scatter plots
scatter_traces = []

# Create a trace for each model
for i, (model, acc) in enumerate(accuracies.items()):
    x = list(acc.keys())
    y = list(acc.values())

    # Perform interpolation for smoothening the line
    x_smooth = np.linspace(min(x), max(x), 300)
    spl = make_interp_spline(x, y)
    y_smooth = spl(x_smooth)

    scatter_trace = go.Scatter(
        x=x,
        y=y,
        mode='lines',
        name=model,
        line=dict(color=colors[i], width=2),
        # marker=dict(color=colors[i], size=8),
    )

    scatter_traces.append(scatter_trace)

# Add a horizontal dotted line at 0.9 accuracy
scatter_traces.append(go.Scatter(
    x=x_smooth,
    y=[0.9] * len(x_smooth),
    mode='lines',
    name='0.9 Accuracy',
    line=dict(color='black', dash='dash'),
))

# Create the layout for the scatter plot
layout_scatter = go.Layout(
    title='Evaluation Accuracies vs Interval Lengths :- qsort',
    xaxis=dict(title='Interval lengths'),
    yaxis=dict(title='Evaluation Accuracies'),
)

# Create the figure with scatter traces and layout
fig_scatter = go.Figure(data=scatter_traces, layout=layout_scatter)

# Save the scatter plot as HTML
scatter_plot_filename = os.path.join(plot_folder, 'Scatter Plot_qsort.html')
fig_scatter.write_html(scatter_plot_filename)

# --- Grouped Bar Chart and KDE Plot ---

# Create data for the grouped bar chart and KDE plot
# x_labels = labels
# y_values = [list(accuracies[model].values()) for model in accuracies.keys()]

# Set the width of each bar
bar_width = 0.2

# Calculate the x positions of the bars
bar_positions = np.arange(num_labels)

# Create a list to store the traces for distribution plots
distribution_traces = []

# Create subplots with 1 row and 2 columns
fig = sp.make_subplots(rows=1, cols=2, subplot_titles=('Evaluation accuracies vs Interval Length', 'Kernel Density Estimation (Distribution)'))

# Plot the grouped bar chart
for i, (model, acc) in enumerate(accuracies.items()):
    y = [acc.get(label, 0.0) for label in labels]
    fig.add_trace(go.Bar(x=bar_positions + i * bar_width, y=y, marker_color=colors[i], width=bar_width, name=model), row=1, col=1)

    # Add the data for KDE plot
    data = [acc.get(label, 0.0) for label in labels]
    kde = gaussian_kde(data)
    x_vals = np.linspace(min(data), max(data), 100)
    y_vals = kde.evaluate(x_vals)
    distribution_traces.append(go.Scatter(x=x_vals, y=y_vals, mode='lines', name=model, line=dict(color=colors[i])))

# Add the traces for distribution plots to the second subplot
for trace in distribution_traces:
    fig.add_trace(trace, row=1, col=2)

# Update layout of the first subplot (grouped bar chart)
fig.update_layout(
    xaxis=dict(
        title='Interval lengths',
        tickmode='array',
        tickvals=bar_positions + bar_width * (num_labels - 1) / 2,
        ticktext=labels
    ),
    yaxis=dict(
        title='Evaluation Accuracies'
    ),
    title='Evaluation Accuracies vs Interval Lengths'
)

# Update layout of the second subplot (KDE plot)
fig.update_layout(
    xaxis2=dict(
        title='Evaluation Accuracies'
    ),
    yaxis2=dict(
        title='Density'
    ),
    title=''
)

# Show the figure
fig.show()


# Create the figure with grouped bar chart and KDE plot traces and layout
fig_grouped_bar_kde = go.Figure(fig)

# Save the grouped-bar chart and KDE plot as HTML in the "Plots" folder
grouped_bar_kde_filename = os.path.join(plot_folder, 'Analysis_qsort.html')
fig_grouped_bar_kde.write_html(grouped_bar_kde_filename)
