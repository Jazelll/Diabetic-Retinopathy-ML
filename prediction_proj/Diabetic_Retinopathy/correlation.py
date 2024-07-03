import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 

# Replace 'path/to/your/file.csv' with the actual path to your data
data = pd.read_csv('diabetes_data.csv')

# Create correlation matrix
correlation_matrix = data.corr()

# Create a new figure for the plot
plt.figure(figsize=(12, 10))

# Create a heatmap with annotations (correlation values)
ax = plt.imshow(correlation_matrix, cmap='coolwarm')

# Add colorbar
plt.colorbar(ax)

# Add labels for each cell (feature names)
for (i, j), z in np.ndenumerate(correlation_matrix):
    plt.text(j, i, "{:0.2f}".format(z), ha='center', va='center', fontsize=8)

# Set ticks for x and y axes (feature names)
plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=45, ha='right')
plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)

# Set plot title
plt.title('Correlation Matrix - Early Stage Diabetes')

# Show the plot
plt.tight_layout()
plt.show()


