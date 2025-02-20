import numpy as np
from matplotlib import pyplot as plt 
from mpl_toolkits import mplot3d
import plotly.graph_objs as go

fig = plt.figure()
vals = np.argwhere(~np.isnan(grid))

vals = vals[vals[:,2].argsort()]
points = [grid[*val] for val in vals]
plt.plot(vals[:, 2], points)
plt.savefig("./vertical_spread.png", dpi=300, bbox_inches="tight")

exit()

###################################################################
x, y, z = np.indices(grid.shape)

# Flatten the arrays to make them suitable for plotting
x_flat = x.flatten()
y_flat = y.flatten()
z_flat = z.flatten()
values_flat = grid.flatten()

# Create a 3D scatter plot
trace = go.Scatter3d(
    x=x_flat,
    y=y_flat,
    z=z_flat,
    mode='markers',
    marker=dict(
        size=5,
        color=values_flat,  # Color by the values in the grid
        colorscale='Viridis',  # You can choose any color scale
        colorbar=dict(title="Values")  # Color bar for the value scale
    )
)

# Define the layout
layout = go.Layout(
    title='3D Scatter Plot of Grid Points',
    scene=dict(
        xaxis_title='X Axis',
        yaxis_title='Y Axis',
        zaxis_title='Z Axis'
    )
)

# Plot the figure
fig = go.Figure(data=[trace], layout=layout)

# # Save the figure to an HTML file (interactive)
# fig.write_html("3d_scatter_plot.html")

# Alternatively, save the figure as a static PNG image (requires the "kaleido" package)
fig.write_image("3d_scatter_plot.png")
########################################################################################