import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider
import matplotlib.cm as cm

def plot_random_walk(ax, ax_dist, ax_percent, size, steps, bin_width=3, interval=2, dim=3, catch=True):
    # Clear previous plots
    ax.clear()
    ax_dist.clear()
    ax_percent.clear()
    
    # Initialize points and storage for positions over time
    x = np.zeros((dim, size), dtype=np.int32)
    positions = np.zeros((steps, dim, size))  # Shape: (steps, (x, y, z), number of points)
    truth = np.ones((dim, size), dtype=bool)
    x += np.random.randint(low=-1, high=2, size=(dim, size)) * 2

    # Store percentage of points within distance of 1 from the origin
    percent_within_1 = np.zeros(steps)

    # Simulate and store positions over time
    catchdist = np.sqrt(dim)
    distances = np.linalg.norm(x, axis=0)
    for t in range(steps):
        if catch:
            truth = distances > catchdist
        positions[t] = x[:]
        distances = np.linalg.norm(x, axis=0)
        percent_within_1[t] = np.sum(distances < catchdist) / size * 100
        x += np.random.randint(low=-1, high=2, size=(dim, size)) * truth
        
        
        

    # Color map for distinct point trajectories
    cmap_3d = plt.cm.get_cmap("gist_rainbow")
    colors_3d = cmap_3d(np.linspace(0, 1, size))

    # Plot each point's trajectory as a line in 2D or 3D space
    for i in range(size):
        if dim == 3:
            ax.plot(positions[:, 0, i], positions[:, 1, i], positions[:, 2, i],
                    color=colors_3d[i], linewidth=1, alpha=1)
        elif dim == 2:
            ax.plot(positions[:, 0, i], positions[:, 1, i],
                    color=colors_3d[i], linewidth=1, alpha=1)
        elif dim == 1:
            ax.plot(np.arange(positions[:, 0, i].shape[0]), positions[:, 0, i],
                    color=colors_3d[i], linewidth=1, alpha=1)

    # Mark the final positions with black dots
    if dim == 3:
        ax.scatter(positions[-1, 0, :], positions[-1, 1, :], positions[-1, 2, :], 
                   color="black", s=50, alpha=1)
        ax.set_zlabel("Z-axis")
    elif dim == 2:
        ax.scatter(positions[-1, 0, :], positions[-1, 1, :], 
                   color="black", s=50, alpha=1)
    elif dim == 1:
        ax.scatter(positions[-1, 0, :]*0 + positions[:, 0, i].shape[0], positions[-1, 0, :],
                   color="black", s=50, alpha=1)
        
    # Set labels for main plot
    ax.set_title("Trajectories of Points Over Time")
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis" if dim >= 2 else "Position")

    # Prepare to plot distance distribution over time on `ax_dist`
    cmap_dist = plt.cm.get_cmap("viridis")
    colors_dist = cmap_dist(np.linspace(0, 1, steps // interval + 1))

    # Calculate and plot distance distributions every `interval` steps
    for i, t in enumerate(range(0, steps, interval)):
        distances = np.linalg.norm(positions[t, :, :], axis=0)
        max_distance = np.max(distances)
        bins = np.arange(0, max_distance + bin_width, bin_width)
        hist, bin_edges = np.histogram(distances, bins=bins)

        # Plot line for this interval's distribution
        ax_dist.plot(bin_edges[:-1] + bin_width / 2, hist, color=colors_dist[i], label=f'Step {t}')

    # Set labels for distance distribution plot
    ax_dist.set_title("Distance Distribution Over Time")
    ax_dist.set_xlabel("Distance from Origin (binned)")
    ax_dist.set_ylabel("Number of Points")
    

    # Plot percentage of points within distance of 1 over time
    ax_percent.plot(np.arange(steps), percent_within_1, color="blue", linewidth=1)
    ax_percent.set_title("Percentage of Points Within Distance 1 Over Time")
    ax_percent.set_xlabel("Steps")
    ax_percent.set_ylabel("Percentage (%)")

    plt.draw()

# Initial parameters
size_init = 4
steps_init = 100
dim_init = 2
catch_init = 1  # Using 1 for "catch" on (True), 0 for "catch" off (False)

# Plot setup
fig = plt.figure(figsize=(18, 7))
ax = fig.add_subplot(131, projection='3d' if dim_init == 3 else None)
ax_dist = fig.add_subplot(132)
ax_percent = fig.add_subplot(133)

# Initial plot
plot_random_walk(ax, ax_dist, ax_percent, size_init, steps_init, dim=dim_init, catch=bool(catch_init))

# Add sliders for `size`, `steps`, `dim`, and `catch`
ax_size = plt.axes([0.15, 0.03, 0.3, 0.02], facecolor='lightgoldenrodyellow')
ax_steps = plt.axes([0.15, 0.01, 0.3, 0.02], facecolor='lightgoldenrodyellow')
ax_dim = plt.axes([0.55, 0.03, 0.3, 0.02], facecolor='lightgoldenrodyellow')
ax_catch = plt.axes([0.55, 0.01, 0.3, 0.02], facecolor='lightgoldenrodyellow')

size_slider = Slider(ax_size, 'Size', 1, 2000, valinit=size_init, valstep=1)
steps_slider = Slider(ax_steps, 'Steps', 100, 1000, valinit=steps_init, valstep=10)
dim_slider = Slider(ax_dim, 'Dim', 1, 3, valinit=dim_init, valstep=1)
catch_slider = Slider(ax_catch, 'Catch', 0, 1, valinit=catch_init, valstep=1)

# Update function for sliders
def update(val):
    global ax
    fig.delaxes(ax)  # Clear previous plot axis
    ax = fig.add_subplot(131, projection='3d' if int(dim_slider.val) == 3 else None)
    plot_random_walk(
        ax, ax_dist, ax_percent, 
        int(size_slider.val), int(steps_slider.val),
        dim=int(dim_slider.val), catch=bool(catch_slider.val)
    )

# Attach update function to sliders
size_slider.on_changed(update)
steps_slider.on_changed(update)
dim_slider.on_changed(update)
catch_slider.on_changed(update)

plt.show()
