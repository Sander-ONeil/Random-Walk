import numpy as np


# x =np.random.randint(low = -1, high = 2,size = (2,100))

# print(np.linalg.norm(x,axis = 0)>= 1 )

def walk(dims):
    size = 30000

    x = np.zeros((dims,size),dtype = np.int32)
    truth = x == 0
    x += np.random.randint(low = -1, high = 2,size = (dims,size))*2


    for t in range (15000):
        x += truth*np.random.randint(low = -1, high = 2,size = (dims,size))

        truth = np.linalg.norm(x,axis = 0)>= 1 
        #print(x)

    print('Walk in ',dims,' dimensions ',np.sum(np.linalg.norm(x,axis = 0)<= 1) /size,' finding way back')

# walk(1)
# walk(2)
# walk(3)
# walk(4)

# size = 3000

# x = np.zeros((dims,size),dtype = np.int32)
# truth = x == 0
# x += np.random.randint(low = -1, high = 2,size = (dims,size))*2


# for t in range (100):
#     x += truth*np.random.randint(low = -1, high = 2,size = (dims,size))

#     truth = np.linalg.norm(x,axis = 0)>= 1 


# print('Walk in ',dims,' dimensions ',np.sum(np.linalg.norm(x,axis = 0)<= 1) /size,' finding way back')
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.cm import ScalarMappable

# # Parameters
# size = 3000
# steps = 500

# # Initialize points
# x = np.zeros((2, size), dtype=np.int32)
# truth = x == 0
# x += np.random.randint(low=-1, high=2, size=(2, size)) * 2

# # Color map
# colors = plt.cm.viridis(np.linspace(0, 1, steps))

# # Plot setup
# plt.figure(figsize=(10, 10))
# plt.title("Movement of Points Over Time")
# plt.xlabel("X-axis")
# plt.ylabel("Y-axis")

# # Run through each time step, plotting points with slightly different colors
# for t in range(steps):
#     # Apply updates to points
#     x += truth * np.random.randint(low=-1-2, high=2+2, size=(2, size))
#     truth = np.linalg.norm(x, axis=0) >= 1
    
#     # Plot the current points with the corresponding color
#     if t == 400:
#         plt.scatter(x[0], x[1], color=colors[t], s=1, alpha=0.7)

# # Color bar for reference
# sm = ScalarMappable(cmap="viridis")
# sm.set_array([])
# plt.colorbar(sm, label="Time Step")

# plt.show()

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Parameters
size = 100
steps = 100

# Initialize points
x = np.zeros((2, size), dtype=np.int32)
truth = x == 0
x += np.random.randint(low=-1, high=2, size=(2, size)) * 2

# Color map
colors = plt.cm.viridis(np.linspace(0, 1, steps))

# Plot setup
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.set_title("Movement of Points Over Time in 3D")
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Time Step")

# Run through each time step, plotting points with slightly different colors
for t in range(steps):
    # Apply updates to points
    x += truth * np.random.randint(low=-1, high=2, size=(2, size))
    truth = np.linalg.norm(x, axis=0) >= 1
    
    # Plot the current points with the respective color
    ax.scatter(x[0], x[1], t, color=colors[t], s=1, alpha=0.7)

plt.show()
