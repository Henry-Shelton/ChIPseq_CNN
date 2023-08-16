print("import vis modules")
import numpy as np
import matplotlib.pyplot as plt

# Assuming 'x', 'ref', and 'y' are your variables
x = np.random.random((1, 12000, 1))
ref = np.random.random((1, 12000, 1))
y = np.random.random((1, 1, 12000))

# Create a figure with subplots
fig, axs = plt.subplots(3, 1, figsize=(10, 15))

# Plot x
axs[0].plot(x[0, :, 0])
axs[0].set_title('x: (1, 12000, 1)')

# Plot ref
axs[1].plot(ref[0, :, 0])
axs[1].set_title('ref: (1, 12000, 1)')

# Plot y
axs[2].plot(y[0, 0, :])
axs[2].set_title('y: (1, 1, 12000)')

# Adjust spacing between subplots
plt.tight_layout()

# show the graph
plt.savefig("mygraph.png")