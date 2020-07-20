import matplotlib.pyplot as plt
import numpy as np

deg = np.pi/180

# Define the basis vectors
basis_x = [1, 0]
basis_y = [np.cos(60*deg), np.sin(60*deg)]

# Put the basis vectors into a matrix for multiplication later
basis = np.vstack((basis_x, basis_y)).T

# Generate some xy data and put it into a matrix too
x_coords = [j for i in range(10) for j in range(10)]
y_coords = [i for i in range(10) for j in range(10)]
vec_space = np.vstack((x_coords, y_coords))

# Do the basis transformation, using @ to do the matrix multiplication
# "np.matmul" would also work if you prefer
new_vec_space = basis@vec_space

# Output the "before" data on ax1 and "after" data on ax2
fig, (ax1, ax2) = plt.subplots(ncols=2)

for x, y in vec_space.T:
    ax1.plot(x, y, 'o')

for x, y in new_vec_space.T:
    ax2.plot(x, y, 'o')

# Equally scaled axes makes the dots look nicer
ax1.set_aspect('equal')
ax2.set_aspect('equal')
plt.show()
