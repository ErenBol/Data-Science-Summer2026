import numpy as np

# =========================
# NumPy Task 1 Foundations
# Array creation, shape, dtype, indexing, slicing
# =========================


# Exercise 1 — Create a structured array
# Create a 1D array containing numbers from 10 to 50, increasing by 5.
# Expected: [10 15 20 25 30 35 40 45 50]

arr1 = np.arange(10, 55, 5)
print("Exercise 1:", arr1)


# Exercise 2 — Reshape carefully
# Convert arr1 into a 3x3 matrix.
# Print its shape, ndim, size, and dtype.

matrix = np.reshape(arr1, (3, 3))
print("Exercise 2 matrix:\n", matrix)
print("shape:", matrix.shape)
print("ndim:", matrix.ndim)
print("size:", matrix.size)
print("dtype:", matrix.dtype)


# Exercise 3 — Indexing
# From the matrix, extract:
# a) first row
# b) last column
# c) center element
# d) bottom-right element

first_row = matrix[0, :]
last_column = matrix[:, -1]
center = matrix[1, 1]
bottom_right = matrix[-1, -1]
print("Exercise 3:")
print("first row:", first_row)
print("last column:", last_column)
print("center:", center)
print("bottom-right:", bottom_right)


# Exercise 4 — Slicing submatrices
# From the matrix, extract the 2x2 block:
# [[10, 15],
#  [25, 30]]

block_2x2 = matrix[0:2, 0:2]
print("Exercise 4:\n", block_2x2)


# Exercise 5 — Reverse rows and columns
# Create:
# a) matrix with rows reversed
# b) matrix with columns reversed
# c) matrix fully reversed

rows_order_reversed = np.flip(matrix, axis=0)  # equivalently matrix[::-1,:]
cols_order_reversed = np.flip(matrix, axis=1)  # equivalently matrix[:,::-1]
fully_reversed = np.flip(matrix)  # equivalently matrix[::-1,::-1]


print("Exercise 5:")
print("rows reversed:\n", rows_order_reversed)
print("columns reversed:\n", cols_order_reversed)
print("fully reversed:\n", fully_reversed)


# Exercise 6 — Modify values using indexing
# Copy the matrix.
# Change the center value to 999.
# Change the entire first row to 0.
# Important: do not modify the original matrix.

modified = np.copy(matrix)
modified[1, 1] = 999
modified[0, :] = 0
print("Exercise 6 original:\n", matrix)
print("Exercise 6 modified:\n", modified)


# Exercise 7 — Create a custom 4x4 matrix
# Create this matrix using np.arange and reshape:
# [[ 1,  2,  3,  4],
#  [ 5,  6,  7,  8],
#  [ 9, 10, 11, 12],
#  [13, 14, 15, 16]]

m4 = np.arange(1, 17).reshape((4, 4))
print("Exercise 7:\n", m4)


# Exercise 8 — Checkerboard slicing
# From m4, extract:
# [[ 1,  3],
#  [ 9, 11]]
# Hint: use row step and column step.

checker = m4[0::2, 0::2]
print("Exercise 8:\n", checker)


# Exercise 9 — Border extraction
# From m4, extract:
# a) top row
# b) bottom row
# c) left column
# d) right column
# Then print them clearly.

top = m4[0, :]
bottom = m4[-1, :]
left = m4[:, 0]
right = m4[:, -1]

print("Exercise 9:")
print("top:", top)
print("bottom:", bottom)
print("left:", left)
print("right:", right)


# Exercise 10 — Inner matrix
# From m4, extract the inner 2x2 matrix:
# [[ 6,  7],
#  [10, 11]]

inner = m4[1:3, 1:3]
print("Exercise 10:\n", inner)


# Exercise 11 — Row vector vs column vector
# Create this 1D array:
# [2, 4, 6, 8]
# Then create:
# a) row vector with shape (1, 4)
# b) column vector with shape (4, 1)
# Print all shapes.

v = np.arange(2, 10, 2)
row_v = v.reshape((1, 4))
col_v = v.reshape((4, 1))

print("Exercise 11:")
print("v:", v, "shape:", v.shape)
print("row_v:\n", row_v, "shape:", row_v.shape)
print("col_v:\n", col_v, "shape:", col_v.shape)


# Exercise 12 — Mini challenge: seating chart
# Imagine this 3x5 array is a classroom seating chart.
# Each number is a student ID.
#
# [[101, 102, 103, 104, 105],
#  [106, 107, 108, 109, 110],
#  [111, 112, 113, 114, 115]]
#
# Extract:
# a) front row
# b) back row
# c) middle column
# d) students sitting in the 2x3 center block:
#    [[107, 108, 109],
#     [112, 113, 114]]

seats = np.arange(101, 116).reshape((3, 5))
front_row = seats[0]
back_row = seats[-1]
middle_column = seats[:, 2]
center_block = seats[1:, 1:4]

print("Exercise 12:")
print("seats:\n", seats)
print("front row:", front_row)
print("back row:", back_row)
print("middle column:", middle_column)
print("center block:\n", center_block)
