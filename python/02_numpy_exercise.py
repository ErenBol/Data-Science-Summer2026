import numpy as np

# =========================
# NumPy Task 2 Foundations
# Broadcasting, boolean masking, statistics, reshaping
# =========================


# Exercise 1 — Basic broadcasting
# Create an array from 1 to 10.
# Add 10 to every element.
# Expected: [11 12 13 14 15 16 17 18 19 20]

arr = np.arange(1, 11)
arr_plus_10 = arr + 10

print("Exercise 1:")
print(arr_plus_10)


# Exercise 2 — Element-wise operations
# Using the same arr:
# a) multiply every value by 3
# b) square every value
# c) divide every value by 2

multiplied = arr * 3
squared = arr**2
divided = arr / 2

print("Exercise 2:")
print("multiplied:", multiplied)
print("squared:", squared)
print("divided:", divided)


# Exercise 3 — Boolean mask
# Create an array from 1 to 20.
# Select only values greater than 12.

arr20 = np.arange(1, 21)
greater_than_12 = arr20[arr20 > 12]

print("Exercise 3:")
print(greater_than_12)


# Exercise 4 — Multiple conditions
# From arr20, select numbers that are:
# greater than 5 AND divisible by 3.
# Expected: [ 6  9 12 15 18]

selected = arr20[(arr20 > 5) & (arr20 % 3 == 0)]

print("Exercise 4:")
print(selected)


# Exercise 5 — Replace with boolean mask
# Copy arr20.
# Replace all values smaller than 10 with 0.
# Original arr20 should not change.

modified = arr20.copy()
modified[modified < 10] = 0
print("Exercise 5:")
print("original:", arr20)
print("modified:", modified)


# Exercise 6 — Reshape and statistics
# Create a 3x4 matrix from numbers 1 to 12.
# Calculate:
# a) total sum
# b) mean
# c) standard deviation
# d) max value
# e) min value

matrix = np.arange(1, 13).reshape((3, 4))

total_sum = np.sum(matrix)
mean_value = np.mean(matrix)
std_value = np.std(matrix)
max_value = np.max(matrix)
min_value = np.min(matrix)

print("Exercise 6:")
print(matrix)
print("sum:", total_sum)
print("mean:", mean_value)
print("std:", std_value)
print("max:", max_value)
print("min:", min_value)


# Exercise 7 — Sum along axes
# Using the 3x4 matrix:
# a) calculate column sums
# b) calculate row sums

column_sums = np.sum(matrix, axis=0)
row_sums = np.sum(matrix, axis=1)

print("Exercise 7:")
print("column sums:", column_sums)
print("row sums:", row_sums)


# Exercise 8 — Broadcasting with rows
# Create this matrix:
# [[10, 20, 30],
#  [40, 50, 60],
#  [70, 80, 90]]
#
# Create this row array:
# [1, 2, 3]
#
# Add the row array to every row of the matrix.

m = np.arange(10, 100, 10).reshape((3, 3))
row = np.array([1, 2, 3])

result = m + row

print("Exercise 8:")
print(result)


# Exercise 9 — Broadcasting with columns
# Using the same matrix m:
# Create this column array:
# [[1],
#  [2],
#  [3]]
#
# Add it to the matrix.
# Each row should receive a different added value.

col = np.arange(1, 4).reshape((3, 1))
result_col = m + col

print("Exercise 9:")
print(result_col)


# Exercise 10 — Normalize data
# Given exam scores:
# [50, 60, 70, 80, 90]
#
# Normalize them using:
# normalized = (scores - mean) / std

scores = np.arange(50, 100, 10)

scores_mean = np.mean(scores)
scores_std = np.std(scores)
normalized_scores = (scores - scores_mean) / scores_std

print("Exercise 10:")
print("mean:", scores_mean)
print("std:", scores_std)
print("normalized:", normalized_scores)


# Exercise 11 — Filter rows by condition
# Each row represents a student.
# Columns are: [math, physics, chemistry]
#
# Select students whose math score is greater than or equal to 70.

grades = np.array(
    [[85, 70, 90], [60, 75, 80], [95, 90, 85], [50, 65, 70], [72, 88, 91]]
)

high_math_students = grades[[i for i in range(len(grades)) if grades[i, 0] >= 70]]
high_math_students_numpy = grades[grades[:, 0] >= 70]

print("Exercise 11:")
print(high_math_students)
print(high_math_students_numpy)

# Exercise 12 — Average score per student
# Using grades, calculate the average score of each student.
# Then select students whose average is greater than 80.

student_averages = np.mean(grades, axis=1)
strong_students = grades[student_averages > 80]

print("Exercise 12:")
print("student averages:", student_averages)
print("strong students:\n", strong_students)


# Exercise 13 — Mini challenge: product prices
# Product prices:
# [100, 250, 80, 400, 150, 60]
#
# Tasks:
# a) apply 20% discount to all prices
# b) select discounted prices below 150
# c) calculate average discounted price

prices = np.array([100, 250, 80, 400, 150, 60])

discounted_prices = prices * 0.8
cheap_discounted = discounted_prices[discounted_prices < 150]
avg_discounted = np.mean(discounted_prices)

print("Exercise 13:")
print("discounted prices:", discounted_prices)
print("cheap discounted prices:", cheap_discounted)
print("average discounted price:", avg_discounted)


# Exercise 14 — Mini challenge: monthly sales
# Each row is a store.
# Each column is a month.
#
# Calculate:
# a) total sales per store
# b) total sales per month
# c) best store index
# d) best month index

sales = np.array(
    [[1200, 1500, 1700], [900, 1100, 1000], [2000, 1800, 2200], [700, 950, 1200]]
)

store_totals = np.sum(sales, axis=1)
month_totals = np.sum(sales, axis=0)
best_store = np.argmax(store_totals)
best_month = np.argmax(month_totals)

print("Exercise 14:")
print("store totals:", store_totals)
print("month totals:", month_totals)
print("best store index:", best_store)
print("best month index:", best_month)


# Exercise 15 — np.where
# Create an array from 1 to 10.
# Create a new array where:
# numbers >= 5 become "high"
# numbers < 5 become "low"

nums = np.arange(1, 11)
labels = np.where(nums >= 5, "high", "low")

print("Exercise 15:")
print(labels)


# =========================
# Short Notes
# =========================

# Broadcasting:
# NumPy can apply operations between arrays of different shapes
# if their shapes are compatible.

# Boolean masking:
# A condition like arr > 5 creates True/False values.
# Then arr[arr > 5] selects only the True positions.

# Axis:
# axis=0 gives column-wise results.
# axis=1 gives row-wise results.
