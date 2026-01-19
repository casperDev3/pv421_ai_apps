import numpy as np
import time

# 1000 елементів для більш помітного ефекту
a = [
    i for i in range(10000000)
]
b = np.array(
    [i for i in range(10000000)]
)

# порівняти час виконання операції множення кожного елемента списку a на 2
start_time_list = time.time()
result_list = [x * 2 for x in a]
end_time_list = time.time()
# print("Result using list comprehension:", result_list)
print("Time taken using list comprehension: {:.10f} seconds".format(end_time_list - start_time_list ))

# порівняти час виконання операції множення кожного елемента масиву b на 2
start_time = time.time()
result_array = b * 2
end_time = time.time()
# print("Result using NumPy array:", result_array)
print("Time taken using NumPy array: {:.10f} seconds".format(end_time - start_time ))

# Вивести співвідношення часу виконання
list_time = end_time_list - start_time_list
array_time = end_time - start_time

if array_time > 0:
    ratio = list_time / array_time
    print("List comprehension is {:.2f} times slower than NumPy array operation.".format(ratio))
else:
    print("NumPy array operation time is too small to calculate ratio.")
