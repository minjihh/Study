def multiply(x):
    return x * 10

numbers = [1,2,3,4]

result = map(multiply, numbers)   # map(함수, 리스트)

print(result)
# <map object at 0x000001E62A53ACE0>

print(list(result))
# [10, 20, 30, 40]



