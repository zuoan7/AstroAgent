import time

def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args,**kwargs)
        end = time.time()
        print(f"耗时:{end - start}秒")
        return result
    return wrapper

n = input().split()
print(list(map(int, n)))