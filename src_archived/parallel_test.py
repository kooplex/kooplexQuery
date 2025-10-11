import time
from concurrent.futures import ThreadPoolExecutor


def method_1(x=2):
    time.sleep(1)
    print("2")
    return f"Result from method 1 with input {x}: {x * 2}"

def method_2():
    time.sleep(1)
    print("3")
    time.sleep(0)
    return "Result from method 2"

with ThreadPoolExecutor() as executor:
            future1 = executor.submit(method_1)
            future2 = executor.submit(method_2)
            print("sleeping")
            result1 = future1.result()
            result2 = future2.result()
