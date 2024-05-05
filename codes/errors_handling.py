import threading

def my_function(a, b, c):
    # Simulate a long-running task
    import time
    time.sleep(5)
    return a + b + c

def run_with_timeout(func, args=(), timeout=20):
    result = [None]  # Used to store the result from the function

    def target():
        result[0] = func(*args)

    thread = threading.Thread(target=target)
    thread.start()
    thread.join(timeout)  # Wait for the thread to finish or timeout
    if thread.is_alive():
        return None  # Function timed out. Improve: timeout nhung van tra ket qua
    else:
        return result[0]  # Return value of the function

if __name__ == "__main__":
    # Example usage
    result = run_with_timeout(my_function, args=(1, 2, 3), timeout=10)
    if result is None:
        print("Function timed out")
    else:
        print("Function returned:", result)
