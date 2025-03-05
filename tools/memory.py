

print("HELLO WE HAVE ")



import psutil

# Get the total system memory
total_memory = psutil.virtual_memory().total

# Get the available memory for the current process
available_memory = psutil.virtual_memory().available

# Get the memory usage of the current process
process = psutil.Process()
process_memory = process.memory_info().rss  # Resident Set Size (actual memory usage)

print(f"Total system memory: {total_memory / (1024**3):.2f} GB")
print(f"Available memory: {available_memory / (1024**3):.2f} GB")
print(f"Current process memory usage: {process_memory / (1024**3):.2f} GB")