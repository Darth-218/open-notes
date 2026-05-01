# Python Programming Guide

## Introduction

Python is a versatile programming language used for web development, data science, AI, and more.

## Basics

### Variables

```python
x = 10
name = "Alice"
is_active = True
```

### Functions

```python
def greet(name):
    return f"Hello, {name}!"

print(greet("World"))
```

## Advanced Topics

### Decorators

Decorators allow you to modify the behavior of functions.

```python
def timer(func):
    import time
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f"Took {time.time() - start:.2f}s")
        return result
    return wrapper

@timer
def slow_function():
    import time
    time.sleep(1)
```

### Async Programming

```python
import asyncio

async def fetch_data():
    await asyncio.sleep(1)
    return "data"

async def main():
    result = await fetch_data()
    print(result)

asyncio.run(main())
```