"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.


# Mathematical functions:
# - mul
def mul(x: float, y: float) -> float:
    return x * y


# - id
def id(x: float) -> float:
    return x


# - add
def add(x: float, y: float):
    return x + y


# - neg
def neg(x: float) -> float:
    return mul(-1.0, x)


# - lt
def lt(x: float, y: float) -> bool:
    return x < y


# - eq
def eq(x: float, y: float):
    return x == y


# - max
def max(x: float, y: float) -> float:
    if lt(x, y):
        return y
    return x


# - abs
def abs(x: float, y: float) -> float:
    return max(add(x, neg(y)), add(neg(x), y))


# - is_close
def is_close(x: float, y: float, eps: float = 10e-2) -> bool:
    return abs(x, y) < eps


# - exp
def exp(x: float) -> float:
    return math.exp(x)


# - sigmoid
def sigmoid(x: float) -> float:
    return inv(1.0 + exp(neg(x))) if x >= 0.0 else mul(exp(x), inv(1.0 + exp(neg(x))))


# - relu
def relu(x: float) -> float:
    return max(0.0, x)


# - log
def log(x: float) -> float:
    return math.log(x)


# - log_back
def log_back(x: float, other: float):
    return mul(inv(x), other)  # (1.0 / x) * other


# - inv
def inv(x: float) -> float:
    return 1.0 / x


# - inv_back
def inv_back(x: float, other: float) -> float:
    return mul(neg(1.0), mul(inv(x), inv(x)))  # (-1.0 * 1 / x^2) * other


# - relu_back
def relu_back(x: float, other: float) -> float:
    d = 0.0
    if lt(0.0, x):
        d = 1.0
    return mul(d, other)


#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# TODO: Implement for Task 0.1.


# ## Task 0.3

# Small practice library of elementary higher-order functions.


# Implement the following core functions
# - map
def map(l: Iterable, f: Callable) -> Iterable:
    return [f(x) for x in l]


# - zipWith
def zipWith(l1: Iterable, l2: Iterable, f: Callable) -> Iterable:
    return [f(x[0], x[1]) for x in zip(l1, l2)]


# - reduce
def reduce(l: Iterable, f: Callable):
    if not l:
        return
    a = list(l)[0]
    for x in list(l)[1:]:
        a = f(a, x)
    return a


#
# Use these to implement
# - negList : negate a list
def negList(l: Iterable) -> Iterable:
    return map(l, neg)


# - addLists : add two lists together
def addLists(l1: Iterable, l2: Iterable) -> Iterable:
    return zipWith(l1, l2, add)


# - sum: sum lists
def sum(l: Iterable):
    if not l:
        return 0.0
    return reduce(l, add)


# - prod: take the product of lists
def prod(l: Iterable):
    if not l:
        return 1.0
    return reduce(l, mul)


# TODO: Implement for Task 0.3.
