from memory_profiler import profile
import tenncor as tc
import time

shape = [100, 1000, 1000]

@profile
def main():
    a = tc.api.init.random_uniform(1.,2.)(shape, "W")
    b = tc.api.init.random_uniform(1.,2.)(shape, "W2")
    c = tc.api.matmul(tc.api.matmul(a,b),a)

    stuff = c.get()

    print('hello')

if __name__ == '__main__':
    main()
