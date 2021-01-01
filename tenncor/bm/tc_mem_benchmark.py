from memory_profiler import memory_usage
import tenncor as tc
import time

shape = [1000, 1000, 1000]

#@profile
def main():
    a = tc.api.init.random_uniform(1.,2.)(shape, "W")
    b = tc.api.init.random_uniform(1.,2.)(shape, "W2")
    c = (a + b) * a

    stuff = c.get()

    print('hello')

if __name__ == '__main__':
    main()
    #mem_usage = memory_usage(main)
    #print('Memory usage (in chunks of .1 seconds): %s' % mem_usage)
    #print('Maximum memory usage: %s' % max(mem_usage))
