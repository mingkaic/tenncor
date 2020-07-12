
import os
import sys

def strip_prefixes(path, prefixes):
    realpath = os.path.abspath(path)
    for prefix in prefixes:
        prefix = os.path.abspath(prefix) + '/'
        if realpath.startswith(prefix):
            path = realpath[len(prefix):]
    return path

if __name__ == '__main__':
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = ''

    if len(sys.argv) > 2:
        prefixes = sys.argv[2:]
        path = strip_prefixes(path, prefixes)

    print(path)
