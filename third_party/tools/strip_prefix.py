import os
import sys

if len(sys.argv) > 1:
    path = sys.argv[1]
else:
    path = ''

def strip_prefixes(path, prefixes):
    realpath = os.path.abspath(path)
    for prefix in prefixes:
        prefix = os.path.abspath(prefix) + '/'
        if realpath.startswith(prefix):
            path = realpath[len(prefix):]
    return path

if len(sys.argv) > 2:
    prefixes = sys.argv[2:]
    path = strip_prefixes(path, prefixes)

print(path)
