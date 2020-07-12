
import os
import sys

def strip_affix(path, affixes):
    realpath = os.path.abspath(path)
    for affix in affixes:
        if realpath.endswith(affix):
            path = realpath[:len(realpath) - len(affix)]
    return path

if __name__ == '__main__':
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = ''

    if len(sys.argv) > 2:
        affixes = sys.argv[2:]
        path = strip_affix(path, affixes)

    print(path)
