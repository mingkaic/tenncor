import sys
import os.path
import re

def strip_prefix(path, prefix):
    if path.startswith(prefix):
        path = path[len(prefix):]
    path = path.strip('/')
    return path

def find_includes(txt):
    return re.findall('#include "(.*)"\n', txt)

def replace_includes(txt, subs):
    for orig, repl in subs:
        txt = re.sub(r'#include "' + orig + '"', r'#include "' + repl + '"', txt)
    return txt

remove_prefix = sys.argv[1]

raw_paths = sys.stdin.read()
paths = raw_paths.split('\n')
paths = [path.strip() for path in paths if len(path.strip()) > 0]

file2path = {}
for path in paths:
    filename = os.path.basename(path)
    if filename in file2path:
        fileparts = filename.split('.')
        nxfile = filename
        i = 0
        prefix = fileparts[0]
        affix = '.' + '.'.join(fileparts[1:])
        while nxfile in file2path:
            nxfile = prefix + str(i) + affix
            i += 1
        file2path[nxfile] = path
    else:
        file2path[filename] = path

conversions = []
for nxfile in file2path:
    path = file2path[nxfile]
    realfile = os.path.basename(path)
    realdir = os.path.dirname(path)
    nxpath = os.path.join(realdir, nxfile)
    if path != nxpath:
        conversions.append((path, nxpath))

include_conversions = dict([(strip_prefix(realpath, remove_prefix), strip_prefix(nxpath, remove_prefix))
    for realpath, nxpath in conversions])

# look for includes in each path file, then rename matching includes to mapped include_conversion values
for path in paths:
    nxtxt = None
    with open(path, 'r', encoding='utf-8') as f:
        txt = f.read()
        includes = find_includes(txt)
        subs = []
        for include in includes:
            if include in include_conversions:
                subs.append((include, include_conversions[include]))
        if len(subs) > 0:
            nxtxt = replace_includes(txt, subs)
    if nxtxt:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(nxtxt)

for src, dest in conversions:
    os.rename(src, dest)

for src in include_conversions:
    dest = include_conversions[src]
    print('{},{}'.format(src, dest))
