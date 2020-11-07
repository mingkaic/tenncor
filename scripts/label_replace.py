import sys
import re
import os.path

srcfile = sys.argv[1]
replfile = sys.argv[2]

if not os.path.isfile(replfile):
    exit(0)

mapping = []
with open(replfile, encoding='utf-8') as f:
    for line in f.readlines():
        try:
            line = line.strip()
            dest, src = tuple(line.split(','))
            mapping.append((src, dest))
        except:
            pass

with open(srcfile, encoding='utf-8') as f:
    content = f.read()
    for src, dest in mapping:
        content = re.sub(src, dest, content)
    print(content)
