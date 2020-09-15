import sys
import re

srcfile = sys.argv[1]
replfile = sys.argv[2]

mapping = []
with open(replfile) as f:
    for line in f.readlines():
        try:
            line = line.strip()
            dest, src = tuple(line.split(','))
            mapping.append((src, dest))
        except:
            pass

with open(srcfile) as f:
    content = f.read()
    for src, dest in mapping:
        content = re.sub(src, dest, content)
    print(content)
