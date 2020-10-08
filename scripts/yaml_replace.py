import sys
import re
import os.path

replfile = sys.argv[1]

if not os.path.isfile(replfile):
    exit(0)

raw_paths = sys.stdin.read()
ymlfiles = raw_paths.split('\n')
ymlfiles = [path.strip() for path in ymlfiles if len(path.strip()) > 0]

mapping = []
with open(replfile, encoding='utf-8') as f:
    for line in f.readlines():
        try:
            line = line.strip()
            src, dest = tuple(line.split(','))
            mapping.append((src, dest))
        except:
            pass

for fname in ymlfiles:
    subbed = False
    with open(fname, 'r', encoding='utf-8') as f:
        content = f.read()
        for src, dest in mapping:
            detections = re.findall(src, content)
            if len(detections) > 0:
                subbed = True
            content = re.sub(src, dest, content)
    if subbed:
        with open(fname, 'w', encoding='utf-8') as f:
            f.write(content)
