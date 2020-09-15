import sys
import re

replfile = sys.argv[1]

raw_paths = sys.stdin.read()
ymlfiles = raw_paths.split('\n')
ymlfiles = [path.strip() for path in ymlfiles if len(path.strip()) > 0]

mapping = []
with open(replfile) as f:
    for line in f.readlines():
        try:
            line = line.strip()
            src, dest = tuple(line.split(','))
            mapping.append((src, dest))
        except:
            pass

for fname in ymlfiles:
    subbed = False
    with open(fname, 'r') as f:
        content = f.read()
        for src, dest in mapping:
            detections = re.findall(src, content)
            if len(detections) > 0:
                subbed = True
            content = re.sub(src, dest, content)
    if subbed:
        with open(fname, 'w') as f:
            f.write(content)
