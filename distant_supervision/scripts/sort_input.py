import sys
from collections import defaultdict

def get_rel_type(line):
    return line.split('\t')[4].strip()

def get_sup_type(line):
    return line.split('\t')[6].strip()

def foo():
    return {'direct': [], 'distant': []}

bags = defaultdict(foo)
with open(sys.argv[1]) as f:
    for line in f:
        pair = tuple(line.split('\t')[:2])
        if get_sup_type(line) == 'distant':
            bags[pair]['distant'].append(line)
        else:
            bags[pair]['direct'].append(line)

with open(sys.argv[1] + '.srtd', 'w') as f:
    for lines in bags.values():
        f.writelines(sorted(lines['distant'], key=get_rel_type))
        f.writelines(sorted(lines['direct'], key=get_rel_type))

