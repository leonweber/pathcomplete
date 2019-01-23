import sys
from pathlib import Path
import json

import tqdm
import pandas as pd
from lxml import etree


root = Path(__file__).parent

with open(sys.argv[1]) as f_in, open(sys.argv[2], 'w') as f_out:
    for line in f_in:
        if line.startswith('#'):
            continue
        entry = line.strip().split('\t')
        if '-' in [entry[2], entry[5]]:
            continue
        e1 = entry[2].split(':')[1]
        e2 = entry[5].split(':')[1]
        r = entry[6]

        if len(entry) > 8:
            refs = entry[8]
        else:
            refs = ""

        f_out.write(f"{e1}\t{r}\t{e2}\t{refs}\n")




