import sys
from pathlib import Path
import json

import tqdm
import pandas as pd
from lxml import etree
from pypathway.utils import IdMapping


root = Path(__file__).parent

with open(sys.argv[1]) as f_in, open(sys.argv[2], 'w') as f_out:
    for line in f_in:
        e1, r, e2 = line.split('\t')
        e1 = e1.strip()
        e2 = e2.strip()

        e1 = IdMapping.convert([e1], species='hsa', source='SYMBOL', target='ENTREZID')[0][1]
        e2 = IdMapping.convert([e2], species='hsa', source='SYMBOL', target='ENTREZID')[0][1]
        if (e1 and e2) is None:
            continue
        f_out.write(f"{e1[0]}\t{r}\t{e2[0]}\n")




