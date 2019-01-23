import sys

with open(sys.argv[1]) as f_in, open(sys.argv[2], 'w') as f_out:
    for line in f_in:
        line = line.strip()
        e1, r, e2 = line.split('\t')
        e1_complex = e1.split(';')
        e2_complex = e2.split(';')

        for c1 in e1_complex:
            for c2 in e1_complex:
                if e1 == e2:
                    continue
                else:
                    f_out.write(f"{c1}\tcomplex\t{c2}\t\n")
        for c1 in e2_complex:
            for c2 in e2_complex:
                if e1 == e2:
                    continue
                else:
                    f_out.write(f"{c1}\tcomplex\t{c2}\t\n")
        
        f_out.write(f"{e1_complex[0]}\t{r}\t{e2_complex[0]}\t\n")
        



