import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', required=True)
    parser.add_argument('--e1', required=True)
    parser.add_argument('--e2', required=True)

    args = parser.parse_args()

    with open(args.file) as f:
        lines = [l.split('\t') for l in f]

    found_e1_ref = False
    found_e2_ref = False
    for line in lines:
        if line[0] == args.e1 and 'ProteinReference' in line[1]:
            found_e1_ref = True
            print("Found:", line)
        if line[0] == args.e2 and 'ProteinReference' in line[1]:
            found_e2_ref = True
            print("Found:", line)

    assert found_e1_ref and found_e2_ref

    for line in lines:
        if line[0] == args.e1 and line[2] == args.e2:
            print(line)
        if line[0] == args.e2 and line[2] == args.e1 and line[1] == 'in-complex-with':
            print(line)

