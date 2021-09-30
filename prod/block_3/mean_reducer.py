#!/usr/bin/env python
""" Mean Reducer. """

import sys


def read_mapper_output(file):
    for line in file:
        yield line.split()


def do_reduce(inpt):
    data = read_mapper_output(inpt)
    new_mean = 0
    new_count = 0

    for mean, cnt in data:
        mean = float(mean)
        cnt = int(cnt)
        new_mean = (mean*cnt + new_mean*new_count) / (new_count + cnt)
        new_count += cnt

    return new_mean, new_count


def main():
    mean, cnt = do_reduce(sys.stdin)
    print(f'{mean} {cnt}')
    

if __name__ == "__main__":
    main()

