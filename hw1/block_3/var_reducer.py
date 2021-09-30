#!/usr/bin/env python
""" Var Reducer. """

import sys


def read_mapper_output(file):
    for line in file:
        yield line.split()


def do_reduce(inpt):
    data = read_mapper_output(inpt)
    new_mean = 0
    new_var = 0
    new_count = 0

    for mean, var, cnt in data:
        mean = float(mean)
        var = float(var)
        cnt = int(cnt)
        new_var = (new_count*new_var + cnt*var + new_count*new_mean**2 + cnt*mean**2)/(new_count + cnt) - \
                  ((new_mean*new_count + mean*cnt)/(new_count + cnt))**2
        new_mean = (mean*cnt + new_mean*new_count) / (new_count + cnt)
        new_count += cnt

    return new_mean, new_var, new_count


def main():
    mean, var, cnt = do_reduce(sys.stdin)
    print(f'{mean} {var} {cnt}')


if __name__ == "__main__":
    main()

