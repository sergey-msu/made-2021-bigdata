#!/usr/bin/env python
""" Mean Mapper. """

import sys


def read_input(file):
    price_idx = None
    for line in file:
        segms = line.split(',')
        if price_idx is None:
            price_idx = segms.index('price') - len(segms)
            continue
        if -price_idx >= len(segms):
            continue
        price = segms[price_idx]
        if price == '':
            continue

        yield int(price)


def do_map(inpt):
    prices = read_input(inpt)
    return prices


def main():
    prices = do_map(sys.stdin)
    for price in prices:
        print(f'{price} 1')


if __name__ == "__main__":
    main()

