import argparse
import csv
import operator
import os
import re

import matplotlib.pyplot as plt


def compute_ema(data, ema):
    res = []
    for e in data:
        if not res:
            res.append(e)
        else:
            res.append(res[-1] * ema + e * (1.0 - ema))
    return res


def plot(data, ema, title):
    if ema:
        data = compute_ema(data, ema)
    plt.plot(list(range(len(data))), data)
    plt.title(title)
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory')
    parser.add_argument('--ema', type=float, default=0.999)
    args = parser.parse_args()

    d = []
    for filename in sorted(os.listdir(args.directory)):
        file_path = os.path.join(args.directory, filename)
        if os.path.isfile(file_path) and re.fullmatch(r'\d+\.monitor\.csv', filename):
            with open(file_path) as file:
                csv_reader = csv.reader(file)
                next(csv_reader)
                next(csv_reader)
                d += [[float(e) for e in row] for row in csv_reader]
    d.sort(key=operator.itemgetter(2))

    plot([e[0] for e in d], ema=args.ema, title='Reward')
    plot([e[1] for e in d], ema=args.ema, title='Episode length')


if __name__ == '__main__':
    main()
