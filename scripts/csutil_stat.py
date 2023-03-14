import sys
import os
import argparse
from typing import List
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from cryosparc_compute import dataset


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__
    )
    parser.add_argument('--infile', type=str, required=True, help='Input cs file.')
    parser.add_argument('--outfile-rootname', type=str, help='Root name for output files.')
    parser.add_argument('--targets', nargs='+', type=str, help='Target features to get statistics. Multiple targets can be specified via whitespace separated list.')
    parser.add_argument('--overwrite', action='store_true', help='Allow overwriting output files.')
    parser.add_argument('--num-bins', type=str, default='auto', help='Number of bins for histogram plot. Default "auto"')

    args = parser.parse_args()

    print('##### Command #####\n\t' + ' '.join(sys.argv))
    args_print_str = '##### Input parameters #####\n'
    for opt, val in vars(args).items():
        args_print_str += '\t{} : {}\n'.format(opt, val)
    print(args_print_str)
    return args


def main(infile: str, outfile_rootname: str, overwrite: bool, num_bins: str, targets: List[str]) -> None:
    assert os.path.exists(infile), f'Input file {infile} not exist'

    if num_bins == 'auto':
        bins = num_bins
    else:
        try:
            bins = int(num_bins)
        except ValueError as err:
            print(f'Invalid value for num_bins: {num_bins}  : {err}')

    indata = dataset.Dataset().load(infile)
    inarr = indata.to_records()
    for target in targets:
        if target not in inarr.dtype.names:
            sys.exit(f'No such target: {target}. Available targets are: {inarr.dtype.names}')

        outfile = f'{outfile_rootname}_{target.replace("/", "_")}.png'
        if not overwrite and os.path.exists(outfile):
            sys.exit(f'Abort processing because the output file {outfile} already exists. Specify --overwrite to overwrite.')

        dat = inarr[target]
        stat = scipy.stats.describe(dat)
        fig, ax = plt.subplots(layout='constrained')
        ax.hist(dat, bins=bins)
        ax.set_xlabel(target)
        ax.set_ylabel('Frequency')
        ax.text(
            0.99, 0.99,
            f'Mean {stat.mean:.6f}\nMin {stat.minmax[0]:.6f}\nMax {stat.minmax[1]:.6f}\nStdev {np.sqrt(stat.variance):.6f}\n#Ptcls {stat.nobs}',
            va='top', ha='right', transform=ax.transAxes
        )
        plt.savefig(outfile)


if __name__ == '__main__':
    args = parse_args()
    main(
        args.infile,
        args.outfile_rootname,
        args.overwrite,
        args.num_bins,
        args.targets
    )
