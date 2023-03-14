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
    parser.add_argument('--infile-passthrough', type=str, help='Input passthrough cs file.')
    parser.add_argument('--outfile-rootname', type=str, required=True, help='Root name for output files.')
    parser.add_argument('--target', type=str, required=True, help='Target feature for filtering.')
    parser.add_argument('--sigma', type=float, help='Only mean ± sigma * stdev particles will be retained.')
    parser.add_argument('--minval', type=float, help='Min value')
    parser.add_argument('--maxval', type=float, help='Max value')
    parser.add_argument('--overwrite', action='store_true', help='Allow overwriting output files.')

    args = parser.parse_args()

    print('##### Command #####\n\t' + ' '.join(sys.argv))
    args_print_str = '##### Input parameters #####\n'
    for opt, val in vars(args).items():
        args_print_str += '\t{} : {}\n'.format(opt, val)
    print(args_print_str)
    return args


def main(infile: str, infile_passthrough: str, outfile_rootname: str, target: str, sigma: float, minval: float, maxval: float, overwrite: bool) -> None:
    assert os.path.exists(infile), f'Input file {infile} does not exist.'

    indata = dataset.Dataset().load(infile)
    inarr = indata.to_records()
    assert target in inarr.dtype.names, f'Target {target} does not exist in {infile}.'

    if infile_passthrough is not None:
        assert os.path.exists(infile_passthrough), f'Input passthrough file {infile_passthrough} does not exist.'
        indatapassthrough = dataset.Dataset().load(infile_passthrough)
        indata = indata.innerjoin(indatapassthrough)

    dat = inarr[target]
    dat_stat = scipy.stats.describe(dat)

    if sigma is not None and (minval is not None or maxval is not None):
        sys.exit('--sigma and (--minval or --maxval) cannot be specified at once.')

    if sigma is not None:
        minval = dat_stat.mean - sigma * np.sqrt(dat_stat.variance)
        maxval = dat_stat.mean + sigma * np.sqrt(dat_stat.variance)
        print(f'Automatically sets (min, max) = ({minval}, {maxval})')

    if minval is None:
        minval = sys.float_info.min
    if maxval is None:
        maxval = sys.float_info.max

    outhist = f'{outfile_rootname}_hist.png'
    if not overwrite and os.path.exists(outhist):
        sys.exit(f'{outhist} already exists. --overwrite for overwriting output files.')
    mask = (minval <= dat) & (dat <= maxval)
    dat_out = dat[mask]
    dat_out_stat = scipy.stats.describe(dat_out)
    fig, ax = plt.subplots(layout='constrained')
    ax.hist(dat_out, bins='auto')
    ax.set_xlabel(target)
    ax.set_ylabel('Frequency')
    ax.text(
        0.99, 0.99,
        f'Mean {dat_out_stat.mean:.6f}\nMin {dat_out_stat.minmax[0]:.6f}\nMax {dat_out_stat.minmax[1]:.6f}\nStdev {np.sqrt(dat_out_stat.variance):.6f}\n#Ptcls {dat_out_stat.nobs}',
        va='top', ha='right', transform=ax.transAxes
    )
    plt.savefig(outhist)
    print(f'Output histogram is saved as {outhist}')

    outfile = f'{outfile_rootname}.cs'
    outdataset = dataset.Dataset(inarr[mask])
    outdataset.save(outfile)
    print(f'Output dataset is saved as {outfile}')

    print(f'Num particles {dat_stat.nobs} -> {dat_out_stat.nobs} ({dat_stat.nobs - dat_out_stat.nobs} particles = {(dat_stat.nobs - dat_out_stat.nobs) / dat_stat.nobs * 100:.1f} % were discarded.)')


if __name__ == '__main__':
    args = parse_args()
    main(
        args.infile,
        args.infile_passthrough,
        args.outfile_rootname,
        args.target,
        args.sigma,
        args.minval,
        args.maxval,
        args.overwrite
    )
