import sys
import os
import argparse
import pandas as pd
from cryosparc_compute import dataset


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__
    )
    parser.add_argument('--infile', type=str, required=True, help='Input cs file.')
    parser.add_argument('--outfile', type=str, help='Output csv file. Default is the input file with its extension changed to .csv')
    parser.add_argument('--overwrite', action='store_true', help='Allow overwriting output file.')

    args = parser.parse_args()

    print('##### Command #####\n\t' + ' '.join(sys.argv))
    args_print_str = '##### Input parameters #####\n'
    for opt, val in vars(args).items():
        args_print_str += '\t{} : {}\n'.format(opt, val)
    print(args_print_str)
    return args


def main(infile: str, outfile: str, overwrite: bool) -> None:
    assert os.path.exists(infile), f'Input file {infile} not exist'
    if not overwrite:
        assert not os.path.exists(outfile), f'Output file {outfile} already exists. Use --overwrite to overwrite the file.'

    indata = dataset.Dataset().load(infile)
    df = dataset.to_dataframe(indata)
    df.to_csv(outfile, index=False)


if __name__ == '__main__':
    args = parse_args()
    main(
        args.infile,
        args.outfile,
        args.overwrite
    )
