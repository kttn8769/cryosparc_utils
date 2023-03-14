import sys
import os
import argparse
import re
from cryosparc_compute import dataset


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__
    )
    parser.add_argument('--infile', type=str, required=True, help='Input csg file.')
    parser.add_argument('--outfile', type=str, required=True, help='Output csg file.')
    parser.add_argument('--metafile', type=str, required=True, help='Metafile .cs file to replace with.')
    parser.add_argument('--overwrite', action='store_true', help='Allow overwriting output files.')

    args = parser.parse_args()

    print('##### Command #####\n\t' + ' '.join(sys.argv))
    args_print_str = '##### Input parameters #####\n'
    for opt, val in vars(args).items():
        args_print_str += '\t{} : {}\n'.format(opt, val)
    print(args_print_str)
    return args


def main(infile: str, outfile: str, metafile: str, overwrite: bool) -> None:
    assert os.path.exists(infile), f'Input file {infile} not exist'
    if not overwrite and os.path.exists(outfile):
        sys.exit(f'Outfile {outfile} already exists. Specify --overwrite to overwrite.')

    metadata = dataset.Dataset().load(metafile)
    num_ptcls = len(metadata)

    with open(infile) as f:
        inlines = f.readlines()

    outlines = []
    for line in inlines:
        if 'metafile' in line:
            line = re.sub(r"(\s*metafile:\s*'>)(.+\.cs)'", f'\\g<1>{metafile}', line)
        elif 'num_items' in line:
            line = re.sub(r"(\s*num_items:\s*)([0-9]+)", f'\\g<1>{num_ptcls}', line)
        outlines.append(line)

    with open(outfile, 'w') as f:
        f.writelines(outlines)
    print(f'Output file saved as {outfile}')


if __name__ == '__main__':
    args = parse_args()
    main(
        args.infile,
        args.outfile,
        args.metafile,
        args.overwrite
    )
