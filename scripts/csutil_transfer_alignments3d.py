""" cryosparcで切り出しした粒子をRELIONで処理し、それをcryosparcへimportして、再度cryosparcで切り出しを行う際に使うスクリプト。
"""

import sys
import os
import argparse
import pandas as pd
import numpy as np
from cryosparc_compute import dataset


def get_blobpath_basename(arr, dont_remove_uuid=True):
    out_arr = []
    for s in arr:
        # Filename
        s = os.path.basename(s)
        # Remove file extension
        s = os.path.splitext(s)[0]
        if not dont_remove_uuid:
            # Split by '_', discard the first word (new UID), then concatenate
            s = '_'.join(s.split('_')[1:])
        out_arr.append(s)
    if isinstance(arr, np.ndarray):
        out_arr = np.array(out_arr)
    return out_arr


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__
    )
    parser.add_argument('--orig_cs_file', required=True, help='The extracted_particles.cs file of the extraction job with binning.')
    parser.add_argument('--orig_passthrough_file', required=True, help='The P*_J*_passthrough_particles.cs file of the extraction job with binning.')
    parser.add_argument('--imported_cs_file', required=True, help='The imported_particles.cs file of the import job.')
    parser.add_argument('--output_cs_file', required=True, help='Output cs file.')
    parser.add_argument('--dont_remove_blobpath_uuid', action='store_true', help='Do not remove UUIDs from the imported blobpaths')

    args = parser.parse_args()

    print('##### Command #####\n\t' + ' '.join(sys.argv))
    args_print_str = '##### Input parameters #####\n'
    for opt, val in vars(args).items():
        args_print_str += '\t{} : {}\n'.format(opt, val)
    print(args_print_str)
    return args


def main():
    args = parse_args()

    print(f'Loading {args.orig_cs_file} ...')
    orig_dataset = dataset.Dataset().from_file(args.orig_cs_file)
    print(f'Loading {args.orig_passthrough_file} ...')
    orig_passthrough = dataset.Dataset().from_file(args.orig_passthrough_file)
    print(f'Loading {args.imported_cs_file} ...')
    imported_dataset = dataset.Dataset().from_file(args.imported_cs_file)

    assert len(orig_dataset) == len(orig_passthrough)
    # The imported dataset must be a subset of the original dataset.
    assert len(imported_dataset) <= len(orig_dataset)

    # Combine the original dataset and passthrough infos
    print(f'Combining the original dataset and passthrough infos...')
    orig_dataset_passthrough = dataset.Dataset().from_dataset(orig_passthrough).innerjoin(orig_dataset)

    # np.ndarray containing the particle image paths
    imported_blobpaths = imported_dataset.data['blob/path']
    imported_blobpaths_basename = get_blobpath_basename(imported_blobpaths, dont_remove_uuid=args.dont_remove_blobpath_uuid)
    imported_blobpaths_basename_uniq = np.unique(imported_blobpaths_basename)

    df_orig = orig_dataset_passthrough.to_dataframe()
    df_imported = imported_dataset.to_dataframe()
    df_output = pd.DataFrame()

    alignments3d_cols = [col for col in df_imported.columns if 'alignments3D/' in col]

    for i in range(len(imported_blobpaths_basename_uniq)):
        if (i != 0) and (i % 10 == 0):
            print(f'Processed {i} particle stacks.')

        blobpath_query = imported_blobpaths_basename_uniq[i]

        df_imported_queried = df_imported.loc[df_imported['blob/path'].str.contains(blobpath_query)]
        df_imported_queried_sort = df_imported_queried.sort_values('blob/idx', axis=0, inplace=False, ignore_index=True)

        df_orig_queried = df_orig.loc[df_orig['blob/path'].str.contains(blobpath_query)]
        df_orig_idxmatched = df_orig_queried.loc[df_orig_queried['blob/idx'].isin(df_imported_queried_sort['blob/idx'])]
        df_orig_idxmatched_sort = df_orig_idxmatched.sort_values('blob/idx', axis=0, inplace=False, ignore_index=True)

        assert np.all(df_orig_idxmatched_sort['blob/idx'].values == df_imported_queried_sort['blob/idx'].values)

        # Transfer the 3D poses
        df_orig_idxmatched_sort[alignments3d_cols] = df_imported_queried_sort[alignments3d_cols]

        df_output = df_output.append(df_orig_idxmatched_sort, ignore_index=True)

    assert len(df_output) == len(df_imported)

    print(f'The number of the total particles: {len(df_output)}')


if __name__ == '__main__':
    main()
