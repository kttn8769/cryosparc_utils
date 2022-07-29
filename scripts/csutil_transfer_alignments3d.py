""" cryosparcで切り出しした粒子をRELIONで処理し、それをcryosparcへimportして、再度cryosparcで切り出しを行う際に使うスクリプト。
"""

import sys
import os
import argparse
import datetime
import yaml
import pandas as pd
import numpy as np
from cryosparc_compute import dataset


def get_blobpath_basename(arr, num_remove_uuid):
    out_arr = []
    for s in arr:
        # Filename
        s = os.path.basename(s)
        # Remove file extension
        s = os.path.splitext(s)[0]
        s = '_'.join(s.split('_')[num_remove_uuid:])
        out_arr.append(s)
    if isinstance(arr, np.ndarray):
        out_arr = np.array(out_arr)
    return out_arr


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__
    )
    parser.add_argument('--orig_cs_file', required=True, help='The cs file of the cryosparc job which was exported to RELION.')
    parser.add_argument('--orig_csg_file', required=True, help='The csg file of the cryosparc job which was exported to RELION.')
    parser.add_argument('--orig_passthrough_file', required=True, help='The P*_J*_passthrough_particles.cs file of the cryosparc job which was exported to RELION.')
    parser.add_argument('--imported_cs_file', required=True, help='The imported_particles.cs file of the import job. (import from RELION)')
    parser.add_argument('--output_cs_file', required=True, help='Output cs file.'),
    parser.add_argument('--orig_num_remove_blobpath_uuid', type=int, default=1, help='Preceding UUID strings will be removed from blobpaths of the original dataset this many times.')
    parser.add_argument('--imported_num_remove_blobpath_uuid', type=int, default=2, help='Preceding UUID strings will be removed from blobpaths of the imported dataset this many times.')

    args = parser.parse_args()

    print('##### Command #####\n\t' + ' '.join(sys.argv))
    args_print_str = '##### Input parameters #####\n'
    for opt, val in vars(args).items():
        args_print_str += '\t{} : {}\n'.format(opt, val)
    print(args_print_str)
    return args


def main():
    args = parse_args()

    assert not os.path.exists(args.output_cs_file), f'The output cs file {args.output_cs_file} already exists. If you want to overwride the file, manualy remove it before use this script.'
    output_cs_file_basename = os.path.basename(args.output_cs_file)
    output_csg_file = os.path.splitext(args.output_cs_file)[0] + '.csg'
    assert not os.path.exists(output_csg_file), f'The output csg file {output_csg_file} already exists. If you want to overwride the file, manualy remove it before use this script.'

    print(f'Loading {args.orig_csg_file} ...')
    with open(args.orig_csg_file, 'r') as f:
        orig_csg = yaml.load(f, Loader=yaml.FullLoader)

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

    print(f'Preparing the original dataset infos')
    df_orig = orig_dataset_passthrough.to_dataframe()
    df_orig_dtypes = df_orig.dtypes.to_dict()
    # Convert to numpy array for speeding up.
    arr_orig = df_orig.to_numpy(copy=True)
    cols_orig = list(df_orig.columns)
    cols_orig_colidx_blobpath = cols_orig.index('blob/path')
    cols_orig_colidx_blobidx = cols_orig.index('blob/idx')
    print(f'Original blobpath example: {arr_orig[0, cols_orig_colidx_blobpath]}')
    orig_blobpaths_basename = get_blobpath_basename(
        arr_orig[:, cols_orig_colidx_blobpath].flatten().astype(str),
        args.orig_num_remove_blobpath_uuid
    )
    print(f'Original blobpath basename example: {orig_blobpaths_basename[0]}')
    # Sort by blobpath basename for faster search
    arr_orig_sort_idxs = np.argsort(orig_blobpaths_basename, axis=0)
    arr_orig = arr_orig[arr_orig_sort_idxs]
    orig_blobpaths_basename_sorted = orig_blobpaths_basename[arr_orig_sort_idxs]

    print(f'Preparing the imported dataset infos')
    df_imported = imported_dataset.to_dataframe()
    df_imported_dtypes = df_imported.dtypes.to_dict()
    arr_imported = df_imported.to_numpy(copy=True)
    cols_imported = list(df_imported.columns)
    cols_imported_colidx_blobpath = cols_imported.index('blob/path')
    cols_imported_colidx_blobidx = cols_imported.index('blob/idx')
    print(f'Imported blobpath example: {arr_imported[0, cols_imported_colidx_blobpath]}')
    imported_blobpaths_basename = get_blobpath_basename(
        arr_imported[:, cols_imported_colidx_blobpath].flatten().astype(str),
        args.imported_num_remove_blobpath_uuid
    )
    print(f'Imported blobpath basename example: {imported_blobpaths_basename[0]}')
    arr_imported_sort_idxs = np.argsort(imported_blobpaths_basename, axis=0)
    arr_imported = arr_imported[arr_imported_sort_idxs]
    imported_blobpaths_basename_sorted = imported_blobpaths_basename[arr_imported_sort_idxs]
    imported_blobpaths_basename_uniq = np.unique(imported_blobpaths_basename_sorted)

    alignments3d_cols = [col for col in cols_imported if 'alignments3D/' in col]

    output_df_list = []

    print('Processing ....')
    n = len(imported_blobpaths_basename_uniq)
    progbar_length = 100
    progbar_step = int(n / progbar_length)
    # progbar_step = 1
    num_step = 0
    for i in range(len(imported_blobpaths_basename_uniq)):
    # for i in range(10):
        if i % progbar_step == 0:
            sys.stdout.write('\r')
            sys.stdout.write(
                'Progress: [{:{}}] ({:7d} / {:7d})'
                .format(
                    '|' * num_step,
                    progbar_length,
                    i + 1,
                    n
                )
            )
            num_step += 1

        blobpath_query = imported_blobpaths_basename_uniq[i]

        arr_imported_same_blobpath_end_idx = len(imported_blobpaths_basename_sorted)
        for j in range(len(imported_blobpaths_basename_sorted)):
            if blobpath_query != imported_blobpaths_basename_sorted[j]:
                arr_imported_same_blobpath_end_idx = j
                break
        arr_imported_same_blobpath = arr_imported[:arr_imported_same_blobpath_end_idx]
        arr_imported_same_blobpath = arr_imported_same_blobpath[
            np.argsort(arr_imported_same_blobpath[:, cols_imported_colidx_blobidx].flatten().astype(np.int))
        ]
        arr_imported = arr_imported[arr_imported_same_blobpath_end_idx:]
        imported_blobpaths_basename_sorted = imported_blobpaths_basename_sorted[arr_imported_same_blobpath_end_idx:]

        for j in range(len(orig_blobpaths_basename_sorted)):
            if blobpath_query == orig_blobpaths_basename_sorted[j]:
                break
        arr_orig_same_blobpath_start_idx = j
        arr_orig_same_blobpath_end_idx = len(orig_blobpaths_basename_sorted)
        for j in range(arr_orig_same_blobpath_start_idx, len(orig_blobpaths_basename_sorted)):
            if blobpath_query != orig_blobpaths_basename_sorted[j]:
                arr_orig_same_blobpath_end_idx = j
                break
        arr_orig_same_blobpath = arr_orig[arr_orig_same_blobpath_start_idx:arr_orig_same_blobpath_end_idx]
        arr_orig_same_blobpath = arr_orig_same_blobpath[
            np.argsort(arr_orig_same_blobpath[:, cols_orig_colidx_blobidx].flatten().astype(np.int))
        ]
        arr_orig = arr_orig[arr_orig_same_blobpath_end_idx:]
        orig_blobpaths_basename_sorted = orig_blobpaths_basename_sorted[arr_orig_same_blobpath_end_idx:]

        arr_orig_same_blobpath_match_or_not = np.isin(
            arr_orig_same_blobpath[:, cols_orig_colidx_blobidx].astype(int),
            arr_imported_same_blobpath[:, cols_imported_colidx_blobidx].astype(int),
            assume_unique=True
        )
        arr_orig_same_blobpath_matched = arr_orig_same_blobpath[arr_orig_same_blobpath_match_or_not]
        # Sort by blob/idx
        arr_orig_same_blobpath_matched = arr_orig_same_blobpath_matched[
            np.argsort(arr_orig_same_blobpath_matched[:, cols_orig_colidx_blobidx].flatten().astype(np.int), axis=0)
        ]

        assert np.all(arr_imported_same_blobpath[:, cols_imported_colidx_blobidx] == arr_orig_same_blobpath_matched[:, cols_orig_colidx_blobidx])


        df_imported_same_blobpath = pd.DataFrame(arr_imported_same_blobpath, columns=cols_imported)
        df_orig_same_blobpath_matched = pd.DataFrame(arr_orig_same_blobpath_matched, columns=cols_orig)

        # Transfer the 3D poses
        df_orig_same_blobpath_matched[alignments3d_cols] = df_imported_same_blobpath[alignments3d_cols]

        output_df_list.append(df_orig_same_blobpath_matched)

    sys.stdout.write('\r')
    sys.stdout.write('Progress: [{:{}}] ({:7d} / {:7d})'.format('|' * progbar_length, progbar_length,  n, n))

    print('\nConcatenating dataframes...')
    df_out = pd.concat(output_df_list, ignore_index=True)
    df_out_dtypes = df_orig_dtypes
    for k, v in df_imported_dtypes.items():
        if 'alignments3D' in k:
            df_out_dtypes[k] = v
    df_out = df_out.astype(df_out_dtypes)

    assert len(df_out) == len(df_imported)

    print(f'The number of the total particles: {len(df_out)}')
    num_items = len(df_out)
    print(f'Saving output cs file...')
    output_dataset = dataset.Dataset().from_dataframe(df_out)
    output_dataset.to_file(args.output_cs_file)
    print(f'The output cs file {args.output_cs_file} saved.')

    orig_csg['group']['description'] = 'Created by csutil_transfer_alignments3d.py of https://github.com/kttn8769/cryosparc_utils.git'
    orig_csg['created'] = datetime.datetime.now()
    results_keys = orig_csg['results'].keys()
    for key in results_keys:
        orig_csg['results'][key]['metafile'] = f'>{output_cs_file_basename}'
        orig_csg['results'][key]['num_items'] = num_items
    if 'alignments3D' not in results_keys:
        orig_csg['results']['alignments3D'] = {
            'metafile': f'>{output_cs_file_basename}',
            'num_items': num_items,
            'type': 'particle.alignments3D'
        }
    with open(output_csg_file, 'w') as f:
        yaml.dump(orig_csg, stream=f)
    print(f'The accompanying csg file {output_csg_file}. Use this file for the input of Import Result Group job in cryoSPARC.')
    print('Program finished! Good luck!!')


if __name__ == '__main__':
    main()
