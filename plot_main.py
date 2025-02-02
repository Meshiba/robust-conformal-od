import argparse
import os
from utils_plot import plot_xy, plot_table
import pandas as pd


def get_data(file_path):
    results = pd.read_pickle(file_path)
    return results


def plot_figures(results, x, y, save_path=None,exp=False, filter_={}, file_desc='', table=False):
    for k,v in filter_.items():
        results = results[results[k].astype(str) == str(v)]
    if results.empty:
        print('Results dataframe is empty... exit')
        return
    if table:
        plot_table(results, x=x, save_path=save_path, file_desc=file_desc)
        return
    plot_xy(results, x=x, y=y, level=None, save_path=save_path, extract_data=True, file_desc=file_desc, exp=exp)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_dir', type=str, nargs='+', default='./results/')
    parser.add_argument('--plot_dir', type=str, default='./plots/')
    parser.add_argument('--separate', action='store_true', help='Create separate plot figure for each results file.')
    parser.add_argument('--x', type=str, default='dataset')
    parser.add_argument('--y', type=str, nargs='+', default=None)
    parser.add_argument('--filter_k', type=str, nargs='+', default=[])
    parser.add_argument('--filter_v', nargs='+', default=[])
    parser.add_argument('--file_desc', type=str, default='Suffix to append to the filename for the plot output.')
    parser.add_argument('--exp', action='store_true', help='Plot exp.')
    parser.add_argument('--table', action='store_true', help='Plot LaTeX table (ignoring the y argument and plotting '
                                                             'only Power and Type-I error).')
    args = parser.parse_args()
    if len(args.filter_k) != len(args.filter_v):
        raise ValueError('The length of filter keys and values must be the same.')
    return args


def main(args):
    filter_dict = {}
    for i in range(len(args.filter_k)):
        filter_dict[args.filter_k[i]] = args.filter_v[i]
    all_results = pd.DataFrame({})
    if not isinstance(args.result_dir, list):
        args.result_dir = [args.result_dir]
    result_files = []
    for res_dir in args.result_dir:
        result_files.extend([os.path.join(res_dir, f) for f in os.listdir(res_dir)])
    for f in result_files:
        results = get_data(f)
        if args.separate:
            plot_figures(results, x=args.x, y=args.y, save_path=(args.plot_dir + os.path.basename(f).rstrip('.pkl') + '/'), 
                         exp=args.exp, filter_=filter_dict, file_desc=args.file_desc)
        else:
            all_results = pd.concat([all_results, results])
    if not args.separate:
        if args.exp:
            exp = args.result_dir[0].rstrip('/results.pkl')
            plot_path = args.plot_dir + os.path.basename(exp) + '/'
        else:
            plot_path = args.plot_dir
        plot_figures(all_results, x=args.x, y=args.y, save_path=plot_path, filter_=filter_dict, file_desc=args.file_desc,
                exp=args.exp, table=args.table)


if __name__ == "__main__":
    args = get_args()
    main(args)


