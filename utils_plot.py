import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
import pandas as pd
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

param2desc = {'Outliers proportion (cal)': 'outliers_prop',
              'Original calibration size': 'calib_size', 'Clean calibration size': 'labeling_budget',
              'Signal amplitude': 'signal_amp', 'level': 'level'}
desc2param = {}
for k,v in param2desc.items():
    desc2param[v] = k
method2legend = {'Naive': 'Standard', 'Oracle': 'Oracle (infeasible)', 'NT': 'Naive-Trim (invalid)',
                 'LT': 'Label-Trim', 'Clean': 'Small-Clean'}
palette4fig = {'Naive': 'red', 'Oracle': 'limegreen', 'NT': 'darkorange', 'Clean': 'dimgray', 'LT': 'cyan'}
markers4fig = {'Naive': '>', 'Oracle': 'D', 'NT': 'P', 'Clean': 's', 'LT': '^'}
hue_order_g = ['Naive', 'Oracle', 'NT', 'Clean', 'LT']
hue_order_g_legend = [method2legend[m] if m in method2legend.keys() else m for m in hue_order_g]


def desc4paper(x):
    desc4paper_dict = {'mu_outlier': 'Signal amplitude',
                       'n_cal': 'Size of calibration set',
                       'dataset': 'Dataset',
                       'r-variance': 'Variance',
                       'Outliers proportion (cal)': 'Contamination rate',
                       'Original calibration size': 'Calibration set size',
                       'Clean calibration size': 'Labeling budget',
                       'initial_labeled': 'Labeling budget',
                       'level': r'$\alpha$',
                       'Type-1-Error': 'Type-I Error',
                       }
    if x in desc4paper_dict.keys():
        return desc4paper_dict[x]
    else:
        return x


def plot_xy(results, level, save_path=None, x='dataset', y=[], hue='Type', extract_data=False, file_desc='', exp=False):
    global hue_order_g
    font_size = 16
    font_size_labels = 20
    font_size_legend = 20
    if extract_data:
        level = results['level'].values[0]
        print(f'args for plot: level={level}')
    # filter according to args
    if x != 'level':
        results = results[results['level'] == level]
    if x == 'initial_labeled' or x == 'Clean calibration size':
        results = results.loc[results['Type'].isin(['Oracle', 'Clean', 'LT'])]
    if save_path is not None:
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
    if y is None or len(y) == 0:
        y = ['Power', 'Type-1-Error', 'Rejections', 'Threshold']
        y.extend(['Trimmed', 'Trimmed Inliers', 'Trimmed Outliers (proportion)', 'Trimmed Outliers', 'Model-Threshold'])
        if 'Threshold in 1' in results.columns:
            y.append('Threshold in 1')
    all_results = results
    for y_ in y:
        for box in [True, False]:
            if x == 'postprocess':
                fig = plt.figure(figsize=([10, 3]))
            else:
                fig = plt.figure(figsize=([5, 4]))
            ax = fig.add_subplot(111)
            if y_ == 'Trimmed':
                results = all_results.loc[all_results['Type'].isin(['LT'])]
            else:
                results = all_results
            if box:
                desc = ''
                sns.boxplot(results, x=x, hue=hue, y=y_, palette=palette4fig, ax=ax, hue_order=hue_order_g)
            else:
                if exp:
                    desc = '_point'
                    # arrange markers list
                    _, idx = np.unique(results[hue].values, return_index=True)
                    markers_list = [markers4fig[h] for h in hue_order_g]
                    ax = sns.pointplot(results, x=x, hue=hue, y=y_, palette=palette4fig, markers=markers_list, ax=ax, hue_order=hue_order_g)
                    for line in ax.lines:
                        line.set_alpha(0.6)
                else:
                    desc = '_bar'
                    sns.barplot(results, x=x, hue=hue, y=y_, palette=palette4fig, ax=ax, hue_order=hue_order_g)
            if y_ == 'Type-1-Error' and x != 'level':
                ax.axhline(level, color='black', linestyle='dashed', label='Target type-I level')
                if x == 'initial_labeled' or x == 'Clean calibration size':
                    ymin, ymax = ax.get_ylim()
                    ymax = max(ymax, level+0.005)
                    ax.set_ylim([ymin, ymax])
            elif y_ == 'Type-1-Error' and x == 'level':
                alphas = sorted(list(set(results[x])))
                ax.axline((0, alphas[0]), (len(alphas) - 1, alphas[-1]), color='black', linestyle='dashed', label='Target type-I level')
            if y_ == 'Trimmed':
                if x != 'initial_labeled' and x != 'Clean calibration size':
                    l_budget = results['initial_labeled'].values[0]
                    ax.axhline(l_budget, color='red', linestyle='dashdot', label=r'Labeling budget')
                else:
                    m_list = sorted(list(set(results[x])))
                    ax.axline((0, m_list[0]), (len(m_list) - 1, m_list[-1]), color='red', linestyle='dashdot', label=r'Labeling budget $m$')
                if x == 'Original calibration size' or x == 'Outliers proportion (cal)':
                    values = sorted(list(set(results[x])))
                    if x == 'Outliers proportion (cal)':
                        min_value = values[0] * results['Original calibration size'].values[0]
                        max_value = values[-1] * results['Original calibration size'].values[0]
                    if x == 'Original calibration size':
                        min_value = values[0] * results['p_cal'].values[0]
                        max_value = values[-1] * results['p_cal'].values[0]
                    ax.axline((0, min_value), (len(values) - 1, max_value), color='black', linestyle='dotted', label='# outliers')
                else:
                    n_outliers = results['Original calibration size'].values[0] * results['p_cal'].values[0] 
                    ax.axhline(n_outliers, color='black', linestyle='dotted', label='# outliers')

            plt.rc('legend', fontsize=font_size_legend)
            ax.tick_params(labelrotation=45)
            locs, labels = plt.xticks()
            if len(set(results[x])) > 10:
                new_locs = []
                new_labels = []
                for i in range(len(locs)):
                    if i%2 == 0:
                        new_locs.append(locs[i])
                        new_labels.append(labels[i])
                plt.xticks(new_locs, new_labels)
            plt.tick_params(axis='both', which='major', labelsize=font_size)
            ax.set_xlabel(desc4paper(x), fontsize=font_size_labels)
            ax.set_ylabel(y_, fontsize=font_size_labels)
            fig.tight_layout()
            handles, labels = ax.get_legend_handles_labels()
            labels = [method2legend[x] if x in method2legend.keys() else x for x in labels]
            plt.legend(handles, labels, loc='center left', bbox_to_anchor=(1.04, 0.5))
            if save_path is not None:
                plt.savefig(save_path + '/' + file_desc + y_ + desc + '.pdf', bbox_inches="tight")
            if y_ == 'Trimmed' or y_ == 'Type-1-Error':
                handles, labels = ax.get_legend_handles_labels()
                keep = -1 if y_ == 'Type-1-Error' else -2
                plt.legend(handles[keep:], labels[keep:])
                plt.savefig(save_path + '/' + file_desc + y_ + desc + '_short_legend.pdf', bbox_inches="tight")
            ax.get_legend().remove()
            if save_path is not None:
                plt.savefig(save_path + '/' + file_desc + y_ + desc + '_no_legend.pdf', bbox_inches="tight")


def plot(results, level=0.02, save_path=None, extract_data=False):
    if extract_data:
        level = results['level'].values[0]
    if save_path is not None:
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
    y_list = ['Power', 'Type-1-Error', 'Rejections', 'Threshold']
    if 'Threshold in 1' in results.columns:
        y_list.append('Threshold in 1')
    fig, axs = plt.subplots(2, 3, figsize=(9,4), sharex=True)
    for i in range(len(y_list)):
        row = int(i / 3)
        col = i % 3
        sns.boxplot(results, x='Type', y=y_list[i], palette=palette4fig, ax=axs[row][col])
        if y_list[i] == 'Type-1-Error' and len(level) == 1:
            level = float(level[0])
            axs[row][col].axhline(level, color='black', linestyle='dashed')
        axs[row][col].tick_params(labelrotation=90)
    fig.tight_layout()
    if save_path is not None:
        plt.savefig(save_path + '/performance.png', bbox_inches="tight")
    fig, axs = plt.subplots(3, 3, sharex=True, figsize=(9, 6))
    sns.boxplot(results, x='Type', y='Trimmed', ax=axs[0][0], palette=palette4fig)
    sns.boxplot(results, x='Type', y='Trimmed Outliers', ax=axs[0][1], palette=palette4fig)
    sns.boxplot(results, x='Type', y='Trimmed Inliers', ax=axs[0][2], palette=palette4fig)
    sns.boxplot(results, x='Type', y='Model-Threshold', ax=axs[1][0], palette=palette4fig)
    sns.boxplot(results, x='Type', y='Calibration size', ax=axs[1][1], palette=palette4fig)
    sns.boxplot(results, x='Type', y='Calibration inliers', ax=axs[1][2], palette=palette4fig)
    sns.boxplot(results, x='Type', y='Calibration outliers', ax=axs[2][0], palette=palette4fig)
    for i in range(9):
        r, c = int(i / 3), int(i % 3)
        axs[r][c].tick_params(labelrotation=90)
    fig.tight_layout()
    if save_path is not None:
        plt.savefig(save_path + '/trim_info.png', bbox_inches="tight")


def plot_table(results, x='dataset', save_path=None, file_desc=''):
    level = results['level'].values[0]
    ood_datasets = list(set(results['outlr_dataset']))
    if len(ood_datasets) == 1:
        table_mean, table_se, relative_t_mean, relative_t_se = get_mean_se_tables(results, x)
    else:
        # combine results from different ood datasets
        print('Combining results from the following outlr datasets: ', ood_datasets)
        table_mean, table_se = pd.DataFrame({}), pd.DataFrame({})
        relative_t_mean, relative_t_se = pd.DataFrame({}), pd.DataFrame({})
        for ood_dataset in ood_datasets:
            curr_results = results[results['outlr_dataset'] == ood_dataset]
            _table_mean, _table_se, _relative_t_mean, _relative_t_se = get_mean_se_tables(curr_results, x)
            table_mean, table_se = pd.concat([table_mean, _table_mean]), pd.concat([table_se, _table_se])
            relative_t_mean, relative_t_se = pd.concat([relative_t_mean, _relative_t_mean]), pd.concat([relative_t_se, _relative_t_se])
        table_mean = table_mean.groupby(['Type'], sort=False).mean()
        table_se = table_se.groupby(['Type'], sort=False).mean()
        relative_t_mean = relative_t_mean.groupby(['Type'], sort=False).mean()
        relative_t_se = relative_t_se.groupby(['Type'], sort=False).mean()

    # latex table plot
    combined_table = table_mean.copy()
    combined_table_relative = relative_t_mean.copy()
    combined_table_relative_p = relative_t_mean.copy()
    x_values = list(set(results[x].to_list()))
    methods = list(set(results['Type'].to_list()))

    for metric in ['Power', 'Type-1-Error']:
        for x_value in x_values:
            for method in methods:
                mean = table_mean[(metric, 'mean', x_value)][method]
                se = table_se[(metric, 'se', x_value)][method]
                r_mean = relative_t_mean[(metric, 'mean', x_value)][method]
                r_se = relative_t_se[(metric, 'se', x_value)][method]
                s_value = f'{round(mean, 3)} ($\pm$ {round(se, 4)})'
                r_s_value = f'{round(r_mean, 3)} ($\pm$ {round(r_se, 4)})'
                combined_table[(metric, 'mean', x_value)][method] = s_value
                combined_table_relative[(metric, 'mean', x_value)][method] = r_s_value
                
                r_p_s_value = r_s_value if metric == 'Power' else s_value
                combined_table_relative_p[(metric, 'mean', x_value)][method] = r_p_s_value
    combined_table = combined_table.style.apply(color_func, mean_results=table_mean, x_values=x_values, methods=methods, level=level, axis=None)
    combined_table_relative = combined_table_relative.style.apply(color_func, mean_results=table_mean, x_values=x_values, methods=methods, level=level, axis=None)
    combined_table_relative_p = combined_table_relative_p.style.apply(color_func, mean_results=table_mean, x_values=x_values, methods=methods, level=level, axis=None)
    latex_table = combined_table.to_latex()
    latex_r_table = combined_table_relative.to_latex()
    latex_r_p_table = combined_table_relative_p.to_latex()
    for method in method2legend.keys():
        latex_table = latex_table.replace(method + ' ', method2legend[method] + ' ')
        latex_r_table = latex_r_table.replace(method + ' ', method2legend[method] + ' ')
        latex_r_p_table = latex_r_p_table.replace(method + ' ', method2legend[method] + ' ')
    try:
        os.makedirs(save_path + '/table_files/')
    except:
        pass
    with open(save_path + '/table_files/' + file_desc + 'table.tex', 'w') as f:
        f.write(latex_table)
    with open(save_path + '/table_files/' + file_desc + 'relative_table.tex', 'w') as f:
        f.write(latex_r_table)
    with open(save_path + '/table_files/' + file_desc + 'relative_p_table.tex', 'w') as f:
        f.write(latex_r_p_table)


def color_func(results, mean_results, x_values, methods, level, type1=True, power=True):
    colors = results.copy()
    colors.loc[:, :] = None
    for x_value in x_values:
        top_3_values = []
        for method in methods:
            curr_type1 = round(mean_results[('Type-1-Error', 'mean', x_value)][method], 3)
            curr_power = round(mean_results[('Power', 'mean', x_value)][method], 3)
            if curr_type1 >= level + 0.003:
                if type1:
                    colors[('Type-1-Error', 'mean', x_value)][method] = "cellcolor: {red!20};"
                if power:
                    colors[('Power', 'mean', x_value)][method] = "cellcolor: {red!20};"
            else:
                if len(top_3_values) < 3:
                    top_3_values.append(curr_power)
                else:
                    if min(top_3_values) < curr_power:
                        top_3_values.remove(min(top_3_values))
                        top_3_values.append(curr_power)
        gr_colors = {'max': '{Green!100}', 'mid': '{Green!60}', 'min': '{Green!30}'}
        for method in methods:
            curr_type1 = round(mean_results[('Type-1-Error', 'mean', x_value)][method], 3)
            curr_power = round(mean_results[('Power', 'mean', x_value)][method], 3)
            if curr_type1 < level + 0.003:
                if type1 and curr_type1 <= level:
                    colors[('Type-1-Error', 'mean', x_value)][method] = "cellcolor: {white};"   # for alignment
                if min(top_3_values) <= curr_power:
                    if power:
                        if curr_power == max(top_3_values):
                            color = gr_colors['max']
                        elif curr_power == min(top_3_values):
                            color = gr_colors['min']
                        else:
                            color = gr_colors['mid']
                        colors[('Power', 'mean', x_value)][method] = f"bfseries: ; cellcolor: {color};"
                else:
                    if power:
                        colors[('Power', 'mean', x_value)][method] = "cellcolor: {white};"
    return colors


def get_mean_se_tables(results, x):
    results.sort_values(by="Type", key=lambda column: column.map(lambda e: hue_order_g.index(e)), inplace=True)
    # keep only power and type-I-error columns
    curr_results = results[['Power', 'Type-1-Error', 'Type', x]]
    stats = curr_results.groupby(['Type',x], sort=False).describe()
    drop_list = [('Type-1-Error',  '25%'),
                ('Type-1-Error',  '50%'),
                ('Type-1-Error',  '75%'),
                ('Type-1-Error',  'max'),
                ('Type-1-Error',  'min'),
                ('Type-1-Error',  'count'),
                ('Power',  '25%'),
                ('Power',  '50%'),
                ('Power',  '75%'),
                ('Power',  'max'),
                ('Power',  'min'),
                ('Power',  'count'),
                ]
    count = list(stats[('Power',  'count')])[0]
    table = stats.drop(drop_list, axis=1)
    table_mean = pd.pivot_table(table, values=[('Power', 'mean'),('Type-1-Error', 'mean')], index=['Type'], columns=[x])
    table_std = pd.pivot_table(table, values=[('Power', 'std'),('Type-1-Error', 'std')], index=['Type'], columns=[x])
    # sort results by hue_order
    table_mean.sort_values(by="Type", key=lambda column: column.map(lambda e: hue_order_g.index(e)), inplace=True)
    table_std.sort_values(by="Type", key=lambda column: column.map(lambda e: hue_order_g.index(e)), inplace=True)
    # latex table plot
    x_values = list(set(results[x].to_list()))
    methods = list(set(results['Type'].to_list()))
    relative_t_mean = table_mean.copy()
    table_se = table_std.copy().rename(columns={'std': 'se'})
    relative_t_se = table_se.copy()
    for metric in ['Power', 'Type-1-Error']:
        for x_value in x_values:
            oracle_mean = table_mean[(metric, 'mean', x_value)]['Oracle']
            naive_mean = table_mean[(metric, 'mean', x_value)]['Naive']
            for method in methods:
                mean = table_mean[(metric, 'mean', x_value)][method]
                std = table_std[(metric, 'std', x_value)][method]
                se = std / np.sqrt(count)
                table_se[(metric, 'se', x_value)][method] = se
                relative_mean = round(mean / naive_mean, 3)
                relative_se = round(se / naive_mean, 4)
                relative_t_mean[(metric, 'mean', x_value)][method] = relative_mean
                relative_t_se[(metric, 'se', x_value)][method] = relative_se
    return table_mean, table_se, relative_t_mean, relative_t_se


