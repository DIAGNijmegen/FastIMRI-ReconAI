import sys
import json
from pathlib import Path
from scipy.stats import ttest_ind
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

def get_dice_scores():
    models = ['recon_crnn_ne_8_64_5_seg', 'recon_crnn_ne_16_64_5_seg', 'recon_crnn_ne_25_64_5_seg',
              'recon_crnn_ne_32_64_5_seg',
              'recon_crnn_ne_8_128_10_seg', 'recon_crnn_ne_16_128_10_seg', 'recon_crnn_ne_25_128_10_seg',
              'recon_crnn_ne_32_128_10_seg',
              'recon_ne_8_64_5_seg', 'recon_ne_16_64_5_seg', 'recon_ne_25_64_5_seg', 'recon_ne_32_64_5_seg',
              'recon_ne_8_128_10_seg', 'recon_ne_16_128_10_seg', 'recon_ne_25_128_10_seg', 'recon_ne_32_128_10_seg'
              ]

    model_results = dict()
    for model in models:
        with open(Path(f'../../../segmentation/test/{model}/summary.json').resolve()) as json_data:
            summary_json = json.load(json_data)

            model_result = []
            for case in summary_json["metric_per_case"]:
                model_result.append(case["metrics"]["(1,)"]["Dice"])
        model_results[model] = model_result

    # print(model_results['recon_ne_32_128_10_seg'])

    return models, model_results

def print_p_values():
    models, model_results = get_dice_scores()
    for i in range(8):
        print(ttest_ind(model_results[models[i]], model_results[models[i + 8]]))
    for i in range(4):
        print(ttest_ind(model_results[models[i + 8]], model_results[models[i + 12]]))
    for i in range(4):
        print(ttest_ind(model_results[models[i]], model_results[models[i + 4]]))

def get_df_section(model_results, models, idx, model, undersampling):
    df = pd.DataFrame.from_dict(model_results)
    df = df.rename(columns={f"{models[idx]}": "dice"})
    df['model'] = model
    df['undersampling'] = undersampling
    df = df[['dice', 'model', 'undersampling']]
    df['mean'] = df['dice'].mean()
    return df

def get_dataframe():
    models, model_results = get_dice_scores()

    nann_8 = get_df_section(model_results, models, 0, 'NANN', 8)
    nann_16 = get_df_section(model_results, models, 1, 'NANN', 16)
    nann_25 = get_df_section(model_results, models, 2, 'NANN', 25)
    nann_32 = get_df_section(model_results, models, 3, 'NANN', 32)
    nannl_8 = get_df_section(model_results, models, 4, 'NANN-L', 8)
    nannl_16 = get_df_section(model_results, models, 5, 'NANN-L', 16)
    nannl_25 = get_df_section(model_results, models, 6, 'NANN-L', 25)
    nannl_32 = get_df_section(model_results, models, 7, 'NANN-L', 32)

    stnn_8 = get_df_section(model_results, models, 8, 'STNN', 8)
    stnn_16 = get_df_section(model_results, models, 9, 'STNN', 16)
    stnn_25 = get_df_section(model_results, models, 10, 'STNN', 25)
    stnn_32 = get_df_section(model_results, models, 11, 'STNN', 32)
    stnnl_8 = get_df_section(model_results, models, 12, 'STNN-L', 8)
    stnnl_16 = get_df_section(model_results, models, 13, 'STNN-L', 16)
    stnnl_25 = get_df_section(model_results, models, 14, 'STNN-L', 25)
    stnnl_32 = get_df_section(model_results, models, 15, 'STNN-L', 32)

    frames = [nann_8, nann_16, nann_25, nann_32, nannl_8, nannl_16, nannl_25, nannl_32,
              stnn_8, stnn_16, stnn_25, stnn_32, stnnl_8, stnnl_16, stnnl_25, stnnl_32]

    # print(nann_25.describe())

    return pd.concat(frames)

def print_plots():
    df = get_dataframe()
    sns.set({'figure.figsize': (12, 8), 'axes.facecolor': 'white', 'figure.facecolor': 'white'})
    sns.set_theme(style='white')

    means = df.groupby(['model', 'undersampling'])['dice'].aggregate({'mean'})
    means = means.rename(columns={'mean': 'dice'})
    print(means.head())

    fig, ax = plt.subplots()

    violinplot = sns.violinplot(df, x='undersampling', y='dice', hue='model', width=1, bw=.15, cut=0,
                                palette=["grey", "royalblue", "darkorange", "yellow"])
                                # inner=None, dodge=True, ax=ax)
    scatter = sns.scatterplot(data=df, x='undersampling', y='mean', ax=ax)  # , palette=["black"])
    violinplot.legend(loc='lower left')

    fig = violinplot.get_figure()
    fig.savefig(f'../../../segmentation/violin.png')

    # plt.clf()
    #
    # lineplot = sns.lineplot(df, x='undersampling', y='dice', hue='model',
    # palette=["grey", "royalblue", "darkorange", "yellow"])
    # lineplot.legend(loc='lower left')
    # fig = lineplot.get_figure()
    # fig.savefig(f'../../../segmentation/lineplot.png')




def main():

    # print_p_values()
    print_plots()


if __name__ == '__main__':
    main()
