import numpy as np
import matplotlib.pyplot as plt

category_names = ['Strongly Disagree', 'Disagree', 'Neutral',
                  'Agree', 'Strongly Agree']
results = {
    'I am concerned that batteries used to power smart devices\n and embedded systems are harming the environment.': [np.nan,
                                                                                                                      np.nan,
                                                                                                                      2,
                                                                                                                      4,
                                                                                                                      2],
    'MakeCode-Iceberg\'s checkpointing while operating\n under intermittent power would save time. ': [1, np.nan, 1, 2, 4],
    'MakeCode-Iceberg enables the development and use of applications\n outside of a controlled environment.': [np.nan, 1, 2,
                                                                                                                1, 4],
    'I would choose MakeCode-Iceberg over Regular MakeCode\n when developing programs with energy harvesting devices.': [
        np.nan, 1, 1, np.nan, 6]
}


def survey(results, category_names):
    """
    Parameters
    ----------
    results : dict
        A mapping from question labels to a list of answers per category.
        It is assumed all lists contain the same number of entries and that
        it matches the length of *category_names*.
    category_names : list of str
        The category labels.
    """
    labels = list(results.keys())
    datavals = list(results.values())
    data = np.array(datavals)
    data_cum = data.cumsum(axis=1)
    category_colors = plt.get_cmap('RdYlGn')(
        np.linspace(0.15, 0.85, data.shape[1]))
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.invert_yaxis()
    ax.xaxis.set_visible(False)
    ax.set_xlim(0, np.nansum(data, axis=1).max())
    for i, (colname, color) in enumerate(zip(category_names, category_colors)):
        widths = data[:, i]
        starts = data_cum[:, i] - widths
        rects = ax.barh(labels, widths, left=starts, height=0.5,
                        label=colname, color=color)
        r, g, b, _ = color
        text_color = 'black'
        print(rects)
        ax.bar_label(rects, fontsize=10, label_type='center', color=text_color)

    ax.legend(ncol=len(category_names), bbox_to_anchor=(0, 1),
              loc='lower left', fontsize=10, edgecolor='none')
    return fig, ax


survey(results, category_names)
plt.show()