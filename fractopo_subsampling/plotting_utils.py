"""
Plotting utilities.
"""
import warnings
from itertools import count
from typing import Dict, Generator, Sequence, Tuple, Union

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.cbook import boxplot_stats
from matplotlib.figure import Figure
from pandas.core.groupby import DataFrameGroupBy

import fractopo_subsampling.utils as utils

dist_continous = [
    d for d in dir(stats) if isinstance(getattr(stats, d), stats.rv_continuous)
]


def base_circle_id_coords(filtered: Union[pd.DataFrame, gpd.GeoDataFrame]):
    """
    Id analysis points ordered by x coordinate.

    >>> df = pd.DataFrame(
    ...             {
    ...                 "x": [10, 15, 123, -4],
    ...                 "y": [2, 1, 1, -4],
    ...                 "name": ["a", "b", "c", "d"],
    ...             }
    ...         )
    >>> base_circle_id_coords(df)
    [(-4, -4, 'd'), (10, 2, 'a'), (15, 1, 'b'), (123, 1, 'c')]

    """
    coords = zip(filtered["x"].values, filtered["y"].values, filtered["name"].values)
    coords = sorted(coords, key=lambda vals: vals[0])  # type: ignore
    return coords


def base_circle_id_dict(coords: list):
    """
    Make dict out of ided points.

    >>> coords = [(-4, -4, 'd'), (10, 2, 'a'), (15, 1, 'b'), (123, 1, 'c')]
    >>> base_circle_id_dict(coords)
    {'d': 1, 'a': 2, 'b': 3, 'c': 4}

    """
    counter = count(1)
    id_dict = dict()
    for _, _, name in coords:
        new_id = next(counter)
        id_dict[name] = new_id
    return id_dict


def label_ids_to_map(coords, ax):
    """
    Label ids to a map based on coordinates.
    """
    counter = count(1)
    for x, y, _ in coords:
        new_id = next(counter)
        ax.text(x + 25, y + 25, new_id) if new_id != 11 else ax.text(
            x - 180, y - 100, new_id
        )
    return ax


def plot_base_circles(filtered: gpd.GeoDataFrame, coords: list, ax=None):
    """
    Plot base circle locations.
    """
    ax = filtered.plot(
        figsize=utils.paper_figsize(1),
        ax=ax,
        marker="o",
        markersize=filtered["area"] * 0.01,
        label="Base Circles",
        color="black",
        # legend_kwds={"edgecolor": "black", "loc": "lower right", "framealpha": 1},
    )
    ax = label_ids_to_map(coords, ax)
    return ax


def plot_shoreline(shoreline: gpd.GeoDataFrame, ax):
    """
    Plot shoreline.
    """
    ax = shoreline.plot(ax=ax, linewidth=0.25, label="Shoreline", color="black")
    return ax


def plot_base_circle_map(
    filtered: gpd.GeoDataFrame, shoreline: gpd.GeoDataFrame, ax=None
):
    """
    Plot map with base circles and shoreline.
    """
    coords = base_circle_id_coords(filtered)
    ax = plot_base_circles(filtered=filtered, coords=coords, ax=ax)
    ax = plot_shoreline(shoreline, ax)
    # Focus out western shoreline continuation
    ax.set_xlim(106000, 111000)
    # Move focus a bit up
    ax.set_ylim(6.7187 * 1e6, 6.7205 * 1e6)
    return ax


def preprocess_analysis_points(
    analysis_points: gpd.GeoDataFrame,
    circle_names_with_diameter: Dict[str, float],
    filter_radius: Tuple[float, float],
    relative_coverage_threshold: float,
):
    """
    Preprocess and filter Getaberget base circle analysis points.
    """
    filtered = utils.filter_dataframe(
        analysis_points,
        list(circle_names_with_diameter),
        filter_radius=filter_radius,
        relative_coverage_threshold=relative_coverage_threshold,
    )
    filtered["x"] = [point.x for point in filtered.geometry.values]
    filtered["y"] = [point.y for point in filtered.geometry.values]
    return filtered


def test_all_dists(data, list_of_dists):
    """
    Loop through every scipy continous distribution and gather KS-test results.
    """
    warnings.filterwarnings("ignore")

    results = []
    print(f"Total of {len(list_of_dists)} dists to be tested.")
    for _, name in enumerate(list_of_dists):
        # Skip slow ones
        if name in ["levy_stable"]:
            continue
        # print(f"Testing {name} of idx {idx}.")
        try:
            dist = getattr(stats, name)
            param = dist.fit(data)
            a = stats.kstest(data, name, args=param)
            results.append((name, a[0], a[1]))
        except Exception:
            print(f"Failed {name}.")
            continue

    results.sort(key=lambda x: float(x[2]), reverse=True)
    return results


def plot_dist(data, dist_str: str, ax: Axes):
    """
    Plot a scipy distribution with data.
    """
    # Fit again
    dist = getattr(stats, dist_str)
    params = dist.fit(data)

    x = np.linspace(min(data), max(data))
    pdf_fitted = dist.pdf(x, *params)

    sns.histplot(ax=ax, x=data, stat="density")
    ax.plot(x, pdf_fitted, label=dist_str)
    ax.legend()


def best_fit_dist(data, list_of_dists=dist_continous):
    """
    Find best scipy continous distribution and plot it with data.
    """
    # Init plot
    fig, axes = plt.subplots(1, 3, figsize=utils.paper_figsize(0.5))
    test_results = test_all_dists(data=data, list_of_dists=list_of_dists)
    # Only plot three best
    assert len(axes) == len(test_results[0:3])
    for test_result, ax in zip(test_results[0:3], axes):
        plot_dist(data=data, dist_str=test_result[0], ax=ax)

    return fig, axes


def plot_cc_area_bin_pairs(
    cc: str,
    area: str,
    agg_df: pd.DataFrame,
    reference_value_dict: Dict[str, float],
    param: str,
    area_bin_col: str,
    circle_count_col: str,
):
    """
    Plot circle count area bin pair statistical distribution.
    """
    pair_df = agg_df.loc[agg_df["circle_count_binned"] == cc].loc[
        (agg_df[area_bin_col] == area)
    ]
    if pair_df.shape[0] < 100:
        return
    g = sns.displot(data=pair_df, x=param)
    g.ax.axvline(reference_value_dict[param], linestyle="--", color="black")
    g.ax.set_title(f"Basic histplot of {param}")
    fig, axes = best_fit_dist(
        data=pair_df[param],
    )
    return g, fig, axes


def boxplot_fliers(
    data: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Get matplotlib boxplot flier data.

    >>> boxplot_fliers(np.array([1, 2, 4, 56, 56, 1, 324, 431, 23]))
    (array([324, 431]), array([], dtype=int64), 56, 1)
    """
    assert data.ndim == 1
    data_stats = boxplot_stats(data)
    assert len(data_stats) == 1
    bstats = data_stats[0]
    fliers = bstats["fliers"]
    whishi, whislo = bstats["whishi"], bstats["whislo"]
    fliershi, flierslo = fliers[fliers >= whishi], fliers[fliers <= whislo]
    return fliershi, flierslo, whishi, whislo


def plot_group_pair_boxplots(
    group: pd.DataFrame,
    param: str,
    group_col_second: str,
    group_second_labels: Sequence[str],
    multip_diff: float,
    outlier_proportion_threshold: float,
    reference_value_dict: Dict[str, float],
    i: int,
    j: int,
    ax_gen: Generator,
    cc_gen: Generator,
) -> Axes:
    """
    Plot boxplots for group pair.
    """
    # Generate next ax
    ax = next(ax_gen)

    # Plot seaborn boxplot on ax
    ax = sns.boxplot(
        data=group,
        x=group_col_second,
        y=param,
        ax=ax,
        showfliers=False,
        palette="Greys",
    )

    # Enumerate over the second group labels
    for x_loc, label in enumerate(group_second_labels):

        # Locate from current group only ones matching current label
        data = group[param].loc[group[group_col_second] == label]

        # Ignore empty
        if data.shape[0] == 0:
            continue

        # Get boxplot parameters
        fliershi, flierslo, whishi, whislo = boxplot_fliers(data)

        # Iterate over boxplot bottom and top parameters
        for fliers, whis, plus in zip(
            (fliershi, flierslo), (whishi, whislo), (True, False)
        ):

            # Calculate the median
            median = np.median(data)

            # Calculate whisker distance to median
            whis_dist_to_median = abs(median - whis)

            # Calculate y location
            y_loc = (
                median + whis_dist_to_median * multip_diff
                if plus
                else median - whis_dist_to_median * multip_diff
            )

            # Calculate proportion of outliers
            proportion_of_fliers = 100 * (len(fliers) / data.shape[0])

            # Only plot outlier proportion of it fits criteria
            if (
                proportion_of_fliers > outlier_proportion_threshold
                and data.shape[0] > 100
            ):

                # Plot outlier proportion as text
                ax.text(
                    x_loc,
                    y_loc,
                    "{:.1f}%".format(proportion_of_fliers),
                    ha="center",
                    va="center",
                    fontsize=7,
                    fontstyle="italic",
                )

    # Set y label
    ax.set_ylabel(utils.param_renamer(param))

    # Set x label
    ax.set_xlabel(r"Total Area ($10^3\ m^2$)")

    # Set reference value line
    ax.axhline(
        reference_value_dict[utils.param_renamer(param)],
        linestyle="--",
        zorder=1000,
        color="black",
    )

    # not last row
    if i < 3:
        ax.tick_params(axis="x", bottom=False, labelbottom=False)
        ax.set_xlabel("")

    # first row
    if i == 0:
        ax.set_title(next(cc_gen), fontsize="medium")

    # second or later col
    if j > 0:
        ax.set_ylabel("")

    return ax


def grouped_boxplots(
    aggregate_df: pd.DataFrame,
    reference_value_dict: Dict[str, float],
    group_col_first: str,
    group_col_second: str,
    group_first_labels: Sequence[str],
    group_second_labels: Sequence[str],
    multip_diff: float = 1.135,
    outlier_proportion_threshold: float = 1.0,
):
    """
    Plot group-pair boxplots.
    """
    # Initialize figure and axes
    fig: Figure
    fig, axes = plt.subplots(
        4, len(group_first_labels), figsize=utils.paper_figsize(0.85), sharey="row"
    )

    # Set figure title
    # fig.suptitle(figure_title)

    # Axes iterator
    def axes_generator(axes: np.ndarray):
        for ax in axes.flatten():
            yield ax

    # Group label iterator
    def group_label_gen(group_labels: Sequence[str]):
        for group_label in group_labels:
            yield group_label

    # Init iterators
    ax_gen = axes_generator(axes=axes)
    cc_gen = group_label_gen(group_first_labels)

    # Group by group_col_first column
    grouped = aggregate_df.groupby(group_col_first)

    # Enumerate over parameters in reference_value_dict
    for i, param in enumerate(
        [param for param in reference_value_dict if param in aggregate_df.columns]
    ):

        # Enumerate over the groups in grouped
        for j, (_, group) in enumerate(grouped):

            plot_group_pair_boxplots(
                group=group,
                param=param,
                group_col_second=group_col_second,
                group_second_labels=group_second_labels,
                multip_diff=multip_diff,
                outlier_proportion_threshold=outlier_proportion_threshold,
                reference_value_dict=reference_value_dict,
                i=i,
                j=j,
                ax_gen=ax_gen,
                cc_gen=cc_gen,
            )

    # Adjust subplots
    plt.subplots_adjust(hspace=0.1, wspace=0.1)

    # Remove bottom axes spine
    sns.despine(left=False, bottom=True)
    return fig


def circle_count_limiting(
    aggregate_df: pd.DataFrame,
    count_constraints: Sequence[int],
    param: str,
    reference_value_dict: Dict[str, float],
    example=True,
):
    """
    Limit circle count and plot result for param.
    """
    for count_constraint in count_constraints:
        df_constrained = aggregate_df.loc[
            aggregate_df["circle_count"] <= count_constraint
        ]

        g = sns.JointGrid(
            data=df_constrained,
            x="area",
            y=param,
        )
        g.plot(sns.scatterplot, sns.histplot)
        plt.title(f"<= {count_constraint} circles")
        if param in reference_value_dict:
            g.ax_joint.axhline(reference_value_dict[param], linestyle="--")
        if example:
            break


def get_best_value(values: pd.DataFrame, col: str, radius: str):
    """
    Naively detect best estimate from values dataframe.

    >>> df = pd.DataFrame(
    ...             {
    ...                 "x": [10, 15, 123, -4],
    ...                 "radius": [2, 1, 1, -4],
    ...                 "name": ["a", "b", "c", "d"],
    ...             }
    ...         )
    >>> get_best_value(df, "x", "radius")
    10

    """
    best_value = values[col].loc[values[radius].nlargest(1).index]
    if best_value.shape[0] == 1:
        return best_value.iloc[0]
    else:
        raise ValueError()


def normalized_param_plots(
    param: str, dataframe_grouped: DataFrameGroupBy, example: bool = True
):
    """
    Plot radius or area normalized plots of param values.
    """
    param_normalized = f"{param}_norm"
    _, axes = plt.subplots(2, 1, figsize=utils.paper_figsize(0.8))

    target_datas = []
    for target, radius in utils.Utils.circle_names_with_diameter.items():
        if radius != 50:
            continue
        target_data = dataframe_grouped.get_group(target).copy()
        target_data[param_normalized] = target_data[param] / get_best_value(
            target_data, param, "radius"
        )
        target_data["radius_normalized"] = target_data["radius"] / max(
            target_data["radius"]
        )
        target_data["area_normalized"] = target_data["area"] / max(target_data["area"])
        # categorize based on radius
        target_data["radius Cat"] = [
            "full" if rad < max(target_data["radius"]) / 2 else "limited"
            for rad in target_data["radius"].values
        ]
        target_data["radius Cat"] = target_data["radius Cat"].astype("category")
        target_datas.append(target_data)

        # Plotting
        sns.scatterplot(
            data=target_data,
            x="radius_normalized",
            y=param_normalized,
            hue="radius Cat",
            ax=axes[0],
        )
        sns.scatterplot(
            data=target_data,
            x="area_normalized",
            y=param_normalized,
            hue="radius Cat",
            ax=axes[1],
        )
        if example:
            break

    for ax in axes:
        ax.legend().remove()

    target_datas_df = pd.concat(target_datas)
    g = sns.JointGrid(
        data=target_datas_df,
        x="area_normalized",
        y=param_normalized,
    )
    g.plot(sns.scatterplot, sns.histplot)


def radius_constrained_param(
    dataframe_grouped: DataFrameGroupBy, param: str, target: str
):
    """
    Visualize effect of radius.
    """
    target_data = dataframe_grouped.get_group(target).copy()
    target_data["Number of Traces Cat"] = pd.cut(target_data["Number of Traces"], 5)
    target_data["radius Cat"] = [
        "full" if rad < max(target_data["radius"]) / 2 else "limited"
        for rad in target_data["radius"].values
    ]
    target_data["radius Cat"] = target_data["radius Cat"].astype("category")
    sns.lmplot(data=target_data, x="radius", y=param, hue="radius Cat")


def colorgen(colors):
    """
    Infinitely generate colors.
    """
    while True:
        for color in colors:
            yield color


def plot_distribution(
    dist,
    dist_str: str,
    agg_df: pd.DataFrame,
    circle_group: str,
    area_group: str,
    param: str,
    reference_value_dict: Dict[str, float],
    ax,
    legend: bool,
    area_bin_col: str,
    circle_count_col: str,
):
    """
    Plot beta distribution of circle count and total area grouped for param.
    """
    # Locate values with certain circle count and total area group
    pair_df = agg_df.loc[agg_df[circle_count_col] == circle_group].loc[
        (agg_df[area_bin_col] == area_group)
    ]

    # Get parameter values
    values = pair_df[param].values

    # Value interval
    delta = abs(max(values) - min(values))

    # Fit beta distribution
    a, b, loc, scale = dist.fit(values)
    beta_dist = dist(a, b, loc=loc, scale=scale)

    # Determine x value range
    x = np.linspace(min(values), max(values))

    # Calculate y values for xs
    y = beta_dist.pdf(x)

    # Plot x, y values
    sns.lineplot(
        ax=ax,
        x=x,
        y=y,
        color="black",
        label="Beta Distribution PDF Fit" if legend else None,
        legend=legend,
    )

    # Make a color palette
    colors = sns.color_palette("Reds", n_colors=3)

    # Color generator (infinite)
    color_generator = colorgen(colors)

    # Choose the probability thresholds to plot
    probs = list(reversed(np.arange(0.25, 1.0, step=0.25)))

    # Iterate over probabilities
    for interval, prob in zip([beta_dist.interval(prob) for prob in probs], probs):

        # Plot vertical lines at probabilities at interval edges
        prob_color = next(color_generator)
        prob_text = f"{int(prob * 100)} % of iterations."

        # Iterate over the two interval edges
        for xloc, xoff in zip(interval, (-1, 1)):

            interval_text = f"${round(xloc, 2)}$"
            # Plot vertical line at interval edge
            ax.vlines(
                xloc,
                ymin=0,
                ymax=beta_dist.pdf(xloc),
                color=prob_color,
                label=prob_text if xloc != interval[-1] else None,
            )

            # Plot the interval edge value as text
            ax.text(
                x=xloc + (delta * 0.02 * xoff),
                y=beta_dist.pdf(xloc) / 2.25,
                s=interval_text,
                rotation=90,
                ha="center",
                fontstyle="italic",
                va="center",
                fontsize=8,
            )

        # Test hashing the areas
        # fill_xs = np.linspace(*interval)
        # ax.fill_between(fill_xs, y1=beta_dist.pdf(fill_xs), facecolor=None,
        # edgecolor=None, hatch=next(hatch_generator), alpha=0.01)

    # Plot reference value
    ax.axvline(reference_value_dict[param], linestyle="dashed", color="black")

    # Annotate the reference value
    ax.annotate(
        text="Reference value",
        xy=(reference_value_dict[param], max(y) * 1.02),
        xytext=(reference_value_dict[param] - 0.4 * delta, max(y) * 1.03),
        arrowprops={"arrowstyle": "->"},
    )

    # Remove top and right spines
    sns.despine(top=True, right=True)

    # Set x and y labels
    ax.set_ylabel("Probability Density Function (PDF)")
    ax.set_xlabel(param)

    # Plot the background histplot of true values
    sns.histplot(
        ax=ax,
        x=values,
        stat="density",
        alpha=0.1,
        edgecolor=None,
        color="black",
        label=f"{utils.param_renamer(param)} Histogram",
    )

    # Set legend for plot
    if legend:
        ax.legend(edgecolor="black", loc="upper right")
    else:
        ax.legend().remove()

    def dist_param_str(value: float, name: str):
        """
        Make string repr from param value.
        """
        return f"${name} = {round(value, 3)}$"

    # Kolmigoroff-Smirnov test
    kstest_result = stats.kstest(values, dist_str, args=(a, b, loc, scale))
    statistic = kstest_result[0]
    pvalue = kstest_result[1]

    # Collect some distribution parameters into multi-line-string
    vals = (
        a,
        b,
        beta_dist.median(),
        beta_dist.std(),
        beta_dist.var(),
        statistic,
        pvalue,
    )
    names = (
        r"\alpha",
        r"\beta",
        "median",
        "std",
        "var",
        r"KS\ statistic",
        r"KS\ pvalue",
    )
    assert len(vals) == len(names)
    param_text = "Beta Distribution\n"
    for val, name in zip(vals, names):
        param_text += dist_param_str(val, name)
        param_text += "\n" if name != names[-1] else ""

    # Plot the collected text
    ax.text(
        0.1,
        0.25,
        s=param_text,
        ha="center",
        ma="right",
        fontsize=8,
        transform=ax.transAxes,
    )

    # Figure title
    circle_group_text = circle_group.replace("-", " to ")
    ax.set_title(
        f"Subsampling iterations with circle count from {circle_group_text}"
        " and total area between "
        f"{int(area_group[0])*1000}-{int(area_group[2])*1000} $m^2$."
    )

    # Set x scale
    ax.set_xlim(min(values) - 0.25 * delta, 0.9 * max(values) + 0.5 * delta)

    # Set y scale
    ax.set_ylim(0, max(y) * 1.2)

    # Set param name nicely
    ax.set_xlabel(utils.param_renamer(param))


def plot_group_pair_counts(
    agg_df: pd.DataFrame,
    x: str,
    hue: str,
    xlabel: str,
    ylabel: str,
    title: str,
    legend_title: str,
):
    """
    Plot group pair counts.
    """
    fig, ax = plt.subplots(figsize=(7, 3))
    ax = sns.countplot(
        data=agg_df,
        x=x,
        hue=hue,
        ax=ax,
        linewidth=1,
        linestyle="-",
        edgecolor="black",
        palette="Greys",
    )
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    legend = ax.legend()
    legend.set_title(legend_title)

    return fig
