"""Plotting functions for posggym.agents analysis."""
from itertools import product
from typing import List, Optional, Tuple

import numpy as np

try:
    import matplotlib
    import matplotlib.pyplot as plt
except ImportError as e:
    raise ImportError(
        "Missing dependency for plotting functionality. Run `pip install matplotlib` "
        "to install required dependency."
    ) from e


try:
    import pandas as pd
except ImportError as e:
    raise ImportError(
        "Missing dependency for plotting functionality. Run `pip install pandas` "
        "to install required dependency."
    ) from e


def add_95CI(df: pd.DataFrame) -> pd.DataFrame:
    """Add 95% CI columns to dataframe."""

    def conf_int(row, prefix):
        std = row[f"{prefix}_std"]
        n = row["num_episodes"]
        return 1.96 * (std / np.sqrt(n))

    prefix = ""
    for col in df.columns:
        if not col.endswith("_std"):
            continue
        prefix = col.replace("_std", "")
        df[f"{prefix}_CI"] = df.apply(lambda row: conf_int(row, prefix), axis=1)
    return df


def add_outcome_proportions(df: pd.DataFrame) -> pd.DataFrame:
    """Add proportion columns to dataframe."""

    def prop(row, col_name):
        n = row["num_episodes"]
        total = row[col_name]
        return total / n

    columns = ["num_LOSS", "num_DRAW", "num_WIN", "num_NA"]
    new_column_names = ["prop_LOSS", "prop_DRAW", "prop_WIN", "prop_NA"]
    for col_name, new_name in zip(columns, new_column_names):
        if col_name in df.columns:
            df[new_name] = df.apply(lambda row: prop(row, col_name), axis=1)
    return df


def get_policy_type_and_seed(policy_name: str) -> Tuple[str, str]:
    """Get policy type and seed from policy name."""
    if "seed" not in policy_name:
        return policy_name, "None"

    # policy_name = "policy_type_seed[seed]"
    tokens = policy_name.split("_")
    policy_type = []
    seed_token = None
    for t in tokens:
        if t.startswith("seed"):
            seed_token = t
            break
        policy_type.append(t)

    policy_type = "_".join(policy_type)
    seed = seed_token.replace("seed", "")
    return policy_type, seed


def add_policy_type_and_seed(df: pd.DataFrame) -> pd.DataFrame:
    """Add policy type and seed columns to dataframe."""
    df["policy_type"] = df.apply(
        lambda row: get_policy_type_and_seed(row["policy_name"])[0], axis=1
    )
    df["policy_seed"] = df.apply(
        lambda row: get_policy_type_and_seed(row["policy_name"])[1], axis=1
    )
    return df


def add_co_team_name(df: pd.DataFrame) -> pd.DataFrame:
    """Add co team name to dataframe.

    Also removes unwanted rows.
    """
    # For each policy we want to group rows where that policy is paired with equivalent
    # co-player policies.
    env_symmetric = df["symmetric"].unique().tolist()[0]
    if env_symmetric:
        # For symmetric environments we group rows where the policy is paired with the
        # same co-player policies, independent of the ordering
        same_co_team_ids = set()
        for team_id in df["co_team_id"].unique().tolist():
            # ignore ( and ) and start and end
            pi_names = team_id[1:-1].split(",")
            if all(name == pi_names[0] for name in pi_names):
                same_co_team_ids.add(team_id)

        df = df[df["co_team_id"].isin(same_co_team_ids)]

        def get_team_name(row):
            team_id = row["co_team_id"]
            return team_id[1:-1].split(",")[0]

        df["co_team_name"] = df.apply(get_team_name, axis=1)
    else:
        # for asymmetric environments ordering matters so can't reduce team IDs
        def get_team_name_asymmetric(row):
            team_id = row["co_team_id"]
            return team_id[1:-1]

        df["co_team_name"] = df.apply(get_team_name_asymmetric, axis=1)

    def get_team_type(row):
        pi_names = row["co_team_name"].split(",")
        pi_types = [get_policy_type_and_seed(pi_name)[0] for pi_name in pi_names]
        if all(pi_type == pi_types[0] for pi_type in pi_types):
            return pi_types[0]
        return ",".join(pi_types)

    def get_team_seed(row):
        pi_names = row["co_team_name"].split(",")
        pi_seeds = [get_policy_type_and_seed(pi_name)[1] for pi_name in pi_names]
        if all(pi_seed == pi_seeds[0] for pi_seed in pi_seeds):
            return pi_seeds[0]
        return ",".join(pi_seeds)

    df["co_team_type"] = df.apply(get_team_type, axis=1)
    df["co_team_seed"] = df.apply(get_team_seed, axis=1)
    return df


def import_results(
    result_file: str,
) -> pd.DataFrame:
    """Import experiment results."""
    # disable annoying warning
    pd.options.mode.chained_assignment = None
    df = pd.read_csv(result_file)

    df = add_95CI(df)
    df = add_outcome_proportions(df)
    df = add_policy_type_and_seed(df)
    df = add_co_team_name(df)

    # enable annoyin warning
    pd.options.mode.chained_assignment = "warn"
    return df


def heatmap(
    data,
    row_labels,
    col_labels,
    ax=None,
    show_cbar=True,
    cbar_kw={},
    cbarlabel="",
    **kwargs,
):
    """Create a heatmap from a numpy array and two lists of labels.

    ref:
    matplotlib.org/stable/gallery/images_contours_and_fields/
    image_annotated_heatmap.html

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    show_cbar
        If true a color bard is displayed, otherwise no colorbar is shown.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.

    """
    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    if show_cbar:
        cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
        cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
    else:
        cbar = None

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right", rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="w", linestyle="-", linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(
    im,
    data=None,
    valfmt="{x:.2f}",
    textcolors=("black", "white"),
    threshold=None,
    **textkw,
):
    """Annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.

    """
    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max()) / 2.0

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = {"horizontalalignment": "center", "verticalalignment": "center"}
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


def plot_pairwise_heatmap(
    ax,
    labels: Tuple[List[str], List[str]],
    values: np.ndarray,
    title: Optional[str] = None,
    vrange: Optional[Tuple[float, float]] = None,
    valfmt: Optional[str] = None,
):
    """Plot pairwise values as a heatmap."""
    # Note numpy arrays by default have (0, 0) in the top-left corner.
    # While matplotlib images are displayed with (0, 0) being the bottom-left
    # corner.
    # To get images looking correct we need to reverse the rows of the array
    # And also the row labels
    values = values[-1::-1]

    if vrange is None:
        vrange = (np.nanmin(values), np.nanmax(values))

    if valfmt is None:
        valfmt = "{x:.2f}"

    im, cbar = heatmap(
        data=values,
        row_labels=reversed(labels[0]),
        col_labels=labels[1],
        ax=ax,
        show_cbar=False,
        cmap="viridis",
        vmin=vrange[0],
        vmax=vrange[1],
    )

    annotate_heatmap(
        im,
        valfmt=valfmt,
        textcolors=("white", "black"),
        threshold=vrange[0] + (0.2 * (vrange[1] - vrange[0])),
    )

    if title:
        ax.set_title(title)


def get_pairwise_values(
    plot_df,
    y_key: str,
    policy_key: str = "policy_id",
    coplayer_policy_key: str = "coplayer_policy_id",
    coplayer_policies: Optional[List[str]] = None,
):
    """Get values for each policy pairing."""
    policies = plot_df[policy_key].unique().tolist()
    policies.sort()

    if coplayer_policies is None:
        coplayer_policies = plot_df[coplayer_policy_key].unique().tolist()
        coplayer_policies.sort()

    agent_ids = plot_df["agent_id"].unique()
    agent_ids.sort()

    plot_df = plot_df[plot_df[coplayer_policy_key].isin(coplayer_policies)]
    gb = plot_df.groupby([policy_key, coplayer_policy_key])
    pw_values = np.full((len(policies), len(coplayer_policies)), np.nan)
    for name, group in gb:
        (row_policy, col_policy) = name
        row_policy_idx = policies.index(row_policy)
        col_policy_idx = coplayer_policies.index(col_policy)

        pw_values[row_policy_idx][col_policy_idx] = group.mean(numeric_only=True)[y_key]

    return pw_values, (policies, coplayer_policies)


def plot_pairwise_comparison(
    plot_df,
    y_key: str,
    policy_key: str = "policy_id",
    coplayer_policy_key: str = "coplayer_policy_id",
    y_err_key: Optional[str] = None,
    vrange=None,
    figsize=(20, 20),
    valfmt=None,
    coplayer_policies: Optional[List[str]] = None,
):
    """Plot results for each policy pairings.

    This produces a policy X policy grid-plot

    If `y_err_key` is provided then an additional policy X policy grid-plot is
    produced displaying the err values.

    """
    if valfmt is None:
        valfmt = "{x:.2f}"

    ncols = 2 if y_err_key else 1
    fig, axs = plt.subplots(
        nrows=1,
        ncols=ncols,
        # figsize=figsize,
        squeeze=False,
        sharey=True,
    )

    pw_values, (row_policies, col_policies) = get_pairwise_values(
        plot_df,
        y_key,
        policy_key=policy_key,
        coplayer_policy_key=coplayer_policy_key,
        coplayer_policies=coplayer_policies,
    )

    plot_pairwise_heatmap(
        axs[0][0],
        (row_policies, col_policies),
        pw_values,
        title=None,
        vrange=vrange,
        valfmt=valfmt,
    )

    if y_err_key:
        pw_err_values, _ = get_pairwise_values(
            plot_df,
            y_err_key,
            policy_key=policy_key,
            coplayer_policy_key=coplayer_policy_key,
            coplayer_policies=coplayer_policies,
        )

        plot_pairwise_heatmap(
            axs[0][1],
            (row_policies, col_policies),
            pw_err_values,
            title=None,
            vrange=None,
            valfmt=valfmt,
        )
        fig.tight_layout()

    return fig, axs


def plot_pairwise_population_comparison(
    plot_df,
    y_key: str,
    pop_key: str,
    policy_key: str,
    coplayer_pop_key: str,
    coplayer_policy_key: str,
    vrange=None,
    figsize=(20, 20),
    valfmt=None,
):
    """Plot results for each policy-seed pairings.

    This produces a grid of (grid)-plots:

    Outer-grid: pop X pop
    Inner-grid: policy X policy

    """
    pop_ids = plot_df[pop_key].unique().tolist()
    pop_ids.sort()
    co_pop_ids = plot_df[coplayer_pop_key].unique().tolist()
    co_pop_ids.sort()

    policies = plot_df[policy_key].unique().tolist()
    policies.sort()
    co_policies = plot_df[coplayer_policy_key].unique().tolist()
    co_policies.sort()

    fig, axs = plt.subplots(nrows=len(pop_ids), ncols=len(co_pop_ids), figsize=figsize)

    for row_pop, col_pop in product(pop_ids, co_pop_ids):
        row_pop_idx = pop_ids.index(row_pop)
        col_pop_idx = co_pop_ids.index(col_pop)

        pw_values = np.zeros((len(policies), len(policies)))
        for row_policy, col_policy in product(policies, co_policies):
            row_policy_idx = policies.index(row_policy)
            col_policy_idx = co_policies.index(col_policy)

            y = plot_df[
                (plot_df[policy_key] == row_policy)
                & (plot_df[pop_key] == row_pop)
                & (plot_df[coplayer_policy_key] == col_policy)
                & (plot_df[coplayer_pop_key] == col_pop)
            ][y_key].mean()

            if y is not np.nan and valfmt is None:
                if isinstance(y, float):
                    valfmt = "{x:.2f}"
                if isinstance(y, int):
                    valfmt = "{x}"

            pw_values[row_policy_idx][col_policy_idx] = y

        ax = axs[row_pop_idx][col_pop_idx]
        plot_pairwise_heatmap(
            ax,
            (policies, co_policies),
            pw_values,
            title=None,
            vrange=vrange,
            valfmt=valfmt,
        )

        if row_pop_idx == 0:
            ax.set_title(col_pop)
        if col_pop_idx == 0:
            ax.set_ylabel(row_pop)

    fig.tight_layout()
    return fig, axs


def get_all_mean_pairwise_values(
    plot_df,
    y_key: str,
    policy_key: str,
    pop_key: str,
    coplayer_policy_key: str,
    coplayer_pop_key: str,
):
    """Get mean pairwise values for all policies."""
    policies = plot_df[policy_key].unique().tolist()
    policies.sort()
    co_policies = plot_df[coplayer_policy_key].unique().tolist()
    co_policies.sort()
    pop_ids = plot_df[pop_key].unique().tolist()
    pop_ids.sort()
    co_pop_ids = plot_df[coplayer_pop_key].unique().tolist()
    co_pop_ids.sort()

    xp_pw_returns = np.zeros((len(policies), len(co_policies)))
    sp_pw_returns = np.zeros((len(policies), len(co_policies)))

    for row_policy, col_policy in product(policies, co_policies):
        row_policy_idx = policies.index(row_policy)
        col_policy_idx = co_policies.index(col_policy)

        sp_values = []
        xp_values = []

        for row_pop, col_pop in product(pop_ids, co_pop_ids):
            ys = plot_df[
                (plot_df[policy_key] == row_policy)
                & (plot_df[pop_key] == row_pop)
                & (plot_df[coplayer_policy_key] == col_policy)
                & (plot_df[coplayer_pop_key] == col_pop)
            ][y_key]
            y = ys.mean()

            if row_pop == col_pop or (
                isinstance(col_pop, tuple) and all(row_pop == p for p in col_pop)
            ):
                sp_values.append(y)
            else:
                xp_values.append(y)

        sp_pw_returns[row_policy_idx][col_policy_idx] = np.nanmean(sp_values)
        xp_pw_returns[row_policy_idx][col_policy_idx] = np.nanmean(xp_values)

    return (policies, co_policies), sp_pw_returns, xp_pw_returns


def plot_mean_pairwise_comparison(
    plot_df,
    y_key: str,
    policy_key: str,
    pop_key: str,
    coplayer_policy_key: str,
    coplayer_pop_key: str,
    vrange: Optional[Tuple[float, float]] = None,
    figsize=(12, 6),
    valfmt=None,
):
    """Plot mean pairwise comparison of policies for given y variable."""
    policy_ids, sp_values, xp_values = get_all_mean_pairwise_values(
        plot_df,
        y_key,
        policy_key=policy_key,
        pop_key=pop_key,
        coplayer_policy_key=coplayer_policy_key,
        coplayer_pop_key=coplayer_pop_key,
    )

    if vrange is None:
        min_value = np.nanmin([np.nanmin(sp_values), np.nanmin(xp_values)])
        max_value = np.nanmax([np.nanmax(sp_values), np.nanmax(xp_values)])
        vrange = (min_value, max_value)

    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=figsize)
    plot_pairwise_heatmap(
        axs[0], policy_ids, sp_values, title="Same-Play", vrange=vrange, valfmt=valfmt
    )
    plot_pairwise_heatmap(
        axs[1], policy_ids, xp_values, title="Cross-Play", vrange=vrange, valfmt=valfmt
    )

    pw_diff = sp_values - xp_values
    plot_pairwise_heatmap(
        axs[2],
        policy_ids,
        pw_diff,
        title="Difference",
        vrange=(np.nanmin(pw_diff), np.nanmax(pw_diff)),
        valfmt=valfmt,
    )

    fig.suptitle(y_key)
    fig.tight_layout()
    return fig, axs
    return fig, axs
