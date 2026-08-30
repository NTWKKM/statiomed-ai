import pandas as pd
import plotly.graph_objects as go
from plotly import subplots


def plot_missing_pattern(
    df: pd.DataFrame, max_cols: int = 50, max_rows: int = 400
) -> go.Figure:
    """
    Create a two-panel visualization of a DataFrame's missingness pattern.

    Selects up to max_cols columns (prioritizing columns with missing values), and builds a top bar chart showing percentage missing per displayed column and a bottom heatmap indicating missing cells (1 = missing, 0 = present). If the DataFrame has more rows than max_rows, the heatmap rows are subsampled using a "Golden Ratio" strategy that prioritizes rows containing missing values while preserving a portion of complete rows for context.

    Parameters:
        df (pd.DataFrame): DataFrame to visualize.
        max_cols (int): Maximum number of columns to display (columns with missing values are prioritized).
        max_rows (int): Maximum number of rows to include in the heatmap; when exceeded, rows are subsampled using the Golden Ratio strategy described above.

    Returns:
        fig (go.Figure): Plotly Figure with two subplots: a top bar chart of percent-missing per displayed column and a bottom heatmap of missingness for the sampled rows.
    """
    if df is None or df.empty:
        return go.Figure()

    # --- 1. Column Selection ---
    # Prioritize columns with missing values
    missing_series = df.isna().sum()
    missing_pct = (missing_series / len(df)) * 100

    cols_with_missing = (
        missing_series[missing_series > 0].sort_values(ascending=False).index.tolist()
    )

    if len(cols_with_missing) > max_cols:
        display_cols = cols_with_missing[:max_cols]
    else:
        remaining = max_cols - len(cols_with_missing)
        cols_complete = missing_series[missing_series == 0].index.tolist()
        display_cols = cols_with_missing + cols_complete[:remaining]

    # Subset data
    df_plot = df[display_cols].copy()
    missing_pct_plot = missing_pct[display_cols]

    # --- 2. Row Subsampling (Golden Ratio Smart Sampling) ---
    original_rows = len(df_plot)
    if original_rows > max_rows:
        # Golden Ratio Target: ~61.8% Missing (Problem), ~38.2% Clean (Context)
        # This ensures the visualization focuses on the "problems" while providing a baseline.
        target_missing_count = int(max_rows * 0.618)  # Approx 247 rows
        target_clean_count = max_rows - target_missing_count  # Approx 153 rows

        # Identify rows
        rows_with_missing = df_plot[df_plot.isna().any(axis=1)]
        rows_complete = df_plot.drop(rows_with_missing.index)

        n_missing = len(rows_with_missing)
        n_complete = len(rows_complete)

        if n_missing <= target_missing_count:
            # Scenario A: Missing count is manageable. Show ALL missing rows.
            # Fill the rest with clean rows.
            rows_missing_sample = rows_with_missing

            n_remaining = max_rows - n_missing
            if n_complete > n_remaining:
                rows_complete_sample = rows_complete.sample(
                    n=n_remaining, random_state=42
                )
            else:
                rows_complete_sample = (
                    rows_complete  # Take all if not enough to fill limit
                )

            y_axis_title = f"Row Index (Sampled {len(rows_missing_sample) + len(rows_complete_sample)}/{original_rows} - Prioritized Missing)"

        else:
            # Scenario B: Too many missing rows. Enforce Golden Ratio.
            # However, if clean rows are scarce, we shouldn't force empty space.
            # We fill strictly with missing if clean is not available.

            if n_complete >= target_clean_count:
                # Default Golden Ratio Split
                rows_missing_sample = rows_with_missing.sample(
                    n=target_missing_count, random_state=42
                )
                rows_complete_sample = rows_complete.sample(
                    n=target_clean_count, random_state=42
                )
            else:
                # Clean Scarcity: Take all clean, fill rest with missing
                rows_complete_sample = rows_complete
                remaining_for_missing = max_rows - n_complete
                rows_missing_sample = rows_with_missing.sample(
                    n=remaining_for_missing, random_state=42
                )

            y_axis_title = (
                f"Row Index (Sampled {max_rows}/{original_rows} - Golden Ratio)"
            )

        df_heatmap = pd.concat([rows_missing_sample, rows_complete_sample]).sort_index()

    else:
        df_heatmap = df_plot
        y_axis_title = "Row Index"

    # --- 3. Build Subplots ---
    fig = subplots.make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.2, 0.8],  # 20% Bar, 80% Heatmap
        specs=[[{"type": "bar"}], [{"type": "heatmap"}]],
    )

    # -- Top: Bar Chart --
    fig.add_trace(
        go.Bar(
            x=display_cols,
            y=missing_pct_plot.values,
            name="% Missing",
            marker_color="#E74856",  # Bootstrap danger red (or close)
            hovertemplate="%{x}: %{y:.1f}% Missing<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # -- Bottom: Heatmap --
    # Map: 0=Present, 1=Missing
    z = df_heatmap.isna().astype(int).transpose().values
    x = df_heatmap.index
    y = display_cols

    # Colors: 0 (Present) = Light Grey, 1 (Missing) = Red
    colorscale = [[0.0, "#f8f9fa"], [1.0, "#E74856"]]

    fig.add_trace(
        go.Heatmap(
            z=z,
            x=x,
            y=y,
            colorscale=colorscale,
            showscale=False,
            hovertemplate="Row: %{x}<br>Var: %{y}<br>Status: %{z}<extra></extra>",
        ),
        row=2,
        col=1,
    )

    # --- 4. Layout ---
    fig.update_layout(
        title="<b>Missing Data Pattern</b>",
        plot_bgcolor="white",
        margin=dict(l=10, r=10, t=50, b=40),
        height=min(900, max(500, len(display_cols) * 20 + 200)),
        showlegend=False,
    )

    # Bar chart axis
    fig.update_yaxes(title_text="% Missing", autorange=True, row=1, col=1)

    # Heatmap axis
    fig.update_xaxes(title_text=y_axis_title, row=2, col=1)
    fig.update_yaxes(autorange="reversed", row=2, col=1)  # Top variable at top

    return fig
