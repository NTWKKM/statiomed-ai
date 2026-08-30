from shiny import ui


def create_input_group(title, *inputs, type="required"):
    """
    Create a styled input group with a title and properly styled container.

    Args:
        title (str): The title of the input group.
        *inputs (ui.Tag): Input elements to include in the group.
        type (str): Type of group 'required', 'optional', or 'advanced'.

    Returns:
        ui.Tag: A styled div containing the inputs.
    """
    # Determine class based on type
    if type == "required":
        header_class = "form-section-title form-section-required"
        container_class = "form-section"
    elif type == "optional":
        header_class = "form-section-title form-section-optional"
        container_class = "form-section"
    elif type == "advanced":
        # Special handling for advanced options (accordion style)
        return ui.div(
            ui.tags.details(
                ui.tags.summary("⚙️ " + title),
                *inputs,
            ),
            class_="form-section form-section-advanced",
        )
    else:
        header_class = "form-section-title"
        container_class = "form-section"

    return ui.div(ui.h4(title, class_=header_class), *inputs, class_=container_class)


def create_tooltip_label(label, tooltip_text, id=None):
    """
    Create a label with a question mark icon that shows a tooltip on hover.

    Args:
        label (str): The label text.
        tooltip_text (str): The text to show in the tooltip.
        id (str, optional): ID for the label container.

    Returns:
        ui.Tag: A styled label with tooltip.
    """
    # Removed "?" icon as requested by user. Tooltip is attached to the label container.
    return ui.div(label, class_="label-with-help", title=tooltip_text, id=id)


def create_input_with_help(input_tag, help_text):
    """
    Wrap an input with a help text block below it.

    Args:
        input_tag (ui.Tag): The input element.
        help_text (str): The help text to display.

    Returns:
        ui.Tag: A div containing the input and help text.
    """
    return ui.div(input_tag, ui.tags.div(help_text, class_="input-help-text"))


def create_workflow_indicator(steps, current_step_index):
    """
    Create a horizontal workflow progress indicator.

    Args:
        steps (list[str]): List of step names.
        current_step_index (int): 0-based index of the current active step.

    Returns:
        ui.Tag: A styled workflow indicator.
    """
    items = []
    for i, step in enumerate(steps):
        # Step box
        step_class = "step active" if i <= current_step_index else "step"
        items.append(ui.div(f"{i + 1}. {step}", class_=step_class))

        # Arrow (if not last item)
        if i < len(steps) - 1:
            items.append(ui.div("→", class_="step-divider"))

    return ui.div(*items, class_="workflow-progress")


def create_results_container(title, *content, class_=""):
    """
    Create a standardized container for analysis results.

    Args:
        title (str): Title of the results section.
        *content (ui.Tag): Content elements (tables, plots, etc.).
        class_ (str): Additional CSS classes.

    Returns:
        ui.Tag: A styled results container.
    """
    return ui.div(
        ui.div(ui.h3(title, class_="results-title"), class_="results-header"),
        ui.hr(class_="results-divider"),
        *content,
        class_=f"results-section {class_}".strip(),
    )


def create_loading_state(message="Processing data..."):
    """
    Create an enhanced loading spinner state using CSS animation.

    Args:
        message (str): Message to display below spinner.

    Returns:
        ui.Tag: Styled loading state with CSS spinner.
    """
    return ui.div(
        ui.div(class_="spinner"),
        ui.h5(message, class_="text-secondary mt-3"),
        class_="loading-state",
    )


def create_placeholder_state(
    message="Click 'Run Analysis' to see results here", icon="📊"
):
    """
    Create an empty state placeholder.

    Args:
        message (str): Message to display.
        icon (str): Icon or emoji to display.

    Returns:
        ui.Tag: Styled placeholder.
    """
    return ui.div(
        ui.div(icon, class_="placeholder-icon"),
        ui.p(message, class_="text-muted"),
        class_="placeholder-state",
    )


def create_error_alert(message, title="Error"):
    """
    Create a styled error alert card.

    Args:
        message (str): Error message text.
        title (str): Alert title.

    Returns:
        ui.Tag: Styled error alert.
    """
    return ui.div(
        ui.h5(title, class_="text-danger"),
        ui.p(message),
        class_="alert alert-danger error-alert-card",
    )


def format_stat_table_html(df):
    """
    Format a pandas DataFrame as an HTML table with statistical highlighting.

    Args:
        df (pd.DataFrame): DataFrame to format.

    Returns:
        str: HTML string of the table.
    """
    # Create copy to avoid modifying original
    df_display = df.copy()

    # Format p-values: highlight < 0.05
    if "p_value" in df_display.columns:
        df_display["p_value"] = df_display["p_value"].apply(
            lambda x: (
                f'<span class="sig-p">{x:.4f}</span>'
                if isinstance(x, (int, float)) and x < 0.05
                else f"{x:.4f}"
                if isinstance(x, (int, float))
                else x
            )
        )

    # Format other numeric columns to 2 decimals if they are float
    for col in df_display.select_dtypes(include=["float64", "float32"]).columns:
        if col != "p_value":
            df_display[col] = df_display[col].apply(
                lambda x: f"{x:.3f}" if isinstance(x, (int, float)) else x
            )

    return df_display.to_html(
        classes="table table-hover stat-table", escape=False, index=False
    )


def create_empty_state_ui(
    message="No data available",
    sub_message="Please upload a dataset or run an analysis to view results.",
    icon="📊",
    action_button=None,
):
    """
    Create a standardized empty state component (Audit Section 7.1).

    Args:
        message (str): Main heading message.
        sub_message (str): Subtitle / instruction message.
        icon (str): Icon character to display.
        action_button (ui.Tag): Optional button to include (e.g. "Upload File").

    Returns:
        ui.Tag: A styled empty state div.
    """
    content = [
        ui.div(icon, class_="empty-state-icon"),
        ui.h3(message),
        ui.p(sub_message),
    ]

    if action_button:
        content.append(action_button)

    return ui.div(
        ui.div(*content, class_="empty-state-content"),
        class_="empty-state",
    )


def create_skeleton_loader_ui(rows=3, show_chart=False):
    """
    Create a skeleton loading animation (Audit Section 7.2).

    Args:
        rows (int): Number of text rows to simulate.
        show_chart (bool): Whether to include a chart placeholder.

    Returns:
        ui.Tag: A styled skeleton loader container.
    """
    items = []

    # Text skeletons
    for i in range(rows):
        width_class = "skeleton-text" if i % 2 == 0 else "skeleton-text-sm"
        items.append(ui.div(class_=f"skeleton {width_class}"))

    # Chart skeleton
    if show_chart:
        items.append(ui.div(class_="skeleton skeleton-chart"))

    return ui.div(*items, class_="skeleton-container")


def create_download_status_badge(ready: bool) -> "ui.Tag":
    """
    Create a compact status badge indicating download readiness.

    Args:
        ready: Whether analysis results are available for download.

    Returns:
        ui.Tag: A styled badge with the appropriate icon and text.
    """
    if ready:
        return ui.div("✅ Ready to download", class_="download-status-badge ready")
    return ui.div("⏳ Run analysis first", class_="download-status-badge pending")
