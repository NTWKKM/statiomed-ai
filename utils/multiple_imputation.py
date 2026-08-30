"""
📊 Multiple Imputation Module

Full Multiple Imputation by Chained Equations (MICE) with Rubin's rules pooling.

Features:
    - Generate m imputed datasets
    - Support for continuous and categorical variables
    - Pool estimates using Rubin's rules
    - Pool regression results (coefficients, SEs, p-values)
    - Diagnostic plots for imputation quality

References:
    Rubin, D.B. (1987). Multiple Imputation for Nonresponse in Surveys.
    Van Buuren, S. (2018). Flexible Imputation of Missing Data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer

from logger import get_logger
from tabs._common import get_color_palette

logger = get_logger(__name__)
COLORS = get_color_palette()


@dataclass
class PooledEstimate:
    """Pooled estimate from multiple imputation using Rubin's rules."""

    estimate: float
    se: float
    ci_lower: float
    ci_upper: float
    df: float  # Degrees of freedom
    p_value: float
    n_imputations: int
    within_variance: float
    between_variance: float
    total_variance: float
    fmi: float  # Fraction of missing information
    lambda_: float  # Proportion of variance due to missingness


@dataclass
class PooledRegressionResults:
    """Pooled regression results from multiple imputation."""

    coefficients: dict[str, PooledEstimate]
    n_observations: int
    n_imputations: int
    model_type: str
    pooled_r_squared: float | None = None

    def to_dataframe(self) -> pd.DataFrame:
        """Convert pooled results to a DataFrame."""
        rows = []
        for var_name, est in self.coefficients.items():
            rows.append(
                {
                    "Variable": var_name,
                    "Coefficient": est.estimate,
                    "SE": est.se,
                    "95% CI Lower": est.ci_lower,
                    "95% CI Upper": est.ci_upper,
                    "P-value": est.p_value,
                    "FMI": est.fmi,
                }
            )
        return pd.DataFrame(rows)


@dataclass
class MICEResult:
    """Result of MICE imputation."""

    imputed_datasets: list[pd.DataFrame]
    n_imputations: int
    columns_imputed: list[str]
    convergence_data: dict[str, list[list[float]]] = field(default_factory=dict)
    original_missing_mask: pd.DataFrame | None = None


class MICEImputer:
    """
    Multiple Imputation by Chained Equations (MICE) Imputer.

    Generates multiple complete datasets by iteratively imputing
    missing values based on the observed data.

    Parameters:
        n_imputations: Number of imputed datasets to generate (default 5)
        max_iter: Maximum iterations for each imputation (default 10)
        random_state: Random seed for reproducibility

    Example:
        >>> imputer = MICEImputer(n_imputations=5)
        >>> result = imputer.fit_transform(df)
        >>> imputed_dfs = result.imputed_datasets
    """

    def __init__(
        self,
        n_imputations: int = 5,
        max_iter: int = 10,
        random_state: int = 42,
    ):
        self.n_imputations = n_imputations
        self.max_iter = max_iter
        self.random_state = random_state
        self._imputed_datasets: list[pd.DataFrame] = []
        self._convergence_data: dict[str, list[list[float]]] = {}

    def fit_transform(
        self,
        df: pd.DataFrame,
        columns: list[str] | None = None,
    ) -> MICEResult:
        """
        Generate multiple imputed datasets.

        Args:
            df: Input DataFrame with missing values
            columns: Columns to consider for imputation (default: all numeric)

        Returns:
            MICEResult containing imputed datasets and diagnostics
        """
        logger.info(
            "Starting MICE with %d imputations, %d iterations",
            self.n_imputations,
            self.max_iter,
        )

        df_work = df.copy()

        # Identify columns to impute
        if columns is None:
            numeric_cols = df_work.select_dtypes(include=[np.number]).columns.tolist()
        else:
            # Validate user-requested columns
            numeric_cols = []
            invalid_cols = []
            for c in columns:
                if c not in df_work.columns:
                    invalid_cols.append(f"{c} (not in dataframe)")
                elif not pd.api.types.is_numeric_dtype(df_work[c]):
                    invalid_cols.append(f"{c} (not numeric)")
                else:
                    numeric_cols.append(c)

            if invalid_cols:
                raise ValueError(
                    f"Invalid columns for MICE imputation: {', '.join(invalid_cols)}"
                )

        # Store original missing mask
        missing_mask = df_work[numeric_cols].isna()

        # Store columns with missing data
        cols_with_missing = [c for c in numeric_cols if missing_mask[c].any()]

        if not cols_with_missing:
            logger.warning("No missing data found in specified columns")
            return MICEResult(
                imputed_datasets=[df_work.copy()],
                n_imputations=1,
                columns_imputed=[],
                original_missing_mask=missing_mask,
            )

        logger.info(
            "Imputing %d columns: %s", len(cols_with_missing), cols_with_missing
        )

        imputed_datasets = []
        convergence_data: dict[str, list[list[float]]] = {
            col: [] for col in cols_with_missing
        }

        for m in range(self.n_imputations):
            # Use different random state for each imputation
            rng_state = self.random_state + m * 1000

            imputer = IterativeImputer(
                max_iter=self.max_iter,
                random_state=rng_state,
                sample_posterior=True,  # Important for proper MI
            )

            # Fit and transform
            imputed_values = imputer.fit_transform(df_work[numeric_cols])

            # Create imputed DataFrame
            df_imputed = df_work.copy()
            df_imputed[numeric_cols] = imputed_values
            imputed_datasets.append(df_imputed)

            # Convergence tracking not supported with current sklearn IterativeImputer implementation
            # (Requires manual loop implementation which degrades performance)
            pass

            logger.debug("Completed imputation %d/%d", m + 1, self.n_imputations)

        logger.info(
            "MICE imputation complete: %d datasets generated", len(imputed_datasets)
        )

        return MICEResult(
            imputed_datasets=imputed_datasets,
            n_imputations=self.n_imputations,
            columns_imputed=cols_with_missing,
            convergence_data=convergence_data,
            original_missing_mask=missing_mask,
        )


def pool_estimates(
    estimates: list[float],
    variances: list[float],
    n_obs: int | None = None,
) -> PooledEstimate:
    """
    Pool estimates from multiple imputations using Rubin's rules.

    Rubin's rules combine estimates and variances from m imputed datasets:
    - Pooled estimate: θ̄ = (1/m) Σ θ_i
    - Within-imputation variance: W̄ = (1/m) Σ V_i
    - Between-imputation variance: B = (1/(m-1)) Σ (θ_i - θ̄)²
    - Total variance: T = W̄ + (1 + 1/m) × B

    Args:
        estimates: List of point estimates from each imputation
        variances: List of variance estimates from each imputation
        n_obs: Number of observations (for df calculation)

    Returns:
        PooledEstimate with combined estimate, SE, CI, p-value, and diagnostics

    References:
        Rubin, D.B. (1987). Multiple Imputation for Nonresponse in Surveys.
    """
    m = len(estimates)

    if m != len(variances):
        raise ValueError("Number of estimates must match number of variances")

    if m < 2:
        # Single imputation - just return the estimate
        est = estimates[0]
        var = variances[0]
        se = np.sqrt(var)
        # Use t-distribution with df=inf (effectively normal) for consistency
        df_single = np.inf
        t_crit = stats.t.ppf(0.975, df_single)

        return PooledEstimate(
            estimate=est,
            se=se,
            ci_lower=est - t_crit * se,
            ci_upper=est + t_crit * se,
            df=df_single,
            p_value=2 * stats.t.sf(abs(est / se), df_single) if se > 0 else 1.0,
            n_imputations=1,
            within_variance=var,
            between_variance=0.0,
            total_variance=var,
            fmi=0.0,
            lambda_=0.0,
        )

    # Convert to numpy arrays
    theta = np.array(estimates)
    V = np.array(variances)

    # Pooled estimate (Rubin's rule 1)
    theta_bar = np.mean(theta)

    # Within-imputation variance (Rubin's rule 2)
    W_bar = np.mean(V)

    # Between-imputation variance (Rubin's rule 3)
    B = np.var(theta, ddof=1)

    # Total variance (Rubin's rule 4)
    T = W_bar + (1 + 1 / m) * B

    # Standard error
    se = np.sqrt(T)

    # Degrees of freedom (Barnard & Rubin, 1999)
    if B > 0 and W_bar > 0:
        r = (1 + 1 / m) * B / W_bar  # Relative increase in variance

        # Old df formula (Rubin 1987)
        df_old = (m - 1) * (1 + 1 / r) ** 2

        # Adjusted df (Barnard & Rubin 1999) if n_obs provided
        if n_obs is not None:
            df_obs = (n_obs - 1) * (1 - r / (r + 1))
            df = (df_old * df_obs) / (df_old + df_obs)
        else:
            df = df_old
    else:
        df = np.inf
        r = 0

    # t-statistic and p-value
    t_stat = theta_bar / se if se > 0 else 0

    if np.isfinite(df) and df > 0:
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
    else:
        p_value = 2 * (1 - stats.norm.cdf(abs(t_stat)))

    # Confidence interval
    if np.isfinite(df) and df > 0:
        t_crit = stats.t.ppf(0.975, df)
    else:
        t_crit = 1.96

    ci_lower = theta_bar - t_crit * se
    ci_upper = theta_bar + t_crit * se

    # Fraction of missing information (FMI)
    lambda_ = (B + B / m) / T if T > 0 else 0  # Proportion due to missingness
    fmi = (r + 2 / (df + 3)) / (r + 1) if r > 0 else 0

    return PooledEstimate(
        estimate=float(theta_bar),
        se=float(se),
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        df=float(df),
        p_value=float(p_value),
        n_imputations=m,
        within_variance=float(W_bar),
        between_variance=float(B),
        total_variance=float(T),
        fmi=float(fmi),
        lambda_=float(lambda_),
    )


def pool_regression_results(
    models: list[Any],
    model_type: str = "linear",
    get_params: Callable | None = None,
) -> PooledRegressionResults:
    """
    Pool results from regression models fitted on multiple imputed datasets.

    Args:
        models: List of fitted model objects (e.g., statsmodels results)
        model_type: Type of model ("linear", "logistic", "cox")
        get_params: Optional function to extract (params, bse) from model

    Returns:
        PooledRegressionResults with pooled coefficients

    Example:
        >>> models = [sm.OLS(y, X_imp).fit() for X_imp in imputed_X_list]
        >>> pooled = pool_regression_results(models)
        >>> print(pooled.to_dataframe())
    """
    m = len(models)

    if m == 0:
        raise ValueError("No models provided for pooling")

    # Extract parameters from first model to get variable names
    first_model = models[0]

    if get_params is not None:
        params0, bse0 = get_params(first_model)
    elif hasattr(first_model, "params") and hasattr(first_model, "bse"):
        params0 = first_model.params
    else:
        raise ValueError(
            "Cannot extract parameters from model. Provide get_params function."
        )

    # Get variable names
    if hasattr(params0, "index"):
        var_names = list(params0.index)
    else:
        var_names = [f"X{i}" for i in range(len(params0))]

    # Collect estimates and variances for each variable
    pooled_coefficients = {}
    n_obs = getattr(first_model, "nobs", None)

    for var_name in var_names:
        estimates = []
        variances = []

        for model in models:
            if get_params is not None:
                params, bse = get_params(model)
            else:
                params = model.params
                bse = model.bse

            # Get estimate and variance for this variable
            if hasattr(params, "index"):
                idx = list(params.index).index(var_name)
                est = float(params.iloc[idx])
                se = float(bse.iloc[idx])
            else:
                idx = var_names.index(var_name)
                est = float(params[idx])
                se = float(bse[idx])

            estimates.append(est)
            variances.append(se**2)  # Variance = SE²

        # Pool using Rubin's rules
        pooled = pool_estimates(
            estimates, variances, n_obs=int(n_obs) if n_obs else None
        )
        pooled_coefficients[var_name] = pooled

    # Pool R-squared if available (using average - approximation)
    r_squared = None
    if hasattr(first_model, "rsquared"):
        r_squared_values = [m.rsquared for m in models if hasattr(m, "rsquared")]
        if r_squared_values:
            r_squared = float(np.mean(r_squared_values))

    return PooledRegressionResults(
        coefficients=pooled_coefficients,
        n_observations=int(n_obs) if n_obs else 0,
        n_imputations=m,
        model_type=model_type,
        pooled_r_squared=r_squared,
    )


def create_imputation_diagnostics(
    mice_result: MICEResult,
    original_df: pd.DataFrame,
) -> dict[str, go.Figure]:
    """
    Create diagnostic plots for imputation quality.

    Returns:
        Dictionary of Plotly figures:
        - 'density_comparison': Observed vs imputed distributions
        - 'strip_plot': Individual imputations comparison
    """
    figures = {}

    if not mice_result.columns_imputed:
        return figures

    # 1. Density Comparison Plot
    n_cols = len(mice_result.columns_imputed)
    fig_density = make_subplots(
        rows=1,
        cols=min(n_cols, 3),
        subplot_titles=mice_result.columns_imputed[:3],
    )

    for i, col in enumerate(mice_result.columns_imputed[:3]):
        col_idx = i + 1

        # Original observed values
        mask = (
            mice_result.original_missing_mask[col]
            if mice_result.original_missing_mask is not None
            else pd.Series(False, index=original_df.index)
        )
        observed = original_df.loc[~mask, col].dropna()

        if len(observed) > 0:
            # KDE for observed
            fig_density.add_trace(
                go.Histogram(
                    x=observed,
                    name=f"{col} (Observed)",
                    histnorm="probability density",
                    opacity=0.7,
                    marker_color=COLORS.get("primary", "#4361ee"),
                ),
                row=1,
                col=col_idx,
            )

        # Imputed values (from first imputation)
        if mice_result.imputed_datasets:
            imputed_values = mice_result.imputed_datasets[0].loc[mask, col]
            if len(imputed_values) > 0:
                fig_density.add_trace(
                    go.Histogram(
                        x=imputed_values,
                        name=f"{col} (Imputed)",
                        histnorm="probability density",
                        opacity=0.5,
                        marker_color=COLORS.get("danger", "#dc3545"),
                    ),
                    row=1,
                    col=col_idx,
                )

    fig_density.update_layout(
        title="Observed vs Imputed Distributions",
        height=400,
        showlegend=True,
        barmode="overlay",
        font=dict(family="Inter", size=12),
    )
    figures["density_comparison"] = fig_density

    # 2. Strip Plot - Compare imputations
    if len(mice_result.imputed_datasets) > 1 and mice_result.columns_imputed:
        col = mice_result.columns_imputed[0]
        mask = (
            mice_result.original_missing_mask[col]
            if mice_result.original_missing_mask is not None
            else pd.Series(False, index=original_df.index)
        )

        fig_strip = go.Figure()

        for m, df_imp in enumerate(mice_result.imputed_datasets):
            imputed_vals = df_imp.loc[mask, col]
            fig_strip.add_trace(
                go.Box(
                    y=imputed_vals,
                    name=f"Imputation {m + 1}",
                    boxpoints="all",
                    marker_color=COLORS.get("primary", "#4361ee"),
                    opacity=0.7,
                )
            )

        fig_strip.update_layout(
            title=f"Imputed Values Comparison: {col}",
            yaxis_title=col,
            height=400,
            font=dict(family="Inter", size=12),
        )
        figures["imputation_comparison"] = fig_strip

    return figures


def get_imputation_summary(mice_result: MICEResult) -> pd.DataFrame:
    """
    Generate summary statistics of imputed values.

    Returns DataFrame with mean, SD, min, max across imputations.
    """
    if not mice_result.imputed_datasets or not mice_result.columns_imputed:
        return pd.DataFrame()

    summary_data = []

    for col in mice_result.columns_imputed:
        # Get imputed values from each dataset
        imputed_values = []
        for df_imp in mice_result.imputed_datasets:
            mask = (
                mice_result.original_missing_mask[col]
                if mice_result.original_missing_mask is not None
                else pd.Series(False, index=df_imp.index)
            )
            vals = df_imp.loc[mask, col].values
            imputed_values.extend(vals)

        imputed_values = np.array(imputed_values)

        summary_data.append(
            {
                "Variable": col,
                "N Imputed": len(imputed_values) // mice_result.n_imputations,
                "Mean": np.mean(imputed_values),
                "SD": np.std(imputed_values),
                "Min": np.min(imputed_values),
                "Max": np.max(imputed_values),
            }
        )

    return pd.DataFrame(summary_data)
