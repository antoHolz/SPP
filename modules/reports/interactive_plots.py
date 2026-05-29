"""
Interactive Plotly plotting functions for Gradio viewers and notebooks.
Plotly equivalents of the matplotlib functions in report_dashboard.py and forecast_plot.py.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from modules.reports.plot_utils import MODEL_COLORS_HEX

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


MODEL_COLORS = MODEL_COLORS_HEX


##############################
# Helpers
##############################


def to_numpy(val):
    # 1) Convert tensor or ndarray to numpy for plotting
    if HAS_TORCH and isinstance(val, torch.Tensor):
        return val.detach().cpu().numpy()
    if isinstance(val, np.ndarray):
        return val.copy()
    return None


def _hex_to_rgba(hex_color, alpha):
    # 1) Convert hex color string to rgba() for fill regions
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    return f"rgba({r},{g},{b},{alpha})"


MAX_SMALL_DIM = 4


def inspect_sample(sample, sample_idx=None):
    # 1) Build table rows for each key in the sample dict
    #    Handles torch tensors, numpy arrays, and scalar types.
    #    If sample_idx is set, slices batch dim for .npz prediction data.
    rows = []
    for key in sorted(sample.keys()):
        val = sample[key]
        arr = to_numpy(val)

        ##############################
        # Case A: Numeric array (tensor or ndarray)
        ##############################
        if arr is not None:
            # 2) Slice batch dim when viewing .npz arrays
            if sample_idx is not None and arr.ndim >= 2 and arr.shape[0] > sample_idx:
                arr = arr[sample_idx]

            type_str = "Tensor" if (HAS_TORCH and isinstance(val, torch.Tensor)) else "ndarray"
            dtype_str = str(arr.dtype)
            shape_str = str(list(arr.shape))

            # 3) NaN-safe statistics
            if arr.size > 0 and np.issubdtype(arr.dtype, np.number):
                nan_count = int(np.count_nonzero(np.isnan(arr)))
                nan_str = str(nan_count) if nan_count > 0 else ""
                if nan_count == arr.size:
                    min_str = max_str = mean_str = "NaN"
                else:
                    min_str = f"{np.nanmin(arr):.4f}"
                    max_str = f"{np.nanmax(arr):.4f}"
                    mean_str = f"{np.nanmean(arr):.4f}"
            else:
                min_str = max_str = mean_str = "N/A"
                nan_str = ""

            # 4) Inline value for small arrays
            is_small = all(s <= MAX_SMALL_DIM for s in arr.shape) and arr.size > 0
            if is_small:
                with np.printoptions(precision=4, suppress=True, linewidth=200):
                    value_str = np.array2string(arr, separator=", ")
            else:
                value_str = ""

        ##############################
        # Case B: List or tuple
        ##############################
        elif isinstance(val, (list, tuple)):
            type_str = type(val).__name__
            dtype_str = "N/A"
            shape_str = f"len={len(val)}"
            min_str = max_str = mean_str = "N/A"
            nan_str = ""
            value_str = str(val) if len(val) <= MAX_SMALL_DIM else ""

        ##############################
        # Case C: Scalar (int, float)
        ##############################
        elif isinstance(val, (int, float)):
            type_str = type(val).__name__
            dtype_str = "N/A"
            shape_str = "scalar"
            min_str = max_str = mean_str = str(val)
            nan_str = "1" if (isinstance(val, float) and np.isnan(val)) else ""
            value_str = str(val)

        ##############################
        # Case D: Other types
        ##############################
        else:
            type_str = type(val).__name__
            dtype_str = "N/A"
            shape_str = "N/A"
            min_str = max_str = mean_str = "N/A"
            nan_str = ""
            value_str = ""

        rows.append([key, type_str, dtype_str, shape_str, min_str, max_str, mean_str,
                     nan_str, value_str])

    return rows


def _split_fill_segments(x_vals, lo_vals, hi_vals):
    # 1) Split quantile fill regions into contiguous non-NaN segments.
    #    Returns list of (x_seg, lo_seg, hi_seg) tuples for NaN-safe fill rendering.
    x_arr = np.asarray(x_vals)
    lo_arr = np.asarray(lo_vals, dtype=float)
    hi_arr = np.asarray(hi_vals, dtype=float)
    valid = np.isfinite(lo_arr) & np.isfinite(hi_arr)

    segments = []
    start = None
    for i in range(len(valid)):
        if valid[i] and start is None:
            start = i
        elif not valid[i] and start is not None:
            segments.append((x_arr[start:i], lo_arr[start:i], hi_arr[start:i]))
            start = None
    if start is not None:
        segments.append((x_arr[start:], lo_arr[start:], hi_arr[start:]))
    return segments


##############################
# Multi-model forecast plot
##############################


def multi_model_forecast_plotly(
    model_results, sample_idx=0, dims=None, title=None, height_per_dim=250,
    show_quantiles=True, show_samples=True
):
    ##############################
    # Step 1: Extract shared context and ground truth
    ##############################
    # 1) Use first model's data for context and ground truth (same across models)
    first_name, first_arrays = model_results[0]
    x = first_arrays["x"]
    y_true = first_arrays.get("y_true")

    num_dims = x.shape[-1]
    L_ctx = x.shape[1]

    # 2) Determine which dimensions to plot (filter out-of-range)
    if dims is None:
        dims = list(range(num_dims))
    else:
        dims = [d for d in dims if 0 <= d < num_dims]
        if not dims:
            dims = list(range(num_dims))

    # 3) Build time axes
    t_ctx = list(range(L_ctx))

    ##############################
    # Step 2: Create subplots
    ##############################
    # 4) One subplot per dimension
    n_plots = len(dims)
    fig = make_subplots(
        rows=n_plots,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=[f"dim {d}" for d in dims],
    )

    for plot_idx, d in enumerate(dims):
        row = plot_idx + 1

        # 5) Plot context (shared)
        fig.add_trace(
            go.Scatter(
                x=t_ctx,
                y=x[sample_idx, :, d],
                mode="lines",
                line=dict(color="black", width=1),
                name="context",
                showlegend=(plot_idx == 0),
                legendgroup="context",
            ),
            row=row,
            col=1,
        )

        # 6) Plot ground truth (shared)
        if y_true is not None:
            L_out = y_true.shape[1]
            t_out = list(range(L_ctx, L_ctx + L_out))
            fig.add_trace(
                go.Scatter(
                    x=t_out,
                    y=y_true[sample_idx, :, d],
                    mode="lines",
                    line=dict(color="red", width=1, dash="dash"),
                    name="ground truth",
                    showlegend=(plot_idx == 0),
                    legendgroup="ground_truth",
                ),
                row=row,
                col=1,
            )

        ##############################
        # Step 3: Overlay each model's predictions
        ##############################
        for model_idx, (model_name, arrays) in enumerate(model_results):
            color = MODEL_COLORS[model_idx % len(MODEL_COLORS)]

            # 7) Clamp sample index to this model's batch size
            n_model = arrays.get("y_hat", arrays.get("x", np.empty((0,)))).shape[0]
            if n_model == 0:
                continue
            sidx = min(sample_idx, n_model - 1)

            # 8) Point forecast
            if "y_hat" in arrays and arrays["y_hat"] is not None:
                y_hat = arrays["y_hat"]
                L_out = y_hat.shape[1]
                t_out = list(range(L_ctx, L_ctx + L_out))
                fig.add_trace(
                    go.Scatter(
                        x=t_out,
                        y=y_hat[sidx, :, d],
                        mode="lines",
                        line=dict(color=color, width=1.5),
                        opacity=0.7,
                        name=model_name,
                        showlegend=(plot_idx == 0),
                        legendgroup=f"model_{model_idx}",
                    ),
                    row=row,
                    col=1,
                )

            # 9) Quantile bands (NaN-safe segmented fill)
            if show_quantiles and "y_q" in arrays and arrays["y_q"] is not None:
                y_q = arrays["y_q"]
                L_out = y_q.shape[2]
                t_out = np.arange(L_ctx, L_ctx + L_out)
                num_q = y_q.shape[1]
                for lo_idx in range(num_q // 2):
                    hi_idx = num_q - 1 - lo_idx
                    band_alpha = 0.08 + 0.04 * lo_idx

                    lo_vals = y_q[sidx, lo_idx, :, d]
                    hi_vals = y_q[sidx, hi_idx, :, d]
                    segments = _split_fill_segments(t_out, lo_vals, hi_vals)

                    for x_seg, lo_seg, hi_seg in segments:
                        fig.add_trace(
                            go.Scatter(
                                x=x_seg.tolist(),
                                y=lo_seg.tolist(),
                                mode="lines",
                                line=dict(width=0),
                                showlegend=False,
                                hoverinfo="skip",
                            ),
                            row=row,
                            col=1,
                        )
                        fig.add_trace(
                            go.Scatter(
                                x=x_seg.tolist(),
                                y=hi_seg.tolist(),
                                mode="lines",
                                line=dict(width=0),
                                fill="tonexty",
                                fillcolor=_hex_to_rgba(color, band_alpha),
                                showlegend=False,
                                hoverinfo="skip",
                            ),
                            row=row,
                            col=1,
                        )

            # 10) Sample trajectories (if available)
            if show_samples and "y_s" in arrays and arrays["y_s"] is not None:
                y_s = arrays["y_s"]
                L_out = y_s.shape[2]
                t_out = list(range(L_ctx, L_ctx + L_out))
                num_samples = min(y_s.shape[1], 20)
                for s in range(num_samples):
                    fig.add_trace(
                        go.Scatter(
                            x=t_out,
                            y=y_s[sidx, s, :, d],
                            mode="lines",
                            line=dict(color=color, width=0.5),
                            opacity=0.15,
                            name=f"{model_name} samples" if s == 0 else None,
                            showlegend=(plot_idx == 0 and s == 0),
                            legendgroup=f"model_{model_idx}_samples",
                        ),
                        row=row,
                        col=1,
                    )

    ##############################
    # Step 4: Finalize figure
    ##############################
    # 11) Layout and labels
    fig.update_layout(
        title_text=title,
        height=max(300, height_per_dim * n_plots),
        hovermode="x unified",
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
        margin=dict(l=60, r=20, t=80, b=40),
    )
    fig.update_xaxes(title_text="time step", row=n_plots, col=1)

    return fig


##############################
# Reusable trace builders (no figure / no IO)
##############################
# These three functions return lists of go.Scatter traces that can be added into
# any Plotly subplot grid. They are independently callable from notebooks.


def traces_line_quantile(df_dp, sweep_col, metric_col, models, model_colors,
                          band_alpha=0.18):
    # 1) Build per-model (band, line) trace pairs for a line_quantile view.
    #    Mirrors the matplotlib branch in report_dashboard._draw_metric_panel
    #    (lines 498-514): groupby sweep_col -> quantile([0.25, 0.5, 0.75]).
    #    Returns a flat list ordered as [m1_band, m1_line, m2_band, m2_line, ...]
    #    so callers know each model occupies trace pair (i, i+1).
    traces = []
    for model in models:
        color = model_colors[model]
        rgba = _hex_to_rgba(color, band_alpha)
        x_idx, q25, q50, q75 = _model_quantile_arrays(df_dp, sweep_col, metric_col, model)

        if len(x_idx) == 0:
            # 2) Empty placeholders so trace ordering stays stable for restyle()
            traces.append(go.Scatter(
                x=[], y=[], fill="toself", fillcolor=rgba,
                line=dict(color="rgba(0,0,0,0)"), hoverinfo="skip",
                showlegend=False, legendgroup=model,
            ))
            traces.append(go.Scatter(
                x=[], y=[], mode="lines", line=dict(color=color, width=2),
                name=model, legendgroup=model, showlegend=True,
            ))
            continue

        x_band = np.concatenate([x_idx, x_idx[::-1]])
        y_band = np.concatenate([q75, q25[::-1]])
        traces.append(go.Scatter(
            x=x_band, y=y_band, fill="toself", fillcolor=rgba,
            line=dict(color="rgba(0,0,0,0)"), hoverinfo="skip",
            showlegend=False, legendgroup=model,
        ))
        traces.append(go.Scatter(
            x=x_idx, y=q50, mode="lines", line=dict(color=color, width=2),
            name=model, legendgroup=model, showlegend=True,
            hovertemplate=f"{model}<br>{sweep_col}=%{{x:.4g}}<br>{metric_col} (median)=%{{y:.4g}}<extra></extra>",
        ))
    return traces


def traces_error_growth(cache_entry, model, model_color, band_alpha=0.18):
    # 1) Build per-model (band, line) traces for the error-vs-time view.
    #    `cache_entry` is None or {"t","q25","q50","q75"} (lists of length L_out)
    #    produced by report_dashboard.compute_error_growth_quantiles.
    #    Returns [band, line] so each model occupies trace pair (i, i+1).
    rgba = _hex_to_rgba(model_color, band_alpha)
    if cache_entry is None or not cache_entry.get("t"):
        # 2) Empty placeholders keep trace ordering stable for restyle()/visible toggles.
        return [
            go.Scatter(
                x=[], y=[], fill="toself", fillcolor=rgba,
                line=dict(color="rgba(0,0,0,0)"), hoverinfo="skip",
                showlegend=False, legendgroup=model,
            ),
            go.Scatter(
                x=[], y=[], mode="lines", line=dict(color=model_color, width=2),
                name=model, legendgroup=model, showlegend=False,
            ),
        ]
    t = list(cache_entry["t"])
    q25 = list(cache_entry["q25"])
    q50 = list(cache_entry["q50"])
    q75 = list(cache_entry["q75"])
    x_band = t + t[::-1]
    y_band = q75 + q25[::-1]
    return [
        go.Scatter(
            x=x_band, y=y_band, fill="toself", fillcolor=rgba,
            line=dict(color="rgba(0,0,0,0)"), hoverinfo="skip",
            showlegend=False, legendgroup=model,
        ),
        go.Scatter(
            x=t, y=q50, mode="lines", line=dict(color=model_color, width=2),
            name=model, legendgroup=model, showlegend=False,
            hovertemplate=f"{model}<br>step=%{{x}}<br>err (median)=%{{y:.4g}}<extra></extra>",
        ),
    ]


def _model_quantile_arrays(df_dp, sweep_col, metric_col, model):
    # 1) Helper: groupby + quantile → 4 numpy arrays (x, q25, q50, q75).
    if sweep_col not in df_dp.columns or metric_col not in df_dp.columns:
        return np.array([]), np.array([]), np.array([]), np.array([])
    df_m = df_dp[df_dp["model"] == model]
    if df_m.empty:
        return np.array([]), np.array([]), np.array([]), np.array([])
    qs = df_m.groupby(sweep_col)[metric_col].quantile([0.25, 0.5, 0.75]).unstack(level=-1).sort_index()
    # 2) All-NaN metric column collapses unstack to (0, 0); treat as no data.
    if 0.25 not in qs.columns:
        return np.array([]), np.array([]), np.array([]), np.array([])
    return qs.index.values, qs[0.25].values, qs[0.5].values, qs[0.75].values


def traces_forecast(predictions_by_model, dim, model_colors):
    # 1) Build forecast traces for one dimension.
    #    Returns: [context, ground_truth, model_1_yhat, model_2_yhat, ...]
    first = next((p for p in predictions_by_model.values() if p is not None), None)
    if first is None:
        return []
    L_ctx = first["x"].shape[0]
    L_out = first["y"].shape[0]
    n_dims = first["x"].shape[1]
    d = max(0, min(int(dim), n_dims - 1))
    t_ctx = list(range(L_ctx))
    t_out = list(range(L_ctx, L_ctx + L_out))

    traces = [
        go.Scatter(
            x=t_ctx, y=first["x"][:, d], mode="lines",
            line=dict(color="black", width=1), name="context",
            legendgroup="__context__",
        ),
        go.Scatter(
            x=t_out, y=first["y"][:, d], mode="lines",
            line=dict(color="black", width=1.5, dash="dash"), name="ground truth",
            legendgroup="__truth__",
        ),
    ]
    for model, pred in predictions_by_model.items():
        color = model_colors.get(model, "#999999")
        if pred is None:
            traces.append(go.Scatter(
                x=[], y=[], mode="lines", line=dict(color=color, width=1.2),
                name=model, legendgroup=model, showlegend=False,
            ))
            continue
        traces.append(go.Scatter(
            x=t_out, y=pred["y_hat"][:, d], mode="lines",
            line=dict(color=color, width=1.2), opacity=0.85,
            name=model, legendgroup=model, showlegend=False,
            hovertemplate=f"{model}<br>t=%{{x}}<br>y=%{{y:.4g}}<extra></extra>",
        ))
    return traces


def traces_phase_portrait(predictions_by_model, dim_pair, model_colors):
    # 1) Build phase-portrait traces (2D x-vs-y for the given dim pair).
    #    Returns: [context, ground_truth, model_1_yhat, model_2_yhat, ...]
    da, db = int(dim_pair[0]), int(dim_pair[1])
    first = next((p for p in predictions_by_model.values() if p is not None), None)
    if first is None:
        return []
    n_dims = first["x"].shape[1]
    if n_dims < 2 or da >= n_dims or db >= n_dims:
        # 2) 1D system → return empty traces in the same slots so restyle() stays valid
        n_extra = 2 + len(predictions_by_model)
        return [go.Scatter(x=[], y=[], showlegend=False) for _ in range(n_extra)]

    traces = [
        go.Scatter(
            x=first["x"][:, da], y=first["x"][:, db], mode="lines",
            line=dict(color="black", width=1), name="context",
            legendgroup="__context__", showlegend=False,
        ),
        go.Scatter(
            x=first["y"][:, da], y=first["y"][:, db], mode="lines",
            line=dict(color="black", width=1.5, dash="dash"), name="ground truth",
            legendgroup="__truth__", showlegend=False,
        ),
    ]
    for model, pred in predictions_by_model.items():
        color = model_colors.get(model, "#999999")
        if pred is None:
            traces.append(go.Scatter(
                x=[], y=[], mode="lines", line=dict(color=color, width=1.2),
                name=model, legendgroup=model, showlegend=False,
            ))
            continue
        traces.append(go.Scatter(
            x=pred["y_hat"][:, da], y=pred["y_hat"][:, db], mode="lines",
            line=dict(color=color, width=1.2), opacity=0.85,
            name=model, legendgroup=model, showlegend=False,
        ))
    return traces


##############################
# Sample fields plot
##############################

MAX_FEATURES = 20


def sample_fields_plotly(sample, max_features=MAX_FEATURES, height_per_subplot=250):
    ##############################
    # Step 1: Collect plottable fields
    ##############################
    # 1) Gather numeric arrays with at least 1 dimension
    plottable = {}
    for key in sorted(sample.keys()):
        arr = to_numpy(sample[key])
        if arr is not None and arr.ndim >= 1 and np.issubdtype(arr.dtype, np.number):
            plottable[key] = arr

    # 2) Nothing to plot
    if not plottable:
        fig = go.Figure()
        fig.add_annotation(
            text="No plottable fields in this sample",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font_size=16,
        )
        return fig

    ##############################
    # Step 2: Detect context+prediction pattern
    ##############################
    # 3) Check for x and y both present
    has_xy = "x" in plottable and "y" in plottable

    if has_xy:
        other_keys = [k for k in plottable if k not in ("x", "y")]
        subplot_titles = ["Context (x) + Prediction (y)"] + other_keys
    else:
        other_keys = list(plottable.keys())
        subplot_titles = other_keys

    n_subplots = len(subplot_titles)
    fig = make_subplots(
        rows=n_subplots,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.06,
        subplot_titles=subplot_titles,
    )

    ##############################
    # Step 3: Plot context+prediction (if present)
    ##############################
    if has_xy:
        x_arr = plottable["x"]
        y_arr = plottable["y"]

        # 4) Ensure 2D: (L,) -> (L, 1)
        if x_arr.ndim == 1:
            x_arr = x_arr[:, None]
        if y_arr.ndim == 1:
            y_arr = y_arr[:, None]

        ctx_len = x_arr.shape[0]
        pred_len = y_arr.shape[0]
        n_feat = max(x_arr.shape[1], y_arr.shape[1])

        # 5) Plot each feature as context (solid) + prediction (dashed)
        colors = [MODEL_COLORS[i % len(MODEL_COLORS)] for i in range(n_feat)]
        for feat_idx in range(min(n_feat, max_features)):
            color = colors[feat_idx]
            suffix = f" (feat {feat_idx})" if n_feat > 1 else ""

            if feat_idx < x_arr.shape[1]:
                fig.add_trace(
                    go.Scatter(
                        x=list(range(ctx_len)),
                        y=x_arr[:, feat_idx],
                        mode="lines",
                        line=dict(color=color, width=1.2),
                        name=f"context{suffix}",
                    ),
                    row=1,
                    col=1,
                )
            if feat_idx < y_arr.shape[1]:
                fig.add_trace(
                    go.Scatter(
                        x=list(range(ctx_len, ctx_len + pred_len)),
                        y=y_arr[:, feat_idx],
                        mode="lines",
                        line=dict(color=color, width=1.2, dash="dash"),
                        name=f"prediction{suffix}",
                    ),
                    row=1,
                    col=1,
                )

        # 6) Vertical separator at context/prediction boundary
        fig.add_vline(
            x=ctx_len, line_dash="dot", line_color="gray", opacity=0.7, row=1, col=1
        )

    ##############################
    # Step 4: Plot remaining fields
    ##############################
    # 7) Each other field gets its own subplot
    base_row = 2 if has_xy else 1
    for i, key in enumerate(other_keys):
        arr = plottable[key]
        row = base_row + i

        if arr.ndim == 1:
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(arr))),
                    y=arr,
                    mode="lines",
                    line=dict(width=1.0),
                    name=key,
                    showlegend=True,
                ),
                row=row,
                col=1,
            )
        else:
            n_feat = arr.shape[1]
            for feat_idx in range(min(n_feat, max_features)):
                fig.add_trace(
                    go.Scatter(
                        x=list(range(arr.shape[0])),
                        y=arr[:, feat_idx],
                        mode="lines",
                        line=dict(width=1.0),
                        name=f"{key} feat {feat_idx}",
                        showlegend=(n_feat <= 10),
                    ),
                    row=row,
                    col=1,
                )

    ##############################
    # Step 5: Finalize
    ##############################
    # 8) Update layout
    fig.update_layout(
        height=max(300, height_per_subplot * n_subplots),
        hovermode="x unified",
        margin=dict(l=60, r=20, t=40, b=40),
    )

    # 9) Add x-axis label to bottom subplot
    fig.update_xaxes(title_text="Timestep", row=n_subplots, col=1)

    return fig


##############################
# Interactive 3D phase portrait
##############################


def _normalize_preds_plotly(predictions):
    # 1) Convert all arrays in predictions dict to numpy
    if predictions is None:
        return {}
    out = {}
    for name, pred in predictions.items():
        out[name] = {}
        for key in ["y_hat", "y_q", "y_s"]:
            if key in pred and pred[key] is not None:
                arr = to_numpy(pred[key])
                if arr is not None:
                    out[name][key] = arr
    return out


def interactive_3d_phase_portrait(x=None, y=None, predictions=None,
                                  dim_triple=None, title=None, height=600):
    """Interactive 3D phase portrait via Plotly.

    All inputs are single-sample (no batch dim), matching sample_plots interface.
    x: (L_ctx, D), y: (L_out, D), predictions: dict[str, dict] with y_hat arrays.
    Falls back to 2D for D < 3.
    """
    # 1) Convert inputs to numpy
    x = to_numpy(x) if x is not None else None
    y = to_numpy(y) if y is not None else None
    predictions = _normalize_preds_plotly(predictions)

    if x is None and y is None and not predictions:
        raise ValueError("At least one of x, y, predictions must be provided")

    # 2) Infer dimensionality
    D = None
    for arr in [x, y]:
        if arr is not None:
            D = arr.shape[-1]
            break
    if D is None:
        for pred in predictions.values():
            if "y_hat" in pred:
                D = pred["y_hat"].shape[-1]
                break
    if D is None or D < 2:
        raise ValueError("Phase portrait requires D >= 2")

    # 3) Choose dimensions
    use_3d = D >= 3
    if dim_triple is None:
        dim_triple = (0, 1, 2) if use_3d else (0, 1)

    fig = go.Figure()

    if use_3d:
        di, dj, dk = dim_triple[0], dim_triple[1], dim_triple[2]

        # 4a) Context trajectory (3D)
        if x is not None:
            fig.add_trace(go.Scatter3d(
                x=x[:, di], y=x[:, dj], z=x[:, dk],
                mode="lines+markers",
                line=dict(color="black", width=2),
                marker=dict(size=1),
                name="context", legendgroup="context",
            ))
            fig.add_trace(go.Scatter3d(
                x=[x[0, di]], y=[x[0, dj]], z=[x[0, dk]],
                mode="markers",
                marker=dict(color="black", size=5, symbol="circle"),
                name="start", showlegend=False, legendgroup="context",
            ))

        # 4b) Ground truth (3D)
        if y is not None:
            gt = y
            if x is not None:
                gt = np.concatenate([x[-1:], y], axis=0)
            fig.add_trace(go.Scatter3d(
                x=gt[:, di], y=gt[:, dj], z=gt[:, dk],
                mode="lines",
                line=dict(color="red", width=2, dash="dash"),
                name="ground truth", legendgroup="ground_truth",
            ))

        # 4c) Model predictions (3D)
        for model_idx, (model_name, pred) in enumerate(predictions.items()):
            if "y_hat" not in pred:
                continue
            color = MODEL_COLORS[model_idx % len(MODEL_COLORS)]
            y_hat = pred["y_hat"]
            if x is not None:
                full = np.concatenate([x[-1:], y_hat], axis=0)
            else:
                full = y_hat
            fig.add_trace(go.Scatter3d(
                x=full[:, di], y=full[:, dj], z=full[:, dk],
                mode="lines",
                line=dict(color=color, width=2),
                name=model_name, legendgroup=f"model_{model_idx}",
            ))

            # Sample trajectories
            if "y_s" in pred:
                y_s = pred["y_s"]
                n_samples = min(y_s.shape[0], 10)
                for s in range(n_samples):
                    traj = y_s[s]
                    if x is not None:
                        traj = np.concatenate([x[-1:], traj], axis=0)
                    fig.add_trace(go.Scatter3d(
                        x=traj[:, di], y=traj[:, dj], z=traj[:, dk],
                        mode="lines",
                        line=dict(color=color, width=0.5),
                        opacity=0.15,
                        showlegend=False, legendgroup=f"model_{model_idx}",
                    ))

        fig.update_layout(
            scene=dict(
                xaxis_title=f"dim {di}",
                yaxis_title=f"dim {dj}",
                zaxis_title=f"dim {dk}",
            ),
        )

    else:
        di, dj = dim_triple[0], dim_triple[1]

        # 5a) Context trajectory (2D)
        if x is not None:
            fig.add_trace(go.Scatter(
                x=x[:, di], y=x[:, dj],
                mode="lines+markers",
                line=dict(color="black", width=2),
                marker=dict(size=2),
                name="context",
            ))

        # 5b) Ground truth (2D)
        if y is not None:
            gt = y
            if x is not None:
                gt = np.concatenate([x[-1:], y], axis=0)
            fig.add_trace(go.Scatter(
                x=gt[:, di], y=gt[:, dj],
                mode="lines",
                line=dict(color="red", width=2, dash="dash"),
                name="ground truth",
            ))

        # 5c) Model predictions (2D)
        for model_idx, (model_name, pred) in enumerate(predictions.items()):
            if "y_hat" not in pred:
                continue
            color = MODEL_COLORS[model_idx % len(MODEL_COLORS)]
            y_hat = pred["y_hat"]
            if x is not None:
                full = np.concatenate([x[-1:], y_hat], axis=0)
            else:
                full = y_hat
            fig.add_trace(go.Scatter(
                x=full[:, di], y=full[:, dj],
                mode="lines",
                line=dict(color=color, width=2),
                name=model_name,
            ))

        fig.update_xaxes(title_text=f"dim {di}")
        fig.update_yaxes(title_text=f"dim {dj}", scaleanchor="x")

    # 6) Finalize
    fig.update_layout(
        title_text=title,
        height=height,
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
        margin=dict(l=60, r=20, t=80, b=40),
    )
    return fig


##############################
# Interactive error heatmap
##############################


def interactive_error_heatmap(y=None, predictions=None, title=None, height=400):
    """Interactive heatmap of |y - y_hat| over (time x dimension) per model.

    All inputs are single-sample (no batch dim).
    y: (L, D), predictions: dict[str, dict] with y_hat arrays.
    """
    # 1) Convert inputs
    y = to_numpy(y) if y is not None else None
    predictions = _normalize_preds_plotly(predictions)

    if y is None:
        raise ValueError("interactive_error_heatmap requires y (ground truth)")
    if not predictions:
        raise ValueError("interactive_error_heatmap requires at least one prediction")

    model_names = list(predictions.keys())
    n_models = len(model_names)
    L, D = y.shape

    # 2) Create subplots — one row per model
    fig = make_subplots(
        rows=n_models, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=model_names,
    )

    # 3) Compute and plot error heatmap per model
    for i, name in enumerate(model_names):
        y_hat = predictions[name].get("y_hat")
        if y_hat is None:
            continue
        abs_error = np.abs(y - y_hat)  # (L, D)

        fig.add_trace(
            go.Heatmap(
                z=abs_error.T,  # (D, L) so y-axis = dimensions, x-axis = time
                x=list(range(L)),
                y=[f"dim {d}" for d in range(D)],
                colorscale="Viridis",
                colorbar=dict(title="error", len=0.8 / n_models, y=1 - (i + 0.5) / n_models),
                hovertemplate="t=%{x}<br>%{y}<br>error=%{z:.4f}<extra>" + name + "</extra>",
            ),
            row=i + 1, col=1,
        )

    # 4) Finalize
    fig.update_layout(
        title_text=title,
        height=max(300, height * n_models),
        margin=dict(l=80, r=80, t=80, b=40),
    )
    fig.update_xaxes(title_text="time step", row=n_models, col=1)

    return fig


##############################
# Interactive metric explorer
##############################


def interactive_metric_explorer(data, x_metric=None, y_metric=None,
                                color_col="model", hover_cols=None,
                                title=None, height=600, save_dir=None):
    """Interactive scatter with selectable X/Y axes from metric columns.

    data: DataFrame or path to CSV (e.g., results_mean.csv or per_datapoint.csv).
    Points colored by model, hover shows dataset/seed/metrics.
    Includes dropdown selectors for X and Y axis.
    """
    from pathlib import Path
    from modules.reports.plot_utils import resolve_data

    # 1) Load data
    df = resolve_data(data)
    if df is None or len(df) == 0:
        fig = go.Figure()
        fig.add_annotation(text="No data", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False)
        return fig

    # 2) Identify numeric columns as potential metrics
    numeric_cols = [c for c in df.columns if df[c].dtype.kind in "iuf"]
    if len(numeric_cols) < 2:
        raise ValueError("Need at least 2 numeric columns for metric explorer")

    if x_metric is None:
        x_metric = numeric_cols[0]
    if y_metric is None:
        y_metric = numeric_cols[1]

    if hover_cols is None:
        hover_cols = [c for c in ["model", "data", "seed"] if c in df.columns]

    # 3) Assign colors per group
    groups = df[color_col].unique() if color_col in df.columns else ["all"]
    color_map = {g: MODEL_COLORS[i % len(MODEL_COLORS)] for i, g in enumerate(groups)}

    # 4) Build initial scatter traces (one per group for legend)
    fig = go.Figure()
    for group in groups:
        if color_col in df.columns:
            mask = df[color_col] == group
            sub = df[mask]
        else:
            sub = df
        hover_text = sub.apply(
            lambda row: "<br>".join(f"{c}: {row[c]}" for c in hover_cols), axis=1
        )
        fig.add_trace(go.Scatter(
            x=sub[x_metric],
            y=sub[y_metric],
            mode="markers",
            marker=dict(color=color_map[group], size=8, opacity=0.7),
            name=str(group),
            text=hover_text,
            hovertemplate="%{text}<br>" + x_metric + "=%{x:.4f}<br>"
                          + y_metric + "=%{y:.4f}<extra></extra>",
        ))

    # 5) Build dropdown menus for axis selection
    x_buttons = []
    for col in numeric_cols:
        x_buttons.append(dict(
            label=col,
            method="update",
            args=[
                {"x": [df[df[color_col] == g][col] if color_col in df.columns
                        else df[col] for g in groups]},
                {"xaxis.title.text": col},
            ],
        ))

    y_buttons = []
    for col in numeric_cols:
        y_buttons.append(dict(
            label=col,
            method="update",
            args=[
                {"y": [df[df[color_col] == g][col] if color_col in df.columns
                        else df[col] for g in groups]},
                {"yaxis.title.text": col},
            ],
        ))

    fig.update_layout(
        title_text=title or "Metric Explorer",
        height=height,
        xaxis_title=x_metric,
        yaxis_title=y_metric,
        hovermode="closest",
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
        margin=dict(l=60, r=20, t=120, b=40),
        updatemenus=[
            dict(
                buttons=x_buttons,
                direction="down",
                x=0.15, xanchor="left",
                y=1.12, yanchor="top",
                showactive=True,
                active=numeric_cols.index(x_metric),
            ),
            dict(
                buttons=y_buttons,
                direction="down",
                x=0.45, xanchor="left",
                y=1.12, yanchor="top",
                showactive=True,
                active=numeric_cols.index(y_metric),
            ),
        ],
        annotations=[
            dict(text="X-axis:", x=0.10, xref="paper", xanchor="right",
                 y=1.11, yref="paper", showarrow=False),
            dict(text="Y-axis:", x=0.40, xref="paper", xanchor="right",
                 y=1.11, yref="paper", showarrow=False),
        ],
    )

    # 6) Optionally save as HTML
    if save_dir is not None:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(save_path / "metric_explorer.html"))

    return fig
