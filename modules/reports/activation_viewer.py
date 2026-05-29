"""
Interactive activation viewer.

Turns intermediate activations from a model (as produced by
`fmds.models.utils.layers.getActivations`) into an interactive 3D / 2D scatter
backed by Plotly + Gradio. Rotation, zoom, and pan are native Plotly. Clicking
a point updates the right-hand input panel with the time series that produced
it; the manual sample / token number inputs stay available as a fallback.

Click bridge
------------
Gradio 6.x `gr.Plot` does not expose a `select` / `click` event, so we attach
a `plotly_click` listener directly on the underlying chart div from JavaScript
(see `_CLICK_BRIDGE_JS`). The listener writes "sample_idx,token_idx" into a
hidden Gradio Textbox; the textbox's `change` event then triggers the Python
callback that redraws the input panel. A MutationObserver re-attaches the
listener whenever Gradio swaps in a new Plotly chart (after a layer / reducer
/ color change).

Persistence
-----------
`save_session` writes everything needed to rebuild the viewer to a single
`.npz` file (no pickle); `load_session` returns kwargs ready for
`build_activation_viewer_app(**kwargs)`. `save_scatter_html` exports a
single (layer, method, color) projection as a standalone Plotly HTML
(rotation / hover / zoom but no click-to-input -- those need a live server).

Activation contract
-------------------
Each activation tensor is shape (T, B, D):
    T -- token / sequence position
    B -- batch sample
    D -- feature / hidden dimension

`flatten_activations` and `broadcast_labels` (in `fmds.models.utils.layers`)
collapse this to (T*B, D) features plus matching (T*B,) per-point label
arrays. This file consumes that flat representation and never re-implements
the shape juggling.

Reducers
--------
PCA / TSNE / MDS are provided via scikit-learn (always installed). UMAP is
opt-in: if `umap-learn` is importable it appears as a fourth choice; if not
the reducer dropdown only offers the three sklearn options.
"""

import json
import os

import gradio as gr
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE

from modules.models.utils.layers import broadcast_labels, flatten_activations
from modules.reports.interactive_plots import to_numpy
from modules.reports.plot_utils import MODEL_COLORS_HEX


try:
    import umap  # noqa: F401
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False


REDUCER_CHOICES = ["pca", "tsne", "mds"] + (["umap"] if HAS_UMAP else [])
CATEGORICAL_MAX_UNIQUE = 20

# Marker symbols available to Plotly 3D scatter (limited set).
SYMBOL_PALETTE_3D = [
    "circle", "square", "diamond", "cross", "x",
    "circle-open", "square-open", "diamond-open",
]
# 2D scatter accepts a much larger set; pick visually distinct ones.
SYMBOL_PALETTE_2D = [
    "circle", "square", "diamond", "triangle-up", "triangle-down",
    "star", "hexagon", "pentagon", "cross", "x",
]
SYMBOL_NONE = "(none)"


##############################
# Reduction
##############################


def reduce(features, method, n_components, seed=0):
    """
    Project `features` (N, D) down to `n_components` dimensions.

    method: one of 'pca', 'tsne', 'mds', 'umap'. PCA / TSNE / MDS are backed
            by scikit-learn. 'umap' requires the optional `umap-learn`
            package; a missing install raises ImportError with a clear hint.
    Returns: ndarray (N, n_components).
    """
    N = features.shape[0]  # (N, D)

    if method == "pca":
        # 1) Linear projection, fast and deterministic
        reducer = PCA(n_components=n_components, random_state=seed)
        return reducer.fit_transform(features)

    if method == "tsne":
        # 2) Perplexity must be strictly less than N; clamp for tiny batches
        perplexity = float(min(30, max(2, N - 1)))
        reducer = TSNE(n_components=n_components, perplexity=perplexity, random_state=seed)
        return reducer.fit_transform(features)

    if method == "mds":
        reducer = MDS(n_components=n_components, random_state=seed, normalized_stress="auto")
        return reducer.fit_transform(features)

    if method == "umap":
        if not HAS_UMAP:
            raise ImportError(
                "method='umap' requires the optional dependency `umap-learn`. "
                "Install with: pip install umap-learn"
            )
        import umap as _umap
        # 3) n_neighbors must be < N; sklearn-style API
        n_neighbors = int(min(15, max(2, N - 1)))
        reducer = _umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, random_state=seed)
        return reducer.fit_transform(features)

    raise ValueError(f"Unknown method '{method}'. Expected one of {REDUCER_CHOICES}.")


##############################
# Coloring helpers
##############################


def _is_categorical(values):
    # 1) Strings always categorical; numerics categorical iff small cardinality
    arr = np.asarray(values)
    if arr.dtype.kind in ("U", "S", "O"):
        return True
    uniques = np.unique(arr[~_nan_mask(arr)]) if arr.size else arr
    return len(uniques) <= CATEGORICAL_MAX_UNIQUE


def _nan_mask(arr):
    # 1) NaN check that works for object/string dtypes
    if arr.dtype.kind in ("f", "c"):
        return np.isnan(arr)
    return np.zeros(arr.shape, dtype=bool)


def _categorical_color_map(values):
    # 1) Stable order: by first appearance, so legend reads top-to-bottom
    arr = np.asarray(values)
    seen = []
    for v in arr:
        if v not in seen:
            seen.append(v)
    palette = MODEL_COLORS_HEX
    return {v: palette[i % len(palette)] for i, v in enumerate(seen)}


def _symbol_map(values, dim):
    """Map unique categorical values to Plotly marker symbols.

    Palette is dim-dependent: 3D scatter supports only a small symbol set.
    Cycles via modulo if there are more unique values than symbols.
    """
    arr = np.asarray(values)
    seen = []
    for v in arr:
        if v not in seen:
            seen.append(v)
    palette = SYMBOL_PALETTE_3D if dim == 3 else SYMBOL_PALETTE_2D
    return {v: palette[i % len(palette)] for i, v in enumerate(seen)}


##############################
# Scatter figure
##############################


def scatter_plotly(emb, labels, customdata, color_label, dim, title=None,
                   symbol_label=None):
    """
    Build a Plotly scatter (3D if dim==3, else 2D) over the embedding `emb`.

    emb:          (N, dim) ndarray with dim in {2, 3}.
    labels:       dict[str, ndarray (N,)] -- per-point labels, output of
                  `broadcast_labels`. The active coloring is `labels[color_label]`.
    customdata:   (N, 2) array of [sample_idx, token_idx]. Wired into the trace
                  so Gradio's select callback can read it back without
                  maintaining its own index.
    color_label:  key of `labels` driving marker color. Categorical labels get
                  one trace per category (so the legend can toggle them);
                  continuous labels use a single trace with a Viridis colorbar.
    symbol_label: optional second feature mapped to marker symbols. Must be
                  categorical; ignored if missing, equal to SYMBOL_NONE, or
                  not categorical. When active, per-point marker.symbol arrays
                  are set on the data traces and dummy `legendonly` traces are
                  added to display the symbol mapping in the legend.
    Returns: plotly.graph_objects.Figure.
    """
    if dim not in (2, 3):
        raise ValueError(f"dim must be 2 or 3; got {dim}")

    color_vals = np.asarray(labels[color_label])  # (N,)
    categorical = _is_categorical(color_vals)

    # 1a) Symbol encoding only kicks in if a categorical label is selected.
    use_symbol = (
        symbol_label is not None
        and symbol_label != SYMBOL_NONE
        and symbol_label in labels
        and _is_categorical(np.asarray(labels[symbol_label]))
    )
    if use_symbol:
        symbol_vals = np.asarray(labels[symbol_label])  # (N,)
        symbol_map = _symbol_map(symbol_vals, dim)
        # (N,) per-point symbol assignment, looked up once.
        point_symbols = np.array([symbol_map[v] for v in symbol_vals])
    else:
        point_symbols = None
        symbol_map = None

    # 1b) Hover template lists every label so user sees full context per point.
    #    sample_idx / token_idx live in customdata[0:2] (kept first so the JS
    #    click bridge can read them without knowing the label set), and we skip
    #    them in the per-label loop to avoid printing them twice.
    label_keys = list(labels.keys())
    extra_lines = [
        f"{k}: %{{customdata[{i + 2}]}}"
        for i, k in enumerate(label_keys)
        if k not in ("sample_idx", "token_idx")
    ]
    hover_template = "<br>".join(
        ["sample_idx: %{customdata[0]}", "token_idx: %{customdata[1]}"] + extra_lines
    ) + "<extra></extra>"

    # 2) Build the (N, 2 + len(labels)) customdata stack. We keep all label
    #    columns (including the redundant sample_idx / token_idx) so the index
    #    offsets in `extra_lines` stay aligned with the stack.
    label_stack = np.stack([np.asarray(labels[k]) for k in label_keys], axis=1)  # (N, K)
    full_customdata = np.concatenate([customdata, label_stack], axis=1)          # (N, 2+K)

    fig = go.Figure()

    # 3) Data traces. legendgroup="color" only when symbols are also in play so
    #    the legend cleanly splits into a color section and a symbol section.
    color_lg = dict(legendgroup="color", legendgrouptitle_text=f"color: {color_label}") \
        if use_symbol else {}

    if categorical:
        # 3a) One trace per unique value -> legend entries become toggleable
        color_map = _categorical_color_map(color_vals)
        for value, color in color_map.items():
            mask = color_vals == value  # (N,)
            if not mask.any():
                continue
            marker_kw = dict(size=5, color=color, opacity=0.85)
            if use_symbol:
                marker_kw["symbol"] = point_symbols[mask]
            trace_kw = dict(
                mode="markers",
                name=str(value),
                customdata=full_customdata[mask],
                hovertemplate=hover_template,
                marker=marker_kw,
                **color_lg,
            )
            if dim == 3:
                fig.add_trace(go.Scatter3d(
                    x=emb[mask, 0], y=emb[mask, 1], z=emb[mask, 2], **trace_kw
                ))
            else:
                fig.add_trace(go.Scatter(
                    x=emb[mask, 0], y=emb[mask, 1], **trace_kw
                ))
    else:
        # 3b) Continuous: single trace with colorbar
        marker_kw = dict(
            size=5,
            color=color_vals,
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title=color_label),
            opacity=0.85,
        )
        if use_symbol:
            marker_kw["symbol"] = point_symbols
        trace_kw = dict(
            mode="markers",
            customdata=full_customdata,
            hovertemplate=hover_template,
            marker=marker_kw,
        )
        if dim == 3:
            fig.add_trace(go.Scatter3d(x=emb[:, 0], y=emb[:, 1], z=emb[:, 2], **trace_kw))
        else:
            fig.add_trace(go.Scatter(x=emb[:, 0], y=emb[:, 1], **trace_kw))

    # 3c) Symbol legend: dummy `legendonly` traces, one per symbol value, so
    #     the legend documents which shape maps to which value. Drawn in gray
    #     to keep the channel orthogonal to color.
    if use_symbol:
        for value, sym in symbol_map.items():
            sym_trace_kw = dict(
                mode="markers",
                name=str(value),
                marker=dict(symbol=sym, color="#7f7f7f", size=8),
                legendgroup="symbol",
                legendgrouptitle_text=f"symbol: {symbol_label}",
                showlegend=True,
                hoverinfo="skip",
                visible="legendonly",
            )
            if dim == 3:
                fig.add_trace(go.Scatter3d(x=[None], y=[None], z=[None], **sym_trace_kw))
            else:
                fig.add_trace(go.Scatter(x=[None], y=[None], **sym_trace_kw))

    # 4) Layout: tight margins, square aspect on 3D so rotation is intuitive
    default_title = f"colored by {color_label}" + (
        f", symbol by {symbol_label}" if use_symbol else ""
    )
    fig.update_layout(
        title=title or default_title,
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(itemsizing="constant"),
        height=600,
    )
    if dim == 3:
        fig.update_layout(scene=dict(
            xaxis_title="d1", yaxis_title="d2", zaxis_title="d3",
            aspectmode="cube",
        ))
    else:
        fig.update_xaxes(title="d1")
        fig.update_yaxes(title="d2", scaleanchor="x", scaleratio=1)

    return fig


##############################
# Input panel (click target)
##############################


def plot_input_panel(x, y, sample_idx, token_idx, tokens_per_patch=None):
    """
    Plot the context+horizon for a single clicked point.

    x:                 Tensor or ndarray (B, L_ctx, F).
    y:                 optional Tensor/ndarray (B, L_pred, F).
    sample_idx:        which batch sample to draw.
    token_idx:         token position of the clicked activation.
    tokens_per_patch:  if given, shade the slice
                       [token_idx * tokens_per_patch, (token_idx+1) * tokens_per_patch)
                       on the context axis to mark the active patch.
    """
    x_np = to_numpy(x)                          # (B, L_ctx, F)
    y_np = to_numpy(y) if y is not None else None
    L_ctx = x_np.shape[1]
    F = x_np.shape[2]

    fig = make_subplots(rows=F, cols=1, shared_xaxes=True, vertical_spacing=0.05)

    for f in range(F):
        ctx_trace = go.Scatter(
            x=np.arange(L_ctx),
            y=x_np[sample_idx, :, f],
            mode="lines",
            name=f"x[{f}]" if f == 0 else None,
            showlegend=(f == 0),
            line=dict(color="black", width=1.5),
        )
        fig.add_trace(ctx_trace, row=f + 1, col=1)

        if y_np is not None:
            L_pred = y_np.shape[1]
            horizon_trace = go.Scatter(
                x=np.arange(L_ctx, L_ctx + L_pred),
                y=y_np[sample_idx, :, f],
                mode="lines",
                name=f"y[{f}]" if f == 0 else None,
                showlegend=(f == 0),
                line=dict(color="gray", width=1.2, dash="dash"),
            )
            fig.add_trace(horizon_trace, row=f + 1, col=1)

        # 1) Highlight active patch as a shaded band iff tokens_per_patch is known.
        #    Without that hint the relationship between token_idx and time steps
        #    is layer-specific, so we only annotate token_idx in the title.
        if tokens_per_patch is not None:
            x0 = int(token_idx * tokens_per_patch)
            x1 = int((token_idx + 1) * tokens_per_patch)
            fig.add_vrect(x0=x0, x1=x1, fillcolor="orange", opacity=0.2,
                          line_width=0, row=f + 1, col=1)

    fig.update_layout(
        title=f"sample {sample_idx}, token {token_idx}",
        margin=dict(l=0, r=0, t=40, b=0),
        height=max(220, 180 * F),
    )
    return fig


##############################
# Gradio app
##############################


def _resolve_emb(state, layer, method, n_components):
    """
    Look up (or compute and cache) the (N, n_components) embedding for the
    given (layer, method, n_components) triple. Stores results on `state`.
    """
    cache = state["cache"]
    key = (layer, method, int(n_components))
    if key not in cache:
        feats = state["features"][layer]  # (N, D)
        cache[key] = reduce(feats, method, int(n_components), seed=state["seed"])
    return cache[key]


def _redraw(state, layer, method, n_components, color_label, symbol_label):
    """Rebuild the scatter figure given the current dropdown selections."""
    emb = _resolve_emb(state, layer, method, n_components)  # (N, dim)
    labels = state["labels"][layer]
    customdata = state["customdata"][layer]                 # (N, 2)
    title = f"{layer}  |  {method}  |  {n_components}D"
    return scatter_plotly(
        emb, labels, customdata, color_label, int(n_components),
        title=title, symbol_label=symbol_label,
    )


def _on_layer_change(state, layer, method, n_components, color_label, symbol_label):
    """Layer changed: recompute scatter and refresh color/symbol choices."""
    new_keys = list(state["labels"][layer].keys())
    new_color = color_label if color_label in new_keys else new_keys[0]
    # Symbol dropdown always carries an explicit "(none)" sentinel up front.
    new_symbol_choices = [SYMBOL_NONE] + new_keys
    new_symbol = symbol_label if symbol_label in new_symbol_choices else SYMBOL_NONE
    fig = _redraw(state, layer, method, n_components, new_color, new_symbol)
    return (
        fig,
        gr.update(choices=new_keys, value=new_color),
        gr.update(choices=new_symbol_choices, value=new_symbol),
    )


def _on_projection_change(state, layer, method, n_components, color_label, symbol_label):
    """Reducer or n_components changed: recompute the scatter."""
    return _redraw(state, layer, method, n_components, color_label, symbol_label)


def _on_color_change(state, layer, method, n_components, color_label, symbol_label):
    """Coloring changed: re-skin the scatter without re-fitting."""
    return _redraw(state, layer, method, n_components, color_label, symbol_label)


def _on_symbol_change(state, layer, method, n_components, color_label, symbol_label):
    """Symbol mapping changed: re-skin the scatter without re-fitting."""
    return _redraw(state, layer, method, n_components, color_label, symbol_label)


def _on_index_change(state, sample_idx, token_idx):
    """Draw the input panel for the (sample_idx, token_idx) typed by the user."""
    return plot_input_panel(
        state["x"], state["y"], int(sample_idx), int(token_idx),
        tokens_per_patch=state["tokens_per_patch"],
    )


def _on_click_bridge(state, payload):
    """
    Handle a click event forwarded from JS. `payload` is "sample,token".
    Returns (input_panel_figure, sample_idx_update, token_idx_update) so the
    manual Number inputs stay in sync with the clicked point.
    """
    if not payload:
        return gr.update(), gr.update(), gr.update()
    parts = payload.split(",")
    if len(parts) != 2:
        return gr.update(), gr.update(), gr.update()
    try:
        sample_idx = int(parts[0])
        token_idx = int(parts[1])
    except ValueError:
        return gr.update(), gr.update(), gr.update()
    fig = plot_input_panel(
        state["x"], state["y"], sample_idx, token_idx,
        tokens_per_patch=state["tokens_per_patch"],
    )
    return fig, sample_idx, token_idx


# JS attached via Blocks.load(js=...) and Plot.change(js=...).
# Responsibilities:
#   1) Find the plotly chart inside #scatter-plot and attach a plotly_click
#      listener that forwards customdata[0:2] into the #click-bridge textbox.
#   2) Re-attach whenever Gradio swaps in a new chart (layer / reducer / color
#      change all replace the inner div). MutationObserver handles this.
# Quirks worked around:
#   - Gradio's Svelte components bind to `input` events, not `change`. We
#     dispatch both for safety and the Python side listens on `.input`.
#   - Some setups need the native value setter to bypass framework intercepts
#     before the input event will register.
#   - Each step logs to console under the "[fmds-bridge]" tag so users can
#     verify wiring in devtools.
_CLICK_BRIDGE_JS = r"""
() => {
    const TAG = "[fmds-bridge]";
    const FLAG = "_fmds_click_attached";

    const log = (...args) => console.log(TAG, ...args);

    const writeBridge = (payload) => {
        const bridge = document.querySelector(
            "#click-bridge textarea, #click-bridge input"
        );
        if (!bridge) { log("bridge element not found"); return; }
        const proto = bridge.tagName === "TEXTAREA"
            ? HTMLTextAreaElement.prototype
            : HTMLInputElement.prototype;
        const setter = Object.getOwnPropertyDescriptor(proto, "value").set;
        // Toggle to empty then real value so repeat clicks always fire change
        setter.call(bridge, "");
        bridge.dispatchEvent(new Event("input", { bubbles: true }));
        bridge.dispatchEvent(new Event("change", { bubbles: true }));
        setter.call(bridge, payload);
        bridge.dispatchEvent(new Event("input", { bubbles: true }));
        bridge.dispatchEvent(new Event("change", { bubbles: true }));
        log("wrote bridge:", payload);
    };

    const attach = (plotDiv) => {
        if (!plotDiv || plotDiv[FLAG]) return false;
        if (typeof plotDiv.on !== "function") return false;
        plotDiv[FLAG] = true;
        plotDiv.on("plotly_click", (data) => {
            log("plotly_click fired");
            if (!data || !data.points || !data.points.length) return;
            const cd = data.points[0].customdata;
            if (!cd) { log("no customdata on point"); return; }
            writeBridge(cd[0] + "," + cd[1]);
        });
        log("attached plotly_click to", plotDiv);
        return true;
    };

    const tryAttach = () => {
        const container = document.getElementById("scatter-plot");
        if (!container) return false;
        // Plotly inserts a div with class js-plotly-plot; fall back to any
        // descendant that has Plotly's .on method.
        let plotDiv = container.querySelector(".js-plotly-plot");
        if (!plotDiv) {
            const candidates = container.querySelectorAll("div");
            for (const c of candidates) {
                if (typeof c.on === "function") { plotDiv = c; break; }
            }
        }
        return attach(plotDiv);
    };

    // Initial attempt
    if (!tryAttach()) log("initial attach deferred; waiting on MutationObserver");

    // Catch every later chart replacement
    if (!window._fmdsBridgeObserver) {
        window._fmdsBridgeObserver = new MutationObserver(tryAttach);
        window._fmdsBridgeObserver.observe(
            document.body, { childList: true, subtree: true }
        );
        log("MutationObserver installed");
    }
}
"""


def build_activation_viewer_app(activations, x, y=None, labels=None,
                                labels_per_layer=None,
                                tokens_per_patch=None, seed=0):
    """
    Assemble the Gradio Blocks app for interactive activation exploration.

    activations:       dict[str, Tensor (T, B, D)]. One entry per hooked layer.
    x:                 Tensor (B, L_ctx, F).
    y:                 optional Tensor (B, L_pred, F).
    labels:            optional dict[str, array]; broadcast via broadcast_labels.
                       Same dict is applied to every layer (per-sample concept
                       labels are shared across depths).
    labels_per_layer:  optional dict[layer_name, dict[str, array]] of per-layer
                       label overrides. Useful for labels whose shape depends on
                       the layer (e.g. per-layer cluster ids of shape (T, B)).
                       Keys here overwrite same-named keys from `labels`.
    tokens_per_patch:  optional int hint for the input panel highlight band.
    seed:              random_state passed to TSNE / MDS / UMAP.

    Returns: gradio.Blocks. App layout:
        Left panel:
          - Layer dropdown
          - Reducer dropdown
          - n_components radio (2 / 3)
          - Color-by dropdown
          - Sample idx / Token idx number inputs (manual fallback)
        Center: scatter plot (rotatable in 3D, click any point).
        Right:  input time series for the most recently selected point.

    Behavior: switching layer / reducer / n_components recomputes the
    embedding (memoized on a gr.State). Switching color-by only re-skins the
    existing scatter. Clicking a point updates the input panel and syncs the
    sample / token inputs; editing those inputs also updates the panel.

    Honors FMDS_SMOKE=1: builds the Blocks but skips the inner setup that
    would touch heavy compute; the caller is still responsible for not
    calling `.launch()` in smoke mode.
    """
    layer_names = list(activations.keys())
    if not layer_names:
        raise ValueError("activations dict is empty")

    # 1) Flatten every layer once; cache features + per-point labels + customdata
    features_by_layer = {}    # layer -> (N, D)
    labels_by_layer = {}      # layer -> dict[str, (N,)]
    customdata_by_layer = {}  # layer -> (N, 2)
    per_layer = labels_per_layer or {}
    for name in layer_names:
        feats, s_idx, t_idx = flatten_activations(activations[name])  # (N, D), (N,), (N,)
        T, B = activations[name].shape[0], activations[name].shape[1]
        features_by_layer[name] = feats
        # 1a) Start from the global labels broadcast at this layer's (T, B);
        #     merge per-layer overrides on top so layer-specific keys win.
        merged = broadcast_labels(labels, T, B)
        if name in per_layer:
            extra = broadcast_labels(per_layer[name], T, B)
            # broadcast_labels always adds sample_idx / token_idx -- skip those
            # synthetics when merging so we don't double-write.
            for k, v in extra.items():
                if k in ("sample_idx", "token_idx"):
                    continue
                merged[k] = v
        labels_by_layer[name] = merged
        customdata_by_layer[name] = np.stack([s_idx, t_idx], axis=1)  # (N, 2)

    # 2) Initial selections
    initial_layer = layer_names[0]
    initial_method = "pca"
    initial_n_components = 3
    label_keys = list(labels_by_layer[initial_layer].keys())
    initial_color = "sample_idx" if "sample_idx" in label_keys else label_keys[0]
    initial_symbol = SYMBOL_NONE
    symbol_choices = [SYMBOL_NONE] + label_keys

    initial_state = {
        "x": x,
        "y": y,
        "tokens_per_patch": tokens_per_patch,
        "seed": seed,
        "features": features_by_layer,
        "labels": labels_by_layer,
        "customdata": customdata_by_layer,
        "cache": {},
    }

    initial_fig = _redraw(
        initial_state, initial_layer, initial_method,
        initial_n_components, initial_color, initial_symbol,
    )
    initial_panel = plot_input_panel(
        x, y, sample_idx=0, token_idx=0, tokens_per_patch=tokens_per_patch,
    )

    ##############################
    # Block layout
    ##############################
    with gr.Blocks(title="Activation viewer") as app:
        state = gr.State(initial_state)
        # Text channel that the JS bridge writes into on plotly_click.
        # Kept visible and read-only so users can see the bridge ticking;
        # set `visible=False` once you have confirmed clicks are wired.
        click_bridge = gr.Textbox(
            value="",
            elem_id="click-bridge",
            label="Click bridge (read-only, debug)",
            visible=True,
            interactive=False,
            max_lines=1,
        )

        with gr.Row():
            with gr.Column(scale=1):
                layer_dd = gr.Dropdown(
                    choices=layer_names, value=initial_layer, label="Layer",
                )
                method_dd = gr.Dropdown(
                    choices=REDUCER_CHOICES, value=initial_method, label="Reducer",
                )
                n_comp_radio = gr.Radio(
                    choices=[2, 3], value=initial_n_components, label="Components",
                )
                color_dd = gr.Dropdown(
                    choices=label_keys, value=initial_color, label="Color by",
                )
                symbol_dd = gr.Dropdown(
                    choices=symbol_choices, value=initial_symbol, label="Symbol by",
                )
                gr.Markdown("Click any point on the scatter, or set indices manually:")
                sample_in = gr.Number(value=0, precision=0, label="Sample idx")
                token_in = gr.Number(value=0, precision=0, label="Token idx")
            with gr.Column(scale=3):
                scatter = gr.Plot(
                    value=initial_fig, label="Activations", elem_id="scatter-plot",
                )
            with gr.Column(scale=2):
                input_panel = gr.Plot(value=initial_panel, label="Input (sample / token)")

        ##############################
        # Wiring
        ##############################
        layer_dd.change(
            fn=_on_layer_change,
            inputs=[state, layer_dd, method_dd, n_comp_radio, color_dd, symbol_dd],
            outputs=[scatter, color_dd, symbol_dd],
        )
        method_dd.change(
            fn=_on_projection_change,
            inputs=[state, layer_dd, method_dd, n_comp_radio, color_dd, symbol_dd],
            outputs=[scatter],
        )
        n_comp_radio.change(
            fn=_on_projection_change,
            inputs=[state, layer_dd, method_dd, n_comp_radio, color_dd, symbol_dd],
            outputs=[scatter],
        )
        color_dd.change(
            fn=_on_color_change,
            inputs=[state, layer_dd, method_dd, n_comp_radio, color_dd, symbol_dd],
            outputs=[scatter],
        )
        symbol_dd.change(
            fn=_on_symbol_change,
            inputs=[state, layer_dd, method_dd, n_comp_radio, color_dd, symbol_dd],
            outputs=[scatter],
        )
        sample_in.change(
            fn=_on_index_change,
            inputs=[state, sample_in, token_in],
            outputs=[input_panel],
        )
        token_in.change(
            fn=_on_index_change,
            inputs=[state, sample_in, token_in],
            outputs=[input_panel],
        )
        # Bridge fires `.input` reliably on programmatic value updates;
        # `.change` is bound for paranoia in case some Gradio backends only
        # raise `change`. _on_click_bridge is idempotent.
        click_bridge.input(
            fn=_on_click_bridge,
            inputs=[state, click_bridge],
            outputs=[input_panel, sample_in, token_in],
        )
        click_bridge.change(
            fn=_on_click_bridge,
            inputs=[state, click_bridge],
            outputs=[input_panel, sample_in, token_in],
        )

        # Install the plotly_click bridge twice:
        #   - on page load (initial render)
        #   - after every scatter update (Gradio swaps the chart div, so we
        #     need to re-attach -- MutationObserver in the JS catches it but
        #     this is the belt-and-braces).
        app.load(fn=None, inputs=None, outputs=None, js=_CLICK_BRIDGE_JS)
        scatter.change(fn=None, inputs=None, outputs=None, js=_CLICK_BRIDGE_JS)

    return app


##############################
# Persistence
##############################


_SESSION_VERSION = 2


def _arr(v):
    # 1) Torch tensor -> ndarray (drop grad / device); pass through ndarrays
    if hasattr(v, "detach") and hasattr(v, "cpu"):
        return v.detach().cpu().numpy()
    return np.asarray(v)


def save_session(path, activations, x, y=None, labels=None,
                 labels_per_layer=None, tokens_per_patch=None):
    """
    Persist all viewer inputs to a single compressed `.npz` file.

    Layout (no pickle, safe to load anywhere):
      - act_0, act_1, ...                       one array per layer, shape (T, B, D)
      - x                                       (B, L_ctx, F)
      - y                                       (B, L_pred, F), only if provided
      - label_0, label_1, ...                   one array per global label key
      - playerlabel_{li}_{ki}                   per-layer label arrays
      - __meta__                                JSON blob (uint8 bytes) with
                                                layer names, label names,
                                                layer_label_keys (per-layer key
                                                list), tokens_per_patch, has_y,
                                                format version.

    Round-trip via `load_session(path)` then
    `build_activation_viewer_app(**load_session(path))`.
    """
    arrays = {}
    layer_names = list(activations.keys())
    for i, name in enumerate(layer_names):
        arrays[f"act_{i}"] = _arr(activations[name])  # (T, B, D)

    arrays["x"] = _arr(x)  # (B, L_ctx, F)
    if y is not None:
        arrays["y"] = _arr(y)  # (B, L_pred, F)

    # 1) Global labels: one entry per non-synthetic key
    label_keys = []
    if labels:
        for k, v in labels.items():
            if k in ("sample_idx", "token_idx"):
                continue
            label_keys.append(k)
            arrays[f"label_{len(label_keys) - 1}"] = np.asarray(v)

    # 2) Per-layer labels: stored under playerlabel_{layer_idx}_{key_idx}.
    #    Track keys per layer in meta so load_session can rebuild the dict.
    layer_label_keys = {}
    if labels_per_layer:
        layer_index = {name: i for i, name in enumerate(layer_names)}
        for lname, ldict in labels_per_layer.items():
            if lname not in layer_index or not ldict:
                continue
            li = layer_index[lname]
            keys_for_layer = []
            for k, v in ldict.items():
                if k in ("sample_idx", "token_idx"):
                    continue
                keys_for_layer.append(k)
                arrays[f"playerlabel_{li}_{len(keys_for_layer) - 1}"] = np.asarray(v)
            if keys_for_layer:
                layer_label_keys[lname] = keys_for_layer

    meta = {
        "version": _SESSION_VERSION,
        "layers": layer_names,
        "label_keys": label_keys,
        "layer_label_keys": layer_label_keys,
        "has_y": y is not None,
        "tokens_per_patch": tokens_per_patch,
    }
    arrays["__meta__"] = np.frombuffer(
        json.dumps(meta).encode("utf-8"), dtype=np.uint8
    )

    np.savez_compressed(str(path), **arrays)


def load_session(path):
    """
    Inverse of `save_session`. Returns a dict of kwargs ready to splat into
    `build_activation_viewer_app`. Accepts both version 1 (no per-layer
    labels) and version 2 sessions.
    """
    z = np.load(str(path), allow_pickle=False)
    # 1) Decode meta blob first; everything else hangs off it
    meta = json.loads(bytes(z["__meta__"]).decode("utf-8"))
    version = meta.get("version")
    if version not in (1, _SESSION_VERSION):
        raise ValueError(
            f"session format version {version} not supported "
            f"(expected 1 or {_SESSION_VERSION})"
        )

    activations = {name: z[f"act_{i}"] for i, name in enumerate(meta["layers"])}
    x = z["x"]
    y = z["y"] if meta["has_y"] else None
    labels = (
        {k: z[f"label_{i}"] for i, k in enumerate(meta["label_keys"])}
        if meta["label_keys"]
        else None
    )

    # 2) Per-layer labels: only present in v2 sessions
    labels_per_layer = None
    layer_label_keys = meta.get("layer_label_keys") or {}
    if layer_label_keys:
        layer_index = {name: i for i, name in enumerate(meta["layers"])}
        labels_per_layer = {}
        for lname, keys in layer_label_keys.items():
            li = layer_index[lname]
            labels_per_layer[lname] = {
                k: z[f"playerlabel_{li}_{ki}"] for ki, k in enumerate(keys)
            }

    return {
        "activations": activations,
        "x": x,
        "y": y,
        "labels": labels,
        "labels_per_layer": labels_per_layer,
        "tokens_per_patch": meta["tokens_per_patch"],
    }


def save_scatter_html(path, activations, layer, method, n_components, color_label,
                      labels=None, labels_per_layer=None, seed=0, title=None,
                      symbol_label=None):
    """
    Export a standalone Plotly HTML for one (layer, method, n_components,
    color_label[, symbol_label]) projection. The resulting file opens in any
    browser and preserves rotation, hover, and zoom; click-to-input is not
    available without the Gradio server.

    `labels_per_layer[layer]`, if provided, is merged on top of `labels` for
    the rendered layer (matches the viewer's merge semantics).
    `symbol_label`, if provided, maps a second categorical feature to marker
    symbols (see `scatter_plotly`).
    """
    feats, sid, tid = flatten_activations(activations[layer])      # (N, D), (N,), (N,)
    T, B = activations[layer].shape[0], activations[layer].shape[1]
    broadcast = broadcast_labels(labels, T, B)
    if labels_per_layer and layer in labels_per_layer:
        extra = broadcast_labels(labels_per_layer[layer], T, B)
        for k, v in extra.items():
            if k in ("sample_idx", "token_idx"):
                continue
            broadcast[k] = v
    emb = reduce(feats, method, n_components, seed=seed)           # (N, n_components)
    customdata = np.stack([sid, tid], axis=1)                      # (N, 2)
    fig = scatter_plotly(
        emb, broadcast, customdata, color_label, n_components,
        title=title or f"{layer}  |  {method}  |  {n_components}D",
        symbol_label=symbol_label,
    )
    fig.write_html(str(path), include_plotlyjs="cdn")


def is_smoke_mode():
    """True iff FMDS_SMOKE=1, matching the convention in examples/."""
    return os.environ.get("FMDS_SMOKE", "") == "1"
