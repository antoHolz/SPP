"""
Activation extraction utilities for XAI workflows.

Exposes a small API that takes a `model.predict(x=...)` style model and returns
intermediate activations from selected leaf modules:

  - getLayers(module, layer_names=None) -> dict[str, nn.Module]
        Recursively walks `module` and returns its leaf submodules. If
        `layer_names` is given, filters to those exact dotted names.
  - getActivations(model, layers, x) -> (dict[str, Tensor], outputs)
        Registers forward hooks on `layers.values()`, runs `model.predict(x=x)`,
        removes the hooks, and returns the captured activations plus the
        model outputs.

Shape contract for captured activations
---------------------------------------
Activations are stored as the raw `output` of each hooked module. For
transformer-style backbones used in this repo, that is typically a tensor
shaped (T, B, D), where:
    T -- token / sequence position axis
    B -- batch axis
    D -- feature / hidden axis

This file also provides shape helpers that consume that (T, B, D) layout:

  - flatten_activations(act) -- collapse to (T*B, D) in token-major order
  - broadcast_labels(labels, T, B) -- broadcast per-sample or per-token-sample
    labels to the (T*B,) row layout produced by flatten_activations

Both helpers stay here (next to the extraction code) so the whole
activation API lives in one short file.
"""

import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


##############################
# Layer enumeration
##############################


def getLayers(module, name="model", layer_names=None):
    """
    Recursively collect leaf submodules of `module` into a flat dict.

    Reads:   any nn.Module.
    Returns: dict mapping dotted leaf name ("model.encoder.block.0...") to
             the corresponding nn.Module instance.
    Notes:   if `layer_names` is given, the returned dict is restricted to
             that exact set of names (KeyError if a requested name is not a
             leaf of `module`).
    """
    layers = {}
    for i in module.named_children():
        if dict(getattr(module, i[0]).named_children()) == {}:
            layers[name + "." + i[0]] = getattr(module, i[0])
            continue
        else:
            layers.update(getLayers(getattr(module, i[0]), name=name + "." + i[0]))

    if layer_names is None:
        layer_names = []
    if layer_names:
        layers = dict((k, layers[k]) for k in layer_names)
    return layers


def nested_hook(layer_id, activation_store):
    """
    Build a forward hook that stores `output` into `activation_store[layer_id]`.
    Returned closure is registered via `module.register_forward_hook(...)`.
    """
    def myhook(module, input, output):
        activation_store[layer_id] = output
    return myhook


def getActivations(model, layers, x):
    """
    Run a forward pass with hooks attached to every module in `layers.values()`
    and return what they captured.

    Reads:   model with a `predict(x=...)` method, `layers` dict (name -> module)
             as produced by getLayers, input tensor `x`.
    Returns: (activation_store, outputs)
             activation_store: dict[str, Tensor], one entry per hooked layer,
                               typically shaped (T, B, D).
             outputs:          whatever the predict call returned.
    Notes:   the forward uses `model.predict_grad` when the model defines it
             (falling back to `model.predict`). predict_grad keeps the autograd
             graph, so captured activations stay differentiable: a caller can
             build a scalar from `outputs` and call torch.autograd.grad against
             an activation. Callers that do not need gradients can `.detach()`.
    """
    activation_store = {}
    handles = []
    # 1) Attach a forward hook to each requested layer
    for key, value in layers.items():
        handles.append(value.register_forward_hook(nested_hook(key, activation_store)))

    # 2) Forward pass; prefer the differentiable predict_grad path so captured
    #    activations keep their autograd graph (predict runs under no_grad)
    predict_fn = getattr(model, "predict_grad", model.predict)
    outputs = predict_fn(x=x)  # activation_store entries get shape (T, B, D)

    # 3) Drop hooks so the model stays clean for downstream calls
    for handle in handles:
        handle.remove()
    return activation_store, outputs


##############################
# Activation shape helpers
##############################


def _to_numpy(arr):
    # 1) Tensor -> ndarray without grad / device baggage
    if HAS_TORCH and isinstance(arr, torch.Tensor):
        return arr.detach().cpu().numpy()
    return np.asarray(arr)


def flatten_activations(act):
    """
    Collapse a (T, B, D) activation tensor to (T*B, D) in token-major order.

    Reads:   tensor or ndarray with shape (T, B, D).
    Returns: features    -- ndarray (T*B, D)
             sample_idx  -- ndarray (T*B,) of ints in [0, B)
             token_idx   -- ndarray (T*B,) of ints in [0, T)
    Layout:  row r corresponds to (t=r // B, b=r % B). Click callbacks rely
             on this exact ordering to map a scatter point back to its
             (token, sample) pair.
    """
    arr = _to_numpy(act)  # (T, B, D)
    if arr.ndim != 3:
        raise ValueError(f"flatten_activations expects (T, B, D); got shape {arr.shape}")

    T, B, D = arr.shape
    # 1) Reshape token-major: (T, B, D) -> (T*B, D), row r = t*B + b
    features = arr.reshape(T * B, D)  # (T*B, D)

    # 2) Build matching index columns
    t_grid, b_grid = np.meshgrid(np.arange(T), np.arange(B), indexing="ij")  # (T, B), (T, B)
    token_idx = t_grid.reshape(T * B)   # (T*B,)
    sample_idx = b_grid.reshape(T * B)  # (T*B,)

    return features, sample_idx, token_idx


def broadcast_labels(labels, T, B):
    """
    Broadcast each entry of `labels` to (T*B,) matching the token-major
    layout of flatten_activations.

    Accepted per-label shapes:
        (B,) or (B, 1)   -- per-sample, tiled across the T tokens
        (T, B)           -- per-token-per-sample, flattened directly
        (T*B,)           -- already flat, kept as-is
    Anything else raises ValueError.

    Returns: dict[str, ndarray (T*B,)]. Always includes synthetic keys
             'sample_idx' and 'token_idx' (added last so user keys win on
             name clashes only if explicitly overridden upstream).
    """
    out = {}
    if labels:
        for key, val in labels.items():
            arr = _to_numpy(val)
            # 1) Strip trailing singleton dim so (B, 1) becomes (B,)
            if arr.ndim == 2 and arr.shape[1] == 1:
                arr = arr[:, 0]

            # 2) Dispatch on shape
            if arr.shape == (B,):
                tiled = np.broadcast_to(arr[None, :], (T, B)).reshape(T * B)  # (T*B,)
            elif arr.shape == (T, B):
                tiled = arr.reshape(T * B)  # (T*B,)
            elif arr.shape == (T * B,):
                tiled = arr  # (T*B,)
            else:
                raise ValueError(
                    f"label '{key}' has shape {arr.shape}; expected "
                    f"(B={B},), (T={T}, B={B}), or (T*B={T * B},)"
                )
            out[key] = tiled

    # 3) Synthetic per-point indices, always available
    t_grid, b_grid = np.meshgrid(np.arange(T), np.arange(B), indexing="ij")  # (T, B), (T, B)
    out["sample_idx"] = b_grid.reshape(T * B)  # (T*B,)
    out["token_idx"] = t_grid.reshape(T * B)   # (T*B,)
    return out
