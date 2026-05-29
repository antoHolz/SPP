# Windows native-library load-order guard.
#
# On Windows, importing `lightning`/`torch` (Intel OpenMP, libiomp5md.dll) *before*
# scikit-learn / scipy (their own OpenMP + OpenBLAS) corrupts the process heap and
# crashes the interpreter with STATUS_HEAP_CORRUPTION (exit 0xC0000374) -- a silent,
# traceback-less death. Importing scikit-learn first makes the load order safe.
#
# Since every entry point reaches the training stack via `from modules... import ...`,
# this package __init__ runs before `lightning` is ever imported, so preloading sklearn
# here fixes it centrally. Harmless on non-Windows / when sklearn is already loaded.
try:  # pragma: no cover - best-effort guard
    import sklearn.cluster  # noqa: F401
except Exception:
    pass
