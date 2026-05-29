from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


##############################
# Color palettes
##############################

# Canonical model color lists — single source of truth.
# Matplotlib named colors for static plots.
MODEL_COLORS_MPL = [
    "tab:blue", "tab:orange", "tab:green", "tab:purple", "tab:brown",
    "tab:pink", "tab:gray", "tab:olive", "tab:cyan",
]

# Hex equivalents for Plotly.
MODEL_COLORS_HEX = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b",
    "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


##############################
# Path sanitization
##############################


def safe_name(name):
    # 1) Replace path separators for filesystem-safe directory names
    return name.replace("/", "_").replace("\\", "_")


##############################
# Data loading and filtering
##############################


def resolve_data(data, read_fn=pd.read_csv):
    # 1) Accept path (str/Path), DataFrame, or None
    if data is None:
        return None
    if isinstance(data, (str, Path)):
        return read_fn(data)
    return data


def apply_filters(df, include_filter=None, exclude_filter=None):
    # 1) Whitelist filter: keep only rows matching specified values per column
    if include_filter:
        for col, values in include_filter.items():
            if values and col in df.columns:
                df = df[df[col].isin(list(values))]
    # 2) Blacklist filter: exclude rows matching specified values per column
    if exclude_filter:
        for col, values in exclude_filter.items():
            if values and col in df.columns:
                df = df[~df[col].isin(list(values))]
    return df


def sort_data(df, sort_by=None):
    # 1) Sort DataFrame by specified column(s), no-op if sort_by is None
    if df is None or sort_by is None:
        return df
    return df.sort_values(sort_by).reset_index(drop=True)


##############################
# Normalization
##############################


def normalize_columns(df, columns=None, fill_value=0.5):
    # 1) Min-max normalize specified columns to [0, 1]
    #    Constant columns get fill_value (default 0.5)
    result = df.copy()
    cols = columns if columns is not None else result.columns
    for col in cols:
        col_min, col_max = result[col].min(), result[col].max()
        if col_max - col_min > 0:
            result[col] = (result[col] - col_min) / (col_max - col_min)
        else:
            result[col] = fill_value
    return result


##############################
# Figure saving
##############################


def save_and_close(fig, path, dpi=150):
    # 1) Save figure with tight bounding box, then close to free memory
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


##############################
# PNG stitching (combined dashboards)
##############################


def stitch_dataset_pngs(png_paths, out_path, pad_color=(255, 255, 255),
                        labels=None, label_fontsize=18):
    # 1) Stack a list of per-dataset PNGs vertically into one combined PNG.
    #    Much cheaper than re-rendering every subplot through matplotlib.
    from PIL import Image, ImageDraw, ImageFont

    valid = [Path(p) for p in png_paths if p is not None and Path(p).exists()]
    if not valid:
        return None

    # 2) Open, align on common width (left-aligned, pad right with pad_color)
    images = [Image.open(p).convert("RGB") for p in valid]
    common_width = max(img.width for img in images)
    padded = []
    for img in images:
        if img.width == common_width:
            padded.append(img)
            continue
        canvas = Image.new("RGB", (common_width, img.height), pad_color)
        canvas.paste(img, (0, 0))
        padded.append(canvas)

    # 3) Optional row labels (one per image) rendered above each panel
    label_band = 0
    font = None
    if labels is not None:
        try:
            font = ImageFont.truetype("arial.ttf", label_fontsize)
        except (OSError, IOError):
            font = ImageFont.load_default()
        label_band = int(label_fontsize * 1.8)

    total_height = sum(img.height + label_band for img in padded)
    combined = Image.new("RGB", (common_width, total_height), pad_color)

    y = 0
    for i, img in enumerate(padded):
        if label_band and labels is not None:
            draw = ImageDraw.Draw(combined)
            text = str(labels[i]) if i < len(labels) else ""
            draw.text((10, y + 4), text, fill=(0, 0, 0), font=font)
            y += label_band
        combined.paste(img, (0, y))
        y += img.height

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    combined.save(out_path)
    return out_path
