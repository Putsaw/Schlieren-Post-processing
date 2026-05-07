import argparse
import os
import re
import math
import pandas as pd
import matplotlib.pyplot as plt

def _safe_name(s: str) -> str:
    # create a filesystem-safe short name for a column
    return re.sub(r'[^0-9A-Za-z_-]+', '_', s).strip('_')

def main(csv_path, out_path=None):
    # read CSV, ignore any leading comment lines that start with '/'
    df = pd.read_csv(csv_path, comment='/', engine='python')
    # coerce numeric columns (strings like "nan" become NaN)
    df = df.apply(pd.to_numeric, errors='coerce')

    x = 'Frame'
    if x not in df.columns:
        raise SystemExit(f"CSV missing expected column '{x}'")

    # Plot each numeric metric (except Frame) as subplots on a single figure
    numeric_cols = [c for c in df.select_dtypes(include='number').columns if c != x]
    if not numeric_cols:
        raise SystemExit("No numeric columns found to plot. Columns found:\n" + ", ".join(df.columns))

    base = os.path.splitext(csv_path)[0]
    # if out_path is provided and is a directory, use it as output directory
    out_dir = None
    out_prefix = None
    if out_path:
        if os.path.isdir(out_path):
            out_dir = out_path
        else:
            # treat as prefix (without extension)
            out_prefix = os.path.splitext(out_path)[0]

    # choose layout: 1 column for up to 6 plots, otherwise 2 columns
    n = len(numeric_cols)
    ncols = 3
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 2.5 * nrows), sharex=True)
    # ensure axes is a flat list for easy indexing
    if isinstance(axes, plt.Axes):  # type: ignore
        axes = [axes]
    else:
        axes = list(axes.flatten())

    for i, col in enumerate(numeric_cols):
        ax = axes[i]
        ax.plot(df[x], df[col], marker='.', linestyle='-', markersize=3)
        ax.set_ylabel(col)
        ax.grid(True)
        ax.set_title(col)

    # hide any unused subplots
    for j in range(n, len(axes)):
        axes[j].axis('off')

    # label x only on the bottom-most subplots
    for ax in axes[-ncols:]:
        ax.set_xlabel(x)

    fig.suptitle(os.path.basename(csv_path))
    fig.tight_layout(rect=[0, 0, 1, 0.96]) # type: ignore

    # determine output filename
    if out_dir:
        out_file = os.path.join(out_dir, f"{os.path.basename(base)}_all_metrics.png")
    elif out_prefix:
        out_file = f"{out_prefix}_all_metrics.png"
    else:
        out_file = f"{base}_all_metrics.png"

    fig.savefig(out_file, dpi=150)
    print(f"Saved combined plot to: {out_file}")
    plt.show()

    return [out_file]

if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description="Plot metrics CSV (one separate graph per numeric metric). If CSV path is omitted, a file dialog will open.")
    p.add_argument('csv', nargs='?', help='path to CSV file (optional; opens file browser if omitted)')
    p.add_argument('--out', help='optional output directory or output filename prefix')
    args = p.parse_args()

    csv_path = args.csv
    if csv_path is None:
        # open a file dialog to let the user pick the CSV
        try:
            import tkinter as tk
            from tkinter import filedialog
        except Exception:
            raise SystemExit("No CSV provided and tkinter is not available to browse files.")
        root = tk.Tk()
        root.withdraw()
        csv_path = filedialog.askopenfilename(
            title="Select CSV file",
            initialdir=os.getcwd(),
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        root.destroy()
        if not csv_path:
            raise SystemExit("No file selected. Exiting.")

    main(csv_path, args.out)