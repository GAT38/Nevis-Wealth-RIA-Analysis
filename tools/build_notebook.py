"""
build_notebook.py — turn a "percent-format" Python script (cells separated by
`# %%` / `# %% [markdown]`) into a Jupyter notebook and execute it, storing
real cell outputs (stdout, DataFrame HTML, matplotlib PNGs).

Why this exists: the analysis was developed in an environment without a
Jupyter kernel. This runner executes every code cell top-to-bottom in one
namespace, exactly like "Restart & Run All", and writes a standard nbformat-4
.ipynb. Anyone with Jupyter can re-run the resulting notebook unchanged.

Usage:  python tools/build_notebook.py notebooks/ria_analysis.py notebooks/ria_analysis.ipynb
"""
from __future__ import annotations

import ast
import base64
import contextlib
import io
import json
import os
import sys
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def split_cells(src: str):
    cells, kind, buf = [], None, []
    for line in src.splitlines():
        if line.startswith("# %%"):
            if kind is not None:
                cells.append((kind, "\n".join(buf).strip("\n")))
            kind = "markdown" if "[markdown]" in line else "code"
            buf = []
        else:
            buf.append(line)
    if kind is not None:
        cells.append((kind, "\n".join(buf).strip("\n")))
    out = []
    for k, body in cells:
        if k == "markdown":  # strip leading "# " comment markers
            body = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in body.splitlines())
        out.append((k, body))
    return out


def _display_obj(obj):
    data = {"text/plain": repr(obj)}
    if hasattr(obj, "_repr_html_"):
        try:
            data["text/html"] = obj._repr_html_()
        except Exception:  # noqa: BLE001
            pass
    return {"output_type": "execute_result", "data": data, "metadata": {}}


def run_cell(code: str, ns: dict, exec_count: int):
    outputs = []
    stdout = io.StringIO()

    def display(*objs):  # minimal IPython.display.display
        for o in objs:
            out = _display_obj(o)
            out["output_type"] = "display_data"
            flush()
            outputs.append(out)

    def flush():
        txt = stdout.getvalue()
        if txt:
            outputs.append({"output_type": "stream", "name": "stdout", "text": txt})
            stdout.seek(0); stdout.truncate()

    ns["display"] = display
    tree = ast.parse(code)
    last_expr = None
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        last_expr = ast.Expression(tree.body.pop().value)
    try:
        with contextlib.redirect_stdout(stdout):
            exec(compile(tree, f"<cell {exec_count}>", "exec"), ns)
            result = eval(compile(last_expr, f"<cell {exec_count}>", "eval"), ns) if last_expr else None
        flush()
        for num in plt.get_fignums():
            fig = plt.figure(num)
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
            outputs.append({"output_type": "display_data", "metadata": {},
                            "data": {"image/png": base64.b64encode(buf.getvalue()).decode(),
                                     "text/plain": "<Figure>"}})
        plt.close("all")
        if result is not None:
            r = _display_obj(result)
            r["execution_count"] = exec_count
            outputs.append(r)
    except Exception as e:  # noqa: BLE001
        flush()
        outputs.append({"output_type": "error", "ename": type(e).__name__, "evalue": str(e),
                        "traceback": traceback.format_exc().splitlines()})
        raise
    return outputs


def main(src_path: str, out_path: str):
    src = Path(src_path).read_text()
    out_path = str(Path(out_path).resolve())
    nb_dir = Path(out_path).resolve().parent
    os.chdir(nb_dir)                     # cells run with the notebook folder as cwd
    plt.show = lambda *a, **k: None      # figures are captured after each cell
    ns: dict = {"__name__": "__main__"}
    cells, n = [], 0
    for kind, body in split_cells(src):
        if kind == "markdown":
            cells.append({"cell_type": "markdown", "metadata": {}, "source": body})
            continue
        n += 1
        print(f"running cell {n} ...", file=sys.stderr)
        outs = run_cell(body, ns, n)
        cells.append({"cell_type": "code", "execution_count": n, "metadata": {},
                      "outputs": outs, "source": body})
    nb = {"cells": cells, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": sys.version.split()[0]}},
        "nbformat": 4, "nbformat_minor": 5}
    for c in nb["cells"]:  # nbformat stores source as a list of lines
        c["source"] = [l + "\n" for l in c["source"].split("\n")]
        c["source"][-1] = c["source"][-1].rstrip("\n")
        c["id"] = f"cell-{nb['cells'].index(c)}"
    Path(out_path).write_text(json.dumps(nb, indent=1))
    print(f"wrote {out_path} ({n} code cells)", file=sys.stderr)


if __name__ == "__main__":
    main(*sys.argv[1:3])
