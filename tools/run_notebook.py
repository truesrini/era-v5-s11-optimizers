"""Build and execute the notebook from its percent-format source, saving every output.

The notebook source lives in optimizers_and_schedules.py (jupytext "percent" format:
`# %%` starts a code cell, `# %% [markdown]` a markdown cell). This script runs each
code cell in one shared namespace, captures printed text and every matplotlib figure
shown with plt.show(), and writes optimizers_and_schedules.ipynb with those outputs.

It exists because the Python that has CUDA PyTorch here has no Jupyter kernel installed.
The resulting .ipynb is an ordinary notebook and re-runs unchanged in Jupyter or VS Code.

    python tools/run_notebook.py
"""
import base64, io, json, os, re, sys, time, traceback

SRC, OUT = "optimizers_and_schedules.py", "optimizers_and_schedules.ipynb"


def parse_cells(path):
    cells, kind, buf = [], None, []
    for line in open(path, encoding="utf-8").read().splitlines():
        m = re.match(r"^# %%( \[markdown\])?\s*$", line)
        if m:
            if kind: cells.append((kind, buf))
            kind, buf = ("markdown" if m.group(1) else "code"), []
        elif kind:
            buf.append(line)
    if kind: cells.append((kind, buf))
    out = []
    for kind, lines in cells:
        while lines and not lines[-1].strip(): lines.pop()
        while lines and not lines[0].strip(): lines.pop(0)
        if kind == "markdown":
            lines = [l[2:] if l.startswith("# ") else l[1:] if l.startswith("#") else l for l in lines]
        out.append((kind, "\n".join(lines)))
    return out


class Tee(io.TextIOBase):
    def __init__(self, sink): self.buf, self.sink = [], sink
    def write(self, s):
        self.buf.append(s); self.sink.write(s); self.sink.flush(); return len(s)
    def flush(self): self.sink.flush()


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outputs = []           # outputs of the cell currently running
    tee = Tee(sys.__stdout__)

    def flush_text():
        text = "".join(tee.buf); tee.buf.clear()
        if text: outputs.append({"output_type": "stream", "name": "stdout", "text": text.splitlines(keepends=True)})

    def show(*a, **k):
        flush_text()
        for num in plt.get_fignums():
            fig = plt.figure(num); b = io.BytesIO()
            fig.savefig(b, format="png", dpi=100, bbox_inches="tight")
            outputs.append({"output_type": "display_data", "metadata": {},
                            "data": {"image/png": base64.b64encode(b.getvalue()).decode(), "text/plain": [repr(fig)]}})
        plt.close("all")
    plt.show = show

    cells, nb_cells, ns, count = parse_cells(SRC), [], {"__name__": "__main__"}, 0
    t_all = time.time()
    for kind, src in cells:
        if kind == "markdown":
            nb_cells.append({"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}); continue
        count += 1; outputs = []
        print(f"\n----- cell {count} -----", file=sys.__stderr__); t = time.time()
        sys.stdout = tee
        try:
            exec(compile(src, f"<cell {count}>", "exec"), ns)
            failed = None
        except Exception as e:
            failed = e
        finally:
            sys.stdout = sys.__stdout__
        flush_text(); show()
        if failed:
            tb = traceback.format_exception(type(failed), failed, failed.__traceback__)
            outputs.append({"output_type": "error", "ename": type(failed).__name__, "evalue": str(failed), "traceback": tb})
        nb_cells.append({"cell_type": "code", "execution_count": count, "metadata": {}, "outputs": outputs,
                         "source": src.splitlines(keepends=True)})
        print(f"----- cell {count} done in {time.time() - t:.1f}s -----", file=sys.__stderr__)
        if failed:
            print("".join(tb), file=sys.__stderr__); break
    nb = {"cells": nb_cells, "nbformat": 4, "nbformat_minor": 5,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python", "version": sys.version.split()[0]}}}
    json.dump(nb, open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"wrote {OUT} ({count} code cells, {time.time() - t_all:.0f}s)", file=sys.__stderr__)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    main()
