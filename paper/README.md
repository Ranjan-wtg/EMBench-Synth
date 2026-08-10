# Paper build

`paper.tex` is the canonical manuscript source. `paper.md` is the synchronized
plain-text fallback for review and version control.

The figures used by the manuscript are stored in `paper/figures/`. The primary
new result is the deployment-safety certificate:

```bash
python benchmarks/deployment_safety_certificate.py
```

To build the PDF after installing a LaTeX distribution with XeLaTeX support:

```bash
cd paper
xelatex -interaction=nonstopmode paper.tex
xelatex -interaction=nonstopmode paper.tex
```

The generated PDF is intentionally not tracked until it is rebuilt from the
current `paper.tex`; this prevents a stale manuscript from being mistaken for
the current source.
