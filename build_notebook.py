"""Builds both notebooks from the cell lists.

    python build_notebook.py
    jupyter nbconvert --to notebook --execute --inplace sentiment_analysis.ipynb

Outputs
    sentiment_analysis.ipynb         local version (data in data/)
    sentiment_analysis_colab.ipynb   Google Colab version with setup and download cells
"""

from pathlib import Path

import nbformat as nbf

from colab_cells import COLAB_POST, COLAB_PRE
from markdown_text import MD
from notebook_cells import CELLS

LOCAL_DATA_LINE = 'DATA_PATH = Path("data/Amazon_Unlocked_Mobile.csv")'


def render(cells):
    out = []
    for kind, src in cells:
        src = src.strip("\n")
        if kind == "md":
            for key, text in MD.items():
                src = src.replace(key, text.strip("\n"))
        out.append(nbf.v4.new_markdown_cell(src) if kind == "md" else nbf.v4.new_code_cell(src))
    return out


def notebook(cells, colab=False):
    nb = nbf.v4.new_notebook()
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    if colab:
        nb["metadata"]["colab"] = {"provenance": [], "toc_visible": True}
    nb.cells = render(cells)
    return nb


Path("sentiment_analysis.ipynb").write_text(nbf.writes(notebook(CELLS)))

assert any(LOCAL_DATA_LINE in src for _, src in CELLS)
colab_cells = [(k, s.replace(LOCAL_DATA_LINE, "DATA_PATH = Path(COLAB_DATA_PATH)")) for k, s in CELLS]
colab_cells = colab_cells[:1] + COLAB_PRE + colab_cells[1:] + COLAB_POST
Path("sentiment_analysis_colab.ipynb").write_text(nbf.writes(notebook(colab_cells, colab=True)))
print(f"Wrote sentiment_analysis.ipynb ({len(CELLS)} cells) and sentiment_analysis_colab.ipynb ({len(colab_cells)} cells)")
