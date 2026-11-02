from pathlib import Path

p = Path("src/molcore/types/tables.py")
s = p.read_text().replace(
    "        from .._locking import model_locked\nfrom ._shared import CoordinateSpace\n\n",
    "",
)
p.write_text(s)
