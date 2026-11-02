from pathlib import Path

for filename, blockname, contrastline, anchor in (
    (
        "src/molcore/__init__.py",
        "from .types.scientific import (",
        "from .types.contrast import ContrastMethod, ContrastResult, ContrastScope, ContrastSettings",
        "__version__",
    ),
    (
        "src/molcore/types/__init__.py",
        "from .scientific import (",
        "from .contrast import ContrastMethod, ContrastResult, ContrastScope, ContrastSettings",
        "__all__",
    ),
):
    p = Path(filename)
    s = p.read_text().replace(contrastline + "\n", "")
    start = s.index(blockname)
    end = s.index("\n)", start) + 3
    block = s[start:end]
    s = s[:start] + s[end:]
    index = s.index(anchor)
    s = s[:index] + contrastline + "\n" + block + "\n\n" + s[index:]
    p.write_text(s)
