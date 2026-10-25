from pathlib import Path
p=Path('src/molcore/contrast.py')
s=p.read_text().replace("            lo, hi = value - delta, value + delta\n        lower = lo if np.isfinite(lo) and lo < value else float(np.nextafter(value, -np.inf))\n        upper = hi if np.isfinite(hi) and hi > value else float(np.nextafter(value, np.inf))", "            lo, hi = value - delta, value + delta\n            lower = lo if np.isfinite(lo) and lo < value else float(np.nextafter(value, -np.inf))\n            upper = hi if np.isfinite(hi) and hi > value else float(np.nextafter(value, np.inf))")
s=s.replace("        if mask is not None and np.asarray(mask).shape == data.shape[1:]:\n            mask = np.asarray(mask)\n", '')
s=s.replace("    lower, upper = _limits([samples], settings)\n    return ContrastResult", "    lower, upper = _limits([samples], settings)\n    _report(completed, len(indices), None, cancel_event)\n    return ContrastResult", 1)
s=s.replace("    if state_token(experiment) != expected:\n", "    _report(completed, total, None, cancel_event)\n    if state_token(experiment) != expected:\n", 1)
p.write_text(s)
