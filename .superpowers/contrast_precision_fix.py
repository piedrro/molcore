from pathlib import Path

p = Path("src/molcore/contrast.py")
s = p.read_text()
start = s.index("        # NumPy's linear interpolation")
end = s.index("    lower, upper = float(lower), float(upper)", start)
s = s[:start] + """        values.sort()
        limits = []
        for percentile in settings.percentiles:
            rank = (len(values) - 1) * (percentile / 100.)
            lo, hi = int(np.floor(rank)), int(np.ceil(rank))
            a, b = float(values[lo]), float(values[hi])
            fraction = rank - lo
            if lo == hi:
                value = a
            elif np.signbit(a) != np.signbit(b):
                # Opposite-sign endpoints can overflow b-a. A convex sum cannot,
                # and unlike global scaling it preserves tiny finite populations.
                value = (1. - fraction) * a + fraction * b
            elif fraction < .5:
                value = a + (b - a) * fraction
            else:
                value = b - (b - a) * (1. - fraction)
            limits.append(value)
        lower, upper = limits
""" + s[end:]
p.write_text(s)
