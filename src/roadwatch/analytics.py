"""Pure helpers for virtual-line crossing analytics."""
from collections import Counter
from typing import Iterable

def signed_line_side(point, line_start, line_end):
    """Signed 2D cross product: which side of a directed line is the point on?"""
    px, py = point; x1, y1 = line_start; x2, y2 = line_end
    return (x2-x1)*(py-y1) - (y2-y1)*(px-x1)

def crossed_line(previous, current, line_start, line_end, deadband=2.0):
    """Return crossing direction if a point moved from one side to the other."""
    before = signed_line_side(previous, line_start, line_end)
    after = signed_line_side(current, line_start, line_end)
    if abs(before) <= deadband or abs(after) <= deadband:
        return None
    if before < 0 < after: return "negative_to_positive"
    if before > 0 > after: return "positive_to_negative"
    return None

def summarize_crossings(events: Iterable[dict]):
    counts = Counter((e["class_name"], e["direction"]) for e in events)
    return [{"class_name": cls, "direction": direction, "count": count}
            for (cls, direction), count in sorted(counts.items())]
