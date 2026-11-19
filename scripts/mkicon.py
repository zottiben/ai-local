"""Draw ai-local's app icon as a PNG, with no image library on this machine.

A rounded square in the console's accent green, holding a terminal prompt chevron and
a caret: the same two marks the window's own wordmark implies. Supersampled 4x and
box-filtered down, which is the only anti-aliasing available here.

The sizes under crates/ailocal-desktop/icons are committed, so this only runs when the
mark itself changes:

    python3 -I scripts/mkicon.py crates/ailocal-desktop/icons/icon.png
    cd crates/ailocal-desktop/icons
    for s in 32 128 512; do sips -z $s $s icon.png --out "${s}x${s}.png"; done
    sips -z 256 256 icon.png --out "128x128@2x.png"
    # then an .iconset of 16/32/128/256/512 plus their @2x twins, and `iconutil -c icns`

Takes about 15 s at 1024x1024: pure Python, one sample at a time, which is the price
of not adding a build-time dependency for a file that changes once a year.
"""

import struct
import sys
import zlib

SIZE = 1024
SS = 4  # supersample factor

ACCENT = (26, 112, 73)  # --accent, the green the console uses for a primary button
INK = (250, 250, 248)  # --bg, so the mark reads as the page does


# macOS has drawn app icons to a fixed grid since Big Sur: the rounded square is
# 81.4% of the canvas, with a corner radius of 22.37% of the square's own width. An
# icon that fills its canvas sits visibly larger than everything beside it in the Dock.
INSET = 0.093
RADIUS = 0.2237 * (1 - 2 * INSET)


def rounded_square(x, y, n, radius):
    """Inside the squircle-ish rounded square inset in an n x n box."""
    inset = n * INSET
    lo, hi = inset, n - inset
    if not (lo <= x <= hi and lo <= y <= hi):
        return False
    # Corner circles; the straight edges are everything that is not a corner.
    cx = min(max(x, lo + radius), hi - radius)
    cy = min(max(y, lo + radius), hi - radius)
    return (x - cx) ** 2 + (y - cy) ** 2 <= radius**2


def on_segment(px, py, ax, ay, bx, by, half):
    """Within `half` of the segment a-b, with rounded ends."""
    dx, dy = bx - ax, by - ay
    length = dx * dx + dy * dy
    t = 0.0 if length == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / length))
    qx, qy = ax + t * dx, ay + t * dy
    return (px - qx) ** 2 + (py - qy) ** 2 <= half * half


def mark(x, y, n):
    """The chevron and the caret, as a prompt: `>` and `_`.

    Measured off the square rather than the canvas, so changing the inset moves the
    square and its contents together instead of leaving the prompt adrift in it.
    """
    side = n * (1 - 2 * INSET)
    half = side * 0.076 / 2
    # Centred on the square: the pair spans from the chevron's left arm to the end of
    # the caret, so the chevron sits left of the square's own centre line.
    cx, cy = n * INSET + side * 0.435, n * INSET + side * 0.455
    arm = side * 0.165
    if on_segment(x, y, cx - arm, cy - arm, cx + arm * 0.55, cy, half):
        return True
    if on_segment(x, y, cx - arm, cy + arm, cx + arm * 0.55, cy, half):
        return True
    # The caret, on the chevron's baseline and clear of its point.
    base = cy + arm
    return on_segment(x, y, cx + arm * 0.95, base, cx + arm * 1.95, base, half)


def main(path):
    n = SIZE * SS
    radius = n * RADIUS
    rows = []
    for py in range(SIZE):
        row = bytearray()
        for px in range(SIZE):
            r = g = b = a = 0
            for sy in range(SS):
                for sx in range(SS):
                    x, y = px * SS + sx + 0.5, py * SS + sy + 0.5
                    if not rounded_square(x, y, n, radius):
                        continue
                    colour = INK if mark(x, y, n) else ACCENT
                    r += colour[0]
                    g += colour[1]
                    b += colour[2]
                    a += 255
            total = SS * SS
            # Premultiplied would darken the edge against a light background, so
            # average the colour over the covered samples only.
            covered = a // 255
            if covered == 0:
                row += bytes(4)
            else:
                row += bytes((r // covered, g // covered, b // covered, a // total))
        rows.append(bytes(row))

    raw = b"".join(b"\x00" + r for r in rows)
    out = [b"\x89PNG\r\n\x1a\n"]
    for kind, data in (
        (b"IHDR", struct.pack(">IIBBBBB", SIZE, SIZE, 8, 6, 0, 0, 0)),
        (b"IDAT", zlib.compress(raw, 9)),
        (b"IEND", b""),
    ):
        out.append(struct.pack(">I", len(data)) + kind + data)
        out.append(struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF))
    with open(path, "wb") as f:
        f.write(b"".join(out))


if __name__ == "__main__":
    main(sys.argv[1])
