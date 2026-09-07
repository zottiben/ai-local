#!/usr/bin/env python3
"""Parse GGUF metadata from a (possibly truncated) file head.

Lets us read a model's architecture - context length, layer count, KV head layout -
from the first few MB of a blob, so we can decide whether a 16-28 GB download is
worth starting. gguf.GGUFReader mmaps and walks tensor info, so it cannot do this.

usage: gguf_head.py <file> [key-substring ...]
"""
import struct
import sys

U8, I8, U16, I16, U32, I32, F32, BOOL, STR, ARR, U64, I64, F64 = range(13)
_FIX = {U8: ("<B", 1), I8: ("<b", 1), U16: ("<H", 2), I16: ("<h", 2),
        U32: ("<I", 4), I32: ("<i", 4), F32: ("<f", 4), BOOL: ("<?", 1),
        U64: ("<Q", 8), I64: ("<q", 8), F64: ("<d", 8)}


class Trunc(Exception):
    pass


class R:
    def __init__(self, b):
        self.b, self.o = b, 0

    def take(self, n):
        if self.o + n > len(self.b):
            raise Trunc
        v = self.b[self.o:self.o + n]
        self.o += n
        return v

    def scalar(self, t):
        f, n = _FIX[t]
        return struct.unpack(f, self.take(n))[0]

    def string(self):
        return self.take(self.scalar(U64)).decode("utf-8", "replace")

    def value(self, t):
        if t == STR:
            return self.string()
        if t == ARR:
            et = self.scalar(U32)
            n = self.scalar(U64)
            if n > 4096:                      # don't materialise huge arrays
                for _ in range(n):
                    self.value(et)
                return f"[{n} items]"
            return [self.value(et) for _ in range(n)]
        return self.scalar(t)


def main():
    path, filters = sys.argv[1], [s.lower() for s in sys.argv[2:]]
    with open(path, "rb") as fh:
        r = R(fh.read(64 << 20))

    if r.take(4) != b"GGUF":
        sys.exit("not a GGUF file")
    ver = r.scalar(U32)
    n_tensors = r.scalar(U64)
    n_kv = r.scalar(U64)
    print(f"  gguf v{ver}, {n_tensors} tensors, {n_kv} metadata keys")

    for _ in range(n_kv):
        try:
            k = r.string()
            v = r.value(r.scalar(U32))
        except Trunc:
            print("  -- truncated before all metadata was read --")
            break
        if filters and not any(f in k.lower() for f in filters):
            continue
        if isinstance(v, str) and len(v) > 70:
            v = v[:70] + "..."
        print(f"  {k:52} {v}")


if __name__ == "__main__":
    main()
