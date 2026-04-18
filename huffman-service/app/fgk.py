"""
fgk.py -- FGK (Faller-Gallager-Knuth) Adaptive Huffman codec.

The encoder and decoder each maintain an identical tree that evolves
symbol-by-symbol.  Because both start from the same empty tree and apply the
same update rule after each symbol they stay in lock-step -- no tree is
transmitted alongside the compressed data.

Bit convention: left child = 0, right child = 1.

Number convention
-----------------
Higher numbers are closer to the root (root has the highest number in the
entire tree at all times).  We enforce parent.number > child.number strictly.

When we split NYT (currently at some position with number N):
  * The internal node occupies that same position, so it keeps number N.
  * The two new children must have numbers < N, so we issue them from a
    descending counter that starts below N.

We initialise the root/NYT with a large ceiling number so there is always room
to allocate children below it.  _next_child tracks the next available child
number (counts downward from the initial ceiling).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from bitio import BitReader, BitWriter

# Upper bound for number assignment.  256 symbols x 2 (leaves + internals) + 1.
_NUMBER_CEILING = 513


@dataclass(eq=False)
class Node:
    weight: int
    number: int                   # implicit ordering; higher <-> closer to root
    symbol: Optional[int]         # None for internal nodes
    parent: Optional["Node"]
    left: Optional["Node"]        # None for leaves; left -> 0 bit
    right: Optional["Node"]       # None for leaves; right -> 1 bit
    is_nyt: bool = False

    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


class FGKTree:
    """Adaptive Huffman tree using the FGK algorithm."""

    def __init__(self) -> None:
        # NYT leaf is the root at start.  Give it the ceiling number so it is
        # the highest-numbered node (as required for the root).
        self.nyt: Node = Node(
            weight=0,
            number=_NUMBER_CEILING,
            symbol=None,
            parent=None,
            left=None,
            right=None,
            is_nyt=True,
        )
        self.root: Node = self.nyt
        self.leaves: dict[int, Node] = {}   # symbol -> leaf node
        self._all_nodes: list[Node] = [self.nyt]
        # _next_child is the next number to hand out for a new child.
        # It starts just below the ceiling and decrements by 1 for each new node.
        self._next_child: int = _NUMBER_CEILING - 1

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _path_to_root(self, node: Node) -> list[int]:
        """Return bit-path from root down to *node* (first element = first bit)."""
        bits: list[int] = []
        cur = node
        while cur.parent is not None:
            par = cur.parent
            bits.append(0 if par.left is cur else 1)
            cur = par
        bits.reverse()
        return bits

    def _highest_numbered_same_weight(self, node: Node) -> Optional[Node]:
        """Return the highest-numbered node with the same weight as *node*,
        excluding *node* itself and its direct parent.

        We also skip nodes with a higher number than we can legitimately swap
        with (i.e. ancestors of *node*) because swapping a node with its own
        ancestor would corrupt the tree.  Ancestors always have strictly greater
        weights in a valid FGK tree once the invariant is maintained, so in
        practice this filter is redundant -- but the parent exclusion is
        essential for correctness.
        """
        w = node.weight
        par = node.parent
        best: Optional[Node] = None
        for n in self._all_nodes:
            if n is node or n is par:
                continue
            if n.weight == w:
                if best is None or n.number > best.number:
                    best = n
        return best

    def _swap_nodes(self, x: Node, y: Node) -> None:
        """Swap the tree positions of *x* and *y* (updating parent/child links)
        and swap their numbers (numbers travel with positions)."""
        if x is y:
            return
        px, py = x.parent, y.parent

        def replace_child(par: Optional[Node], old: Node, new: Node) -> None:
            if par is None:
                self.root = new
            elif par.left is old:
                par.left = new
            else:
                par.right = new

        replace_child(px, x, y)
        replace_child(py, y, x)
        x.parent = py
        y.parent = px
        x.number, y.number = y.number, x.number

    # ------------------------------------------------------------------
    # Core FGK operations
    # ------------------------------------------------------------------

    def split_nyt(self, symbol: int) -> Node:
        """Expand the NYT leaf into an internal node with two children.

        Left  child = new NYT leaf    (weight 0, lower number)
        Right child = new symbol leaf (weight 1, higher number -- still < internal)

        The internal node keeps NYT's current number (highest of the three),
        preserving parent.number > child.number for both new children.

        Returns the new symbol leaf.
        """
        old_nyt = self.nyt

        # Allocate two child numbers below the internal node's number.
        # new_nyt gets the lower one; symbol_leaf gets the higher one.
        # Both are still strictly less than old_nyt.number.
        nyt_child_number    = self._next_child - 1   # lower  (new NYT)
        symbol_leaf_number  = self._next_child       # higher (symbol leaf)
        self._next_child -= 2

        new_nyt = Node(
            weight=0,
            number=nyt_child_number,
            symbol=None,
            parent=old_nyt,
            left=None,
            right=None,
            is_nyt=True,
        )
        symbol_leaf = Node(
            weight=1,
            number=symbol_leaf_number,
            symbol=symbol,
            parent=old_nyt,
            left=None,
            right=None,
            is_nyt=False,
        )

        # Repurpose old_nyt in-place as the internal node (keeps its number).
        old_nyt.is_nyt = False
        old_nyt.symbol = None
        old_nyt.left   = new_nyt
        old_nyt.right  = symbol_leaf
        # old_nyt.weight stays 0; update_tree will increment it upward.

        self.nyt = new_nyt
        self.leaves[symbol] = symbol_leaf
        self._all_nodes.append(new_nyt)
        self._all_nodes.append(symbol_leaf)

        return symbol_leaf

    def update_tree(self, node: Node) -> None:
        """Walk from *node* up to root, maintaining the FGK sibling property.

        For each node on the path:
          1. Find the highest-numbered node with the same weight.
          2. If it has a strictly higher number than current (meaning current is
             out of order), swap their positions.
          3. Increment the weight.
          4. Move to parent.
        """
        cur = node
        while cur is not None:
            leader = self._highest_numbered_same_weight(cur)
            if leader is not None and leader.number > cur.number:
                self._swap_nodes(cur, leader)
            cur.weight += 1
            cur = cur.parent

    # ------------------------------------------------------------------
    # Symbol encode / decode
    # ------------------------------------------------------------------

    def encode_symbol(self, symbol: int, writer: BitWriter) -> None:
        if symbol in self.leaves:
            leaf = self.leaves[symbol]
            for bit in self._path_to_root(leaf):
                writer.write_bit(bit)
            self.update_tree(leaf)
        else:
            for bit in self._path_to_root(self.nyt):
                writer.write_bit(bit)
            writer.write_bits(symbol, 8)
            leaf = self.split_nyt(symbol)
            # symbol_leaf already has weight=1; start update from its parent
            # (the internal node, weight=0) so it gets incremented too.
            self.update_tree(leaf.parent)

    def decode_symbol(self, reader: BitReader) -> int:
        # Fast path: tree is still a bare NYT (first ever symbol).
        if self.root.is_nyt:
            symbol = reader.read_bits(8)
            leaf = self.split_nyt(symbol)
            self.update_tree(leaf.parent)
            return symbol

        cur = self.root
        while not cur.is_leaf():
            bit = reader.read_bit()
            cur = cur.left if bit == 0 else cur.right

        if cur.is_nyt:
            symbol = reader.read_bits(8)
            leaf = self.split_nyt(symbol)
            self.update_tree(leaf.parent)
        else:
            symbol = cur.symbol
            self.update_tree(cur)
        return symbol


# ---------------------------------------------------------------------------
# Top-level API
# ---------------------------------------------------------------------------

def encode(text: str) -> bytes:
    """Encode *text* using FGK Adaptive Huffman; return compressed bytes.

    Input must contain only code points in range 0–255 (Latin-1).
    Raises ValueError for Unicode text outside this range.
    """
    if any(ord(ch) > 255 for ch in text):
        raise ValueError("FGK codec is 8-bit only; input must contain only code points 0–255")
    writer = BitWriter()
    tree = FGKTree()
    for ch in text:
        tree.encode_symbol(ord(ch), writer)
    return writer.to_bytes()


def decode(data: bytes) -> str:
    """Decode bytes produced by *encode*; return the original string."""
    reader = BitReader(data)
    tree = FGKTree()
    result: list[str] = []
    try:
        while reader.has_more():
            sym = tree.decode_symbol(reader)
            result.append(chr(sym))
    except EOFError as e:
        raise ValueError(f"Corrupt or truncated Huffman stream: {e}") from e
    return "".join(result)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    def _check(label: str, original: str) -> None:
        c = encode(original)
        r = decode(c)
        assert r == original, f"FAIL {label}: {r!r} != {original!r}"
        info = f"{len(original)} chars -> {len(c)} bytes"
        if len(original) > 0:
            info += f", ratio {len(c)/len(original):.2f}"
        print(f"PASS {label}: {info}")

    _check("empty string",      "")
    _check("single char",       "x")
    _check("two same chars",    "aa")
    _check("hello world",       "hello world")
    _check("repeated chars",    "aaaaaabbbbbbcccc")
    _check("pangram",           "the quick brown fox jumps over the lazy dog")
    _check("mono 100x",         "z" * 100)
    _check("all unique ASCII",  "".join(chr(i) for i in range(32, 128)))

    print("\nAll self-tests passed.")
