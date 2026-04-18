from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from bitio import BitReader, BitWriter

_NUM_CEIL = 513  # 256 syms x2 + 1


@dataclass(eq=False)
class Node:
    weight: int
    number: int                 # higher -> closer to root
    symbol: Optional[int]       # None for internal nodes
    parent: Optional["Node"]
    left: Optional["Node"]      # left -> 0 bit
    right: Optional["Node"]     # right -> 1 bit
    is_nyt: bool = False

    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


class FGKTree:

    def __init__(self) -> None:
        # start with just one node - the NYT (not-yet-transmitted) leaf as root
        self.nyt: Node = Node(
            weight=0,
            number=_NUM_CEIL,
            symbol=None,
            parent=None,
            left=None,
            right=None,
            is_nyt=True,
        )
        self.root: Node = self.nyt
        self.leaves: dict[int,Node] = {}
        self._all_nodes: list[Node] = [self.nyt]
        self._next_child: int = _NUM_CEIL - 1  # counts down as we add nodes

    def _path_to_root(self, node: Node) -> list[int]:
        # walk up collecting bits, then reverse so it reads root -> leaf
        bits: list[int] = []
        cur = node
        while cur.parent is not None:
            par = cur.parent
            bits.append(0 if par.left is cur else 1)
            cur = par
        bits.reverse()
        return bits

    def _best_swap(self, node: Node) -> Optional[Node]:
        # highest numbered node with same weight, skip node itself and its parent
        w = node.weight
        par = node.parent
        b: Optional[Node] = None
        for n in self._all_nodes:
            if n is node or n is par:
                continue
            if n.weight == w:
                if b is None or n.number > b.number:
                    b = n
        return b

    def _swap_nodes(self, x: Node, y: Node) -> None:
        # swap tree positions and swap their numbers (numbers travel with positions)
        if x is y:
            return
        px,py = x.parent, y.parent

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
        x.number,y.number = y.number,x.number

    def split_nyt(self, symbol: int) -> Node:
        # new symbol seen - expand NYT into internal node with two children
        old_nyt = self.nyt
        nyt_cn  = self._next_child - 1  # lower -> new nyt
        sym_ln  = self._next_child      # higher -> symbol leaf
        self._next_child -= 2

        new_nyt = Node(
            weight=0,
            number=nyt_cn,
            symbol=None,
            parent=old_nyt,
            left=None,
            right=None,
            is_nyt=True,
        )
        sym_leaf = Node(
            weight=1,
            number=sym_ln,
            symbol=symbol,
            parent=old_nyt,
            left=None,
            right=None,
            is_nyt=False,
        )

        # repurpose old NYT in-place as the internal node
        old_nyt.is_nyt = False
        old_nyt.symbol = None
        old_nyt.left   = new_nyt
        old_nyt.right  = sym_leaf

        self.nyt = new_nyt
        self.leaves[symbol] = sym_leaf
        self._all_nodes.append(new_nyt)
        self._all_nodes.append(sym_leaf)
        return sym_leaf

    def update_tree(self, node: Node) -> None:
        # walk to root: swap if out of order, then bump weight
        cur = node
        while cur is not None:
            leader = self._best_swap(cur)
            if leader is not None and leader.number > cur.number:
                self._swap_nodes(cur, leader)
            cur.weight += 1
            cur = cur.parent

    def encode_symbol(self, symbol: int, writer: BitWriter) -> None:
        if symbol in self.leaves:
            # known symbol - emit its path from root
            leaf = self.leaves[symbol]
            for bit in self._path_to_root(leaf):
                writer.write_bit(bit)
            self.update_tree(leaf)
        else:
            # new symbol - emit NYT path then raw 8-bit symbol
            for bit in self._path_to_root(self.nyt):
                writer.write_bit(bit)
            writer.write_bits(symbol, 8)
            leaf = self.split_nyt(symbol)
            self.update_tree(leaf.parent)

    def decode_symbol(self, reader: BitReader) -> int:
        if self.root.is_nyt:  # first symbol ever, just read 8 bits raw
            symbol = reader.read_bits(8)
            leaf = self.split_nyt(symbol)
            self.update_tree(leaf.parent)
            return symbol

        # follow bits down tree until we hit a leaf
        cur = self.root
        while not cur.is_leaf():
            bit = reader.read_bit()
            cur = cur.left if bit == 0 else cur.right

        if cur.is_nyt:
            # landed on NYT, next 8 bits are the raw symbol
            symbol = reader.read_bits(8)
            leaf = self.split_nyt(symbol)
            self.update_tree(leaf.parent)
        else:
            symbol = cur.symbol
            self.update_tree(cur)
        return symbol


def encode(text: str) -> bytes:
    if any(ord(ch) > 255 for ch in text):
        raise ValueError("FGK codec is 8-bit only; input must contain only code points 0-255")
    writer = BitWriter()
    tree = FGKTree()
    for ch in text:
        tree.encode_symbol(ord(ch), writer)
    return writer.to_bytes()


def decode(data: bytes) -> str:
    reader = BitReader(data)
    tree = FGKTree()
    res: list[str] = []
    try:
        while reader.has_more():
            sym = tree.decode_symbol(reader)
            res.append(chr(sym))
    except EOFError as e:
        raise ValueError(f"Corrupt or truncated Huffman stream: {e}") from e
    return "".join(res)


if __name__ == "__main__":
    def _check(label: str, orig: str) -> None:
        c = encode(orig)
        r = decode(c)
        assert r == orig, f"FAIL {label}: {r!r} != {orig!r}"
        info = f"{len(orig)} chars -> {len(c)} bytes"
        if len(orig) > 0:
            info += f", ratio {len(c)/len(orig):.2f}"
        print(f"ok {label}: {info}")

    _check("empty",      "")
    _check("single",     "x")
    _check("two same",   "aa")
    _check("hello",      "hello world")
    _check("repeated",   "aaaaaabbbbbbcccc")
    _check("pangram",    "the quick brown fox jumps over the lazy dog")
    _check("mono 100x",  "z" * 100)
    _check("all ascii",  "".join(chr(i) for i in range(32,128)))

    print("all ok")
