# FGK Adaptive Huffman Algorithm

## Why Adaptive?

Static Huffman coding requires two passes over the input:
1. Count symbol frequencies.
2. Build a frequency-optimal tree.
3. Encode the input using that tree.

The receiver also needs the tree, which must be transmitted as overhead before the compressed payload begins.

**Adaptive Huffman** (Faller 1973, Gallager 1978, Knuth 1985 — hence FGK) eliminates both constraints. The tree is built incrementally as each symbol is encoded. Because encoder and decoder process the same symbols in the same order and apply identical update rules, their trees remain bit-for-bit identical throughout. No tree transmission is needed; the compressed bitstream is self-contained.

## Core Data Structures

- **Nodes** carry a weight (symbol frequency seen so far) and an implicit **node number**. Nodes are numbered in breadth-first, left-to-right order from the root downward: the root has the highest number, leaves near the bottom have the lowest.
- **NYT node** ("Not Yet Transmitted") — a special leaf with weight 0. It represents all symbols not yet seen.

## Encoding a New Symbol

**First occurrence:**
1. Emit the current path from root to the NYT node (its Huffman code at this moment).
2. Emit the raw 8-bit ASCII value of the symbol.
3. Split the NYT leaf into two children: a new NYT leaf (left) and a new symbol leaf (right), both with weight 0. Both sides of the encoder/decoder perform this split identically.

**Subsequent occurrences:**
1. Emit the current path to the symbol's leaf.
2. Apply the update rule (see below).

## Sibling Property

A binary tree is a valid Huffman tree if and only if its nodes, when listed in non-decreasing order of weight, have siblings adjacent in that list. FGK maintains this property dynamically via the **swap rule**.

## Update Rule (Swap Walk)

After transmitting a symbol, walk from its leaf up to the root:

```
current_node = symbol's leaf
while current_node != root:
    candidate = highest-numbered node with same weight as current_node
                that is not current_node's parent
    if candidate != current_node:
        swap(current_node, candidate)   # exchange tree positions + node numbers
    increment current_node.weight
    current_node = current_node.parent
increment root.weight
```

The swap exchanges two nodes' positions in the tree and their implicit numbers — effectively re-rooting subtrees to maintain the sibling property after each weight increment.

## Why "Highest-Numbered"?

Node numbers run highest at the root and lowest at the deepest leaves. Swapping with the **highest-numbered** node of equal weight ensures the least-frequent nodes remain deepest in the tree (longest codes), while high-frequency nodes rise toward the root (shorter codes). This is the precise condition that maintains the sibling property and keeps the tree optimal after each update.

## Decoder Symmetry

The decoder receives bits and traverses the current tree from root to leaf. When it reaches a leaf, it:
- If the leaf is NYT: reads the next 8 bits as a raw symbol, applies the same split as the encoder.
- Otherwise: identifies the symbol at that leaf, applies the same update rule.

Because both sides apply identical operations to an identical sequence of symbols, the trees remain synchronized without any explicit tree data in the bitstream.

## Compression Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| Compression ratio | compressed\_bytes / original\_bytes | <1 means compression; >1 means expansion |
| Shannon entropy | −Σ p(c) log₂ p(c) | Theoretical minimum bits/symbol for this symbol distribution |
| Encoding efficiency | entropy / avg\_bits\_per\_symbol | 1.0 = optimal; lower means overhead from tree adaptation or short inputs |

**Key insight on short inputs:** FGK pays a startup cost while the tree is sparse. For very short strings (< ~8 symbols), the raw-symbol escapes dominate and the ratio exceeds 1.0 (expansion). Efficiency converges toward 1.0 as the symbol stream grows and the tree stabilizes.
