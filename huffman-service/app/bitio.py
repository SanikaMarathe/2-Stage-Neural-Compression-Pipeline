"""
bitio.py — Bit-level I/O primitives for FGK Adaptive Huffman codec.

Layout of the byte stream produced by BitWriter.to_bytes():
  [pad_bits: 1 byte][data bytes ...]

  pad_bits (0-7): number of zero bits appended to the last byte to fill it.
  Bits are written/read MSB-first within each byte.
"""


class BitWriter:
    """Accumulates individual bits and serialises them to bytes (MSB-first).

    Usage::
        bw = BitWriter()
        bw.write_bit(1)
        bw.write_bits(0b101, 3)
        data = bw.to_bytes()   # returns bytes
    """

    def __init__(self) -> None:
        self._buf: bytearray = bytearray()
        self._current: int = 0   # bits accumulated in the current byte (MSB-first)
        self._bit_count: int = 0  # how many bits are in _current (0-7)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def write_bit(self, bit: int) -> None:
        """Write a single bit (0 or 1)."""
        self._current = (self._current << 1) | (bit & 1)
        self._bit_count += 1
        if self._bit_count == 8:
            self._buf.append(self._current)
            self._current = 0
            self._bit_count = 0

    def write_bits(self, value: int, n: int) -> None:
        """Write the lowest *n* bits of *value*, MSB first."""
        for shift in range(n - 1, -1, -1):
            self.write_bit((value >> shift) & 1)

    def to_bytes(self) -> bytes:
        """Flush any partial byte and return header + data as bytes.

        The header byte encodes the number of padding zero-bits (0-7) that
        were appended to complete the last byte.
        """
        buf = bytearray(self._buf)  # copy so repeated calls are safe

        pad_bits = 0
        if self._bit_count > 0:
            pad_bits = 8 - self._bit_count
            # Shift the accumulated bits to the MSB position and append.
            buf.append(self._current << pad_bits)

        return bytes([pad_bits]) + bytes(buf)


class BitReader:
    """Reads individual bits from a bytes object produced by BitWriter.to_bytes().

    The first byte of *data* must be the pad-bit-count header.
    Bits are returned MSB-first; reading stops before any padding bits.
    """

    def __init__(self, data: bytes) -> None:
        if len(data) < 1:
            raise ValueError("data must contain at least the 1-byte header")
        pad_bits: int = data[0]
        self._data: bytes = data
        self._total_bits: int = (len(data) - 1) * 8 - pad_bits
        self._pos: int = 0  # index into the *bit* stream (0 = first real bit)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def has_more(self) -> bool:
        """Return True if there are unread non-padding bits remaining."""
        return self._pos < self._total_bits

    def read_bit(self) -> int:
        """Read and return the next bit (0 or 1).

        Raises EOFError when all real bits have been consumed.
        """
        if not self.has_more():
            raise EOFError("No more bits to read")
        # Byte index into data (offset +1 to skip header)
        byte_idx = self._pos // 8 + 1
        bit_idx = 7 - (self._pos % 8)  # MSB-first: bit 7 of the byte is first
        self._pos += 1
        return (self._data[byte_idx] >> bit_idx) & 1

    def read_bits(self, n: int) -> int:
        """Read *n* bits and return them as an integer (MSB first)."""
        value = 0
        for _ in range(n):
            value = (value << 1) | self.read_bit()
        return value


# ---------------------------------------------------------------------------
# Inline sanity checks
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # --- Round-trip of a known bit sequence ---
    bw = BitWriter()
    bits = [1, 0, 1, 1, 0, 0, 1, 0, 1]  # 9 bits → 1 full byte + 1 partial
    for b in bits:
        bw.write_bit(b)
    data = bw.to_bytes()
    # Header should indicate 7 padding bits (9 bits → 2 bytes, padded to 16 → 7 pad)
    assert data[0] == 7, f"Expected pad=7, got {data[0]}"
    br = BitReader(data)
    recovered = []
    while br.has_more():
        recovered.append(br.read_bit())
    assert recovered == bits, f"Round-trip mismatch: {recovered} != {bits}"
    print("PASS: round-trip 9-bit sequence")

    # --- Edge case: empty write ---
    bw2 = BitWriter()
    data2 = bw2.to_bytes()
    assert data2 == bytes([0]), f"Empty write should be b'\\x00', got {data2!r}"
    br2 = BitReader(data2)
    assert not br2.has_more(), "Empty BitReader should have no bits"
    print("PASS: empty write")

    # --- Edge case: exactly 8 bits (no padding) ---
    bw3 = BitWriter()
    bw3.write_bits(0b10110100, 8)
    data3 = bw3.to_bytes()
    assert data3[0] == 0, f"Expected pad=0, got {data3[0]}"
    assert len(data3) == 2, f"Expected 2 bytes (header + 1), got {len(data3)}"
    br3 = BitReader(data3)
    val3 = br3.read_bits(8)
    assert val3 == 0b10110100, f"Expected 0b10110100, got {val3:08b}"
    assert not br3.has_more()
    print("PASS: exactly 8 bits, no padding")

    # --- Edge case: single bit (7 bits of padding) ---
    bw4 = BitWriter()
    bw4.write_bit(1)
    data4 = bw4.to_bytes()
    assert data4[0] == 7, f"Expected pad=7, got {data4[0]}"
    br4 = BitReader(data4)
    assert br4.read_bit() == 1
    assert not br4.has_more()
    print("PASS: single bit (7 padding bits)")

    # --- write_bits round-trip ---
    bw5 = BitWriter()
    bw5.write_bits(0b110, 3)
    bw5.write_bits(0b10101, 5)
    data5 = bw5.to_bytes()
    br5 = BitReader(data5)
    assert br5.read_bits(3) == 0b110
    assert br5.read_bits(5) == 0b10101
    assert not br5.has_more()
    print("PASS: write_bits / read_bits round-trip")

    print("\nAll sanity checks passed.")
