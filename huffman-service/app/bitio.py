class BitWriter:

    def __init__(self) -> None:
        self._buf: bytearray = bytearray()   # completed bytes
        self._current: int = 0               # partial byte being built
        self._bcnt: int = 0                  # bits in current partial byte

    def write_bit(self, bit: int) -> None:
        # shift bit into current byte, flush when full
        self._current = (self._current << 1) | (bit & 1)
        self._bcnt += 1
        if self._bcnt == 8:
            self._buf.append(self._current)
            self._current = 0
            self._bcnt = 0

    def write_bits(self, value: int, n: int) -> None:
        # write n bits msb first
        for shift in range(n-1, -1, -1):
            self.write_bit((value >> shift) & 1)

    def to_bytes(self) -> bytes:
        # first byte = num padding bits, then data bytes
        buf = bytearray(self._buf)
        pad = 0
        if self._bcnt > 0:
            pad = 8 - self._bcnt
            buf.append(self._current << pad)  # pad with zeros on the right
        return bytes([pad]) + bytes(buf)


class BitReader:

    def __init__(self, data: bytes) -> None:
        if len(data) < 1:
            raise ValueError("data must contain at least the 1-byte header")
        pad: int = data[0]  # how many trailing bits to ignore
        self._data: bytes = data
        self._total_bits: int = (len(data)-1) * 8 - pad  # real bit count
        self._pos: int = 0

    def has_more(self) -> bool:
        return self._pos < self._total_bits

    def read_bit(self) -> int:
        if not self.has_more():
            raise EOFError("No more bits to read")
        b_idx = self._pos // 8 + 1  # skip header byte
        bi = 7 - (self._pos % 8)    # msb first within each byte
        self._pos += 1
        return (self._data[b_idx] >> bi) & 1

    def read_bits(self, n: int) -> int:
        # read n bits, reassemble into int msb first
        val = 0
        for _ in range(n):
            val = (val << 1) | self.read_bit()
        return val


if __name__ == "__main__":
    bw = BitWriter()
    bits = [1,0,1,1,0,0,1,0,1]
    for b in bits:
        bw.write_bit(b)
    data = bw.to_bytes()
    assert data[0] == 7, f"expected pad=7, got {data[0]}"
    br = BitReader(data)
    got = []
    while br.has_more():
        got.append(br.read_bit())
    assert got == bits
    print("ok: 9-bit roundtrip")

    bw2 = BitWriter()
    d2 = bw2.to_bytes()
    assert d2 == bytes([0])
    br2 = BitReader(d2)
    assert not br2.has_more()
    print("ok: empty write")

    bw3 = BitWriter()
    bw3.write_bits(0b10110100,8)
    d3 = bw3.to_bytes()
    assert d3[0] == 0
    assert len(d3) == 2
    br3 = BitReader(d3)
    v3 = br3.read_bits(8)
    assert v3 == 0b10110100
    assert not br3.has_more()
    print("ok: 8 bits no pad")

    bw4 = BitWriter()
    bw4.write_bit(1)
    d4 = bw4.to_bytes()
    assert d4[0] == 7
    br4 = BitReader(d4)
    assert br4.read_bit() == 1
    assert not br4.has_more()
    print("ok: single bit")

    bw5 = BitWriter()
    bw5.write_bits(0b110,3)
    bw5.write_bits(0b10101,5)
    d5 = bw5.to_bytes()
    br5 = BitReader(d5)
    assert br5.read_bits(3) == 0b110
    assert br5.read_bits(5) == 0b10101
    assert not br5.has_more()
    print("ok: write_bits roundtrip")
