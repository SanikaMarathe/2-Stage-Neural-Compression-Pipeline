"""
test_fgk_golden.py -- Fixed-output "golden" regression tests for the FGK codec.

These tests hardcode the exact byte output for known inputs so that regressions
are caught even when roundtrip tests still pass (e.g. if the bit layout changes
in a compatible but different way).

Golden value captured with:
    python3 -c "
    import sys; sys.path.insert(0, 'huffman-service/app'); sys.path.insert(0, 'huffman-service')
    from app.fgk import encode; print(list(encode('hello')))
    "
Result: [7, 104, 50, 141, 132, 55, 128]
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'huffman-service', 'app'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'huffman-service'))

from app.fgk import encode, decode  # noqa: E402

# ---------------------------------------------------------------------------
# Golden values
# ---------------------------------------------------------------------------

# Hardcoded expected byte output for encode('hello')
_HELLO_GOLDEN = bytes([7, 104, 50, 141, 132, 55, 128])


class TestGoldenHello:
    def test_encode_hello_matches_golden(self):
        """encode('hello') must produce the exact expected byte sequence."""
        assert encode('hello') == _HELLO_GOLDEN

    def test_decode_golden_hello_roundtrip(self):
        """decode of the golden bytes must recover 'hello'."""
        assert decode(_HELLO_GOLDEN) == 'hello'

    def test_golden_is_stable(self):
        """Encoding the same string twice gives the same result (determinism)."""
        assert encode('hello') == encode('hello')


class TestGoldenOther:
    """A few additional deterministic golden cases for common short inputs."""

    @pytest.mark.parametrize("text", [
        "a",
        "aa",
        "ab",
        "abc",
        "hello world",
        "0 1 2 3",
        "4 7 2 9",
    ])
    def test_encode_decode_stable(self, text: str):
        """Re-encoding the same text always yields identical bytes (no randomness)."""
        b1 = encode(text)
        b2 = encode(text)
        assert b1 == b2, f"encode is not deterministic for {text!r}"
        assert decode(b1) == text
