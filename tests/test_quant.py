import pytest

from amagi_inference_calc import quant as Q


def test_exact_block_sizes():
    assert Q.bits_per_weight("Q8_0") == 34 * 8 / 32
    assert Q.bits_per_weight("Q4_0") == 18 * 8 / 32
    assert Q.bits_per_weight("Q6_K") == 210 * 8 / 256
    assert Q.bits_per_weight("MXFP4") == 17 * 8 / 32


@pytest.mark.parametrize("alias,name", [("q4", "Q4_K_M"), ("UD-Q4_K_M", "Q4_K_M"), ("fp16", "F16"),
                                        ("q4_k", "Q4_K_M"), ("MXFP4_MOE", "MXFP4"), ("int8", "Q8_0")])
def test_aliases(alias, name):
    assert Q.normalize(alias) == name


def test_unknown_quant():
    with pytest.raises(KeyError):
        Q.normalize("Q9_Z")


def test_kv_types():
    assert Q.kv_bytes("f16") == 2.0
    assert Q.kv_bytes("q8_0") == pytest.approx(1.0625)
    assert Q.kv_bytes("Q4") == pytest.approx(0.5625)
    with pytest.raises(KeyError):
        Q.kv_bytes("Q3")
