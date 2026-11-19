from server import checkpoints


def test_checkpoints_cover_start_and_end():
    ks = checkpoints(2425)
    assert ks[:12] == list(range(1, 13))
    assert ks[-1] == 2425
    assert ks == sorted(set(ks)) and all(isinstance(k, int) for k in ks)
    assert checkpoints(1) == [1]
