"""Tests for the period-difficulty experiment.

The finding - that the hold-out months are an unusually favourable draw - depends on the
evaluation blocks tiling the post-training span cleanly and on the training window being
held fixed. A sliding window would confound period difficulty with training-set size.
"""
from src.evaluation.period_difficulty import (
    FIRST_EVAL, HOLDOUT, LAST_MONTH, TRAIN_MONTHS, blocks,
)


def test_blocks_do_not_overlap():
    for (_, hi), (lo, _) in zip(blocks(), blocks()[1:]):
        assert lo > hi


def test_blocks_are_contiguous():
    for (_, hi), (lo, _) in zip(blocks(), blocks()[1:]):
        assert lo == hi + 1, "a gap would silently drop months from the comparison"


def test_blocks_cover_the_whole_evaluation_span():
    b = blocks()
    assert b[0][0] == FIRST_EVAL
    assert b[-1][1] == LAST_MONTH


def test_final_block_is_truncated_not_extended():
    """The span need not divide evenly; the last block must not run past the data."""
    assert blocks(36, 70, 3)[-1] == (69, 70)
    assert all(hi <= 70 for _, hi in blocks(36, 70, 3))


def test_evaluation_never_touches_the_training_window():
    assert FIRST_EVAL > TRAIN_MONTHS[1]
    assert all(lo > TRAIN_MONTHS[1] for lo, _ in blocks())


def test_holdout_lies_inside_the_evaluated_span():
    """Otherwise the hold-out could not be placed in the difficulty distribution."""
    assert FIRST_EVAL <= HOLDOUT[0] and HOLDOUT[1] <= LAST_MONTH


def test_block_size_is_respected_except_at_the_end():
    b = blocks(36, 70, 3)
    assert all(hi - lo + 1 == 3 for lo, hi in b[:-1])
