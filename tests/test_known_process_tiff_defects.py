from __future__ import annotations

import numpy as np

from suite2p_in_depth.process_tiff import reslice_zstack


def test_reslice_zstack_preserves_rectangular_shape_and_piezo_input() -> None:
    stack = np.arange(3 * 2 * 4, dtype=np.float32).reshape(3, 2, 4)
    piezo = np.linspace(1, 2, 5)[:, np.newaxis]
    original_piezo = piezo.copy()

    result = reslice_zstack(stack, piezo, spacing=1)

    assert result.shape == stack.shape
    np.testing.assert_array_equal(piezo, original_piezo)


def test_reslice_zstack_supports_non_unit_spacing() -> None:
    stack = np.broadcast_to(np.arange(4, dtype=np.float32)[:, None, None], (4, 3, 5)).copy()
    piezo = np.array([0.0, 2.0, 4.0])

    result = reslice_zstack(stack, piezo, spacing=2)

    assert result.shape == (4, 3, 5)
    np.testing.assert_array_equal(result[:, 1], stack[:, 1])
    np.testing.assert_array_equal(result[1, 0], np.zeros(5, dtype=np.float32))
    np.testing.assert_array_equal(result[1, -1], np.full(5, 2, dtype=np.float32))


def test_reslice_zstack_without_piezo_is_identity_copy() -> None:
    stack = np.arange(24, dtype=np.int16).reshape(3, 2, 4)

    result = reslice_zstack(stack, None, spacing=2)

    np.testing.assert_array_equal(result, stack)
    assert result is not stack
