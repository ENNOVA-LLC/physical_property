import numpy as np
import pytest
from plotly.graph_objects import Figure
from physical_property import PhysicalProperty

# Use a dummy unit for conversion tests. Since no conversion is defined,
# using the same unit will simply return the original values.


def test_value_initialization():
    p = PhysicalProperty(name="test", value=[1, 2, 3], unit="m")
    np.testing.assert_array_equal(p.value, np.array([1, 2, 3], dtype=float))


def test_update_value_within_bounds():
    p = PhysicalProperty(name="test", value=[5], unit="m", bounds=(0, 10))
    p.update_value(7)
    assert p.value == 7


def test_update_value_out_of_bounds():
    p = PhysicalProperty(name="test", value=[5], unit="m", bounds=(0, 10))
    with pytest.raises(ValueError):
        p.update_value(11)  # 11 is above the upper bound


def test_add_to_value():
    p = PhysicalProperty(name="test", value=[1, 2, 3], unit="m")
    p.add_to_value(2)
    np.testing.assert_array_equal(p.value, np.array([3, 4, 5], dtype=float))


def test_append_value():
    p = PhysicalProperty(name="test", value=[1, 2, 3], unit="m")
    p.append_value(4)
    np.testing.assert_array_equal(p.value, np.array([1, 2, 3, 4], dtype=float))


def test_serialization():
    p = PhysicalProperty(
        name="test", value=[1, 2, 3], unit="m", doc="test doc", bounds=(0, 10)
    )
    d = p.to_dict()
    for key in ["type", "name", "value", "unit", "doc", "bounds"]:
        assert key in d
    p_new = PhysicalProperty.from_dict(d)
    np.testing.assert_array_equal(p_new.value, p.value)
    assert p_new.name == p.name
    assert p_new.unit == p.unit
    assert p_new.doc == p.doc
    assert p_new.bounds == p.bounds


def test_conversion_same_unit():
    p = PhysicalProperty(name="test", value=[1, 2, 3], unit="m")
    # Conversion with the same unit should return the original value.
    converted = p.convert("m")
    np.testing.assert_array_equal(converted, p.value)


def test_addition_operator():
    p1 = PhysicalProperty(name="test", value=[1, 2, 3], unit="m")
    p2 = PhysicalProperty(name="test", value=[4, 5, 6], unit="m")
    p3 = p1 + p2
    np.testing.assert_array_equal(p3.value, np.array([5, 7, 9], dtype=float))


def test_subtraction_operator():
    p1 = PhysicalProperty(name="test", value=[5, 7, 9], unit="m")
    p2 = PhysicalProperty(name="test", value=[1, 2, 3], unit="m")
    p3 = p1 - p2
    np.testing.assert_array_equal(p3.value, np.array([4, 5, 6], dtype=float))


def test_interpolate():
    original = [1, 2, 3, 4, 5]
    p = PhysicalProperty(name="test", value=original, unit="m")
    p_interp = p.interpolate(10)
    assert len(p_interp.value) == 10
    # The new values should be linearly spaced between the original range.
    assert np.isclose(p_interp.value[0], 1)
    assert np.isclose(p_interp.value[-1], 5)


def test_plot_default_title():
    # When no x is provided, x-axis defaults to index.
    p = PhysicalProperty(name="test", value=[1, 2, 3], unit="m")
    fig = p.plot()
    assert isinstance(fig, Figure)
    # Default xaxis_title is "x-index" and default yaxis_title is "test (m)" before unit removal.
    # The plot title is created by removing units.
    expected_title = "test vs x-index"
    assert fig.layout.title.text == expected_title


def test_getitem_slicing():
    p = PhysicalProperty(name="test", value=[1, 2, 3, 4, 5], unit="m")
    p_slice = p[1:4]
    np.testing.assert_array_equal(p_slice.value, np.array([2, 3, 4], dtype=float))


def test_format_scalar():
    p = PhysicalProperty(name="test", value=5, unit="m")
    formatted = format(p, ".2f")
    assert formatted == "5.00"
