from homeshield.zones import point_in_polygon, scale_polygon

SQUARE = [[0, 0], [10, 0], [10, 10], [0, 10]]
# U shape: the notch (4..6, 5..10) is outside.
U_SHAPE = [[0, 0], [10, 0], [10, 10], [6, 10], [6, 5], [4, 5], [4, 10], [0, 10]]


def test_point_in_square():
    assert point_in_polygon(5, 5, SQUARE)
    assert not point_in_polygon(15, 5, SQUARE)
    assert not point_in_polygon(-1, -1, SQUARE)


def test_concave_polygon_notch_is_outside():
    assert point_in_polygon(2, 8, U_SHAPE)
    assert point_in_polygon(8, 8, U_SHAPE)
    assert not point_in_polygon(5, 8, U_SHAPE)


def test_degenerate_polygon():
    assert not point_in_polygon(0, 0, [[0, 0], [1, 1]])


def test_scale_polygon_from_reference_frame():
    assert scale_polygon([[320, 240]], 640, 480, 1280, 720) == [[640.0, 360.0]]
