from data import fire_history_archive as a


def _sq(x0, y0, x1, y1, name=None):
    return {"geometry": {"type": "Polygon", "coordinates": [[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]]},
            "properties": {"FIRE_NAME": name}}


def _same(f, g, na="", nb=""):
    return a._same_fire(f, a._bbox(f), g, a._bbox(g), na, nb)


def test_duplicate_listing_with_different_names_is_one_fire():
    assert _same(_sq(0, 0, 1, 1), _sq(0.05, 0, 1, 1), "UNKNOWN", "")


def test_separate_fires_far_apart_are_kept():
    assert not _same(_sq(0, 0, 1, 1), _sq(2, 2, 3, 3), "A", "B")


def test_small_fire_beside_a_big_one_is_not_merged():
    # overlaps the big box but its centre is outside the big polygon and the boxes barely overlap
    assert not _same(_sq(0, 0, 1, 1), _sq(0.95, 0.95, 1.2, 1.2), "BIG", "SMALL")


def test_same_name_overlapping_is_one_fire():
    assert _same(_sq(0, 0, 1, 1), _sq(0.9, 0.9, 2, 2), "CREEK", "CREEK")


def test_norm_strips_fire_words_and_acres():
    assert a._norm("Azusa Fire 26 Acres") == a._norm("AZUSA")
