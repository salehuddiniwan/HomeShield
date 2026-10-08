import numpy as np

from homeshield.detectors.face import best_match, is_good_face
from homeshield.pipeline import associate_faces


def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def test_best_match_picks_closest_above_threshold():
    alice, bob = _unit([1, 0, 0]), _unit([0, 1, 0])
    query = _unit([0.9, 0.1, 0])
    pid, score = best_match(query, [(1, alice), (2, bob)], threshold=0.5)
    assert pid == 1 and score > 0.9


def test_best_match_rejects_below_threshold():
    pid, score = best_match(_unit([0, 0, 1]), [(1, _unit([1, 0, 0]))], threshold=0.4)
    assert pid is None and score < 0.4


def test_best_match_empty_gallery():
    assert best_match(_unit([1, 0]), []) == (None, 0.0)


def _face(**kw):
    f = {"x": 100.0, "y": 50.0, "w": 60.0, "h": 70.0, "det_score": 0.9,
         "embedding": np.ones(512, np.float32)}
    f.update(kw)
    return f


def test_face_quality_gate():
    assert is_good_face(_face())
    assert not is_good_face(_face(w=20.0))                 # too small
    assert not is_good_face(_face(det_score=0.3))          # unsure detection
    assert not is_good_face(_face(embedding=None))


def test_associate_faces_uses_head_region_of_tightest_box():
    persons = [
        {"id": 1, "bbox": [80, 40, 200, 400]},     # face inside the head region
        {"id": 2, "bbox": [400, 40, 500, 400]},    # nobody's face
        {"id": 3, "bbox": [0, 0, 640, 480]},       # huge box also containing face
    ]
    known = _face(match_id=7)
    unknown = _face(x=420.0, y=60.0, match_id=None)
    low = _face(x=110.0, y=330.0, match_id=8)      # in person 1's lower half
    out = associate_faces(persons, [known, unknown, low])
    assert out == {1: known}
