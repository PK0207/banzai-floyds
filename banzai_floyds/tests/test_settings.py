from banzai_floyds.settings import FRAME_SELECTION_CRITERIA


def test_floyds_in_selection_criteria():
    assert any('FLOYDS' in item for item in FRAME_SELECTION_CRITERIA)
