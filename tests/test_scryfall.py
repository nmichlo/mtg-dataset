from pathlib import Path
from uuid import UUID

import pytest

from mtgdata import ScryfallBulkType
from mtgdata import ScryfallImageType
from mtgdata.scryfall import ScryfallCardFace

pytestmark = pytest.mark.basic_test


def test_image_type_sizes():
    # `art_crop` is the only type with no fixed size -- scryfall crops vary per card
    assert [t.size for t in ScryfallImageType] == [
        (146, 204),
        (480, 680),
        (488, 680),
        (672, 936),
        (745, 1040),
        None,
    ]
    assert [t.extension for t in ScryfallImageType] == ["jpg", "jpg", "jpg", "jpg", "png", "jpg"]

    with pytest.raises(ValueError, match="has no fixed size"):
        ScryfallImageType.art_crop.width


@pytest.mark.parametrize(
    ("width", "height", "expected"),
    [
        (None, None, (146, 204)),  # native size
        (224, None, (224, 313)),  # width drives, aspect kept
        (None, 224, (160, 224)),  # height drives, aspect kept
        (224, 160, (224, 313)),  # scales to cover the larger of the two
        (1, 1, (1, 1)),  # never rounds a dimension down to zero
    ],
)
def test_get_scaled_size(width, height, expected):
    assert ScryfallImageType.small.get_scaled_size(width=width, height=height) == expected


def test_card_face_img_path_is_windows_safe():
    # `set_code` reaches the filesystem, so ':' (illegal on windows) and reserved
    # device names like 'con' must not survive into the path
    def _face(set_code: str) -> ScryfallCardFace:
        return ScryfallCardFace(
            id="00000000-0000-0000-0000-000000000001",
            oracle_id="00000000-0000-0000-0000-000000000002",
            name="Card",
            set_code=set_code,
            set_name="Set",
            img_uri="https://example.invalid/card.jpg",
            _img_type=ScryfallImageType.small,
            _bulk_type=ScryfallBulkType.default_cards,
            _sets_dir=Path("data") / "sets",
        )

    assert _face("con").img_path == Path("data") / "sets" / "con_" / "00000000-0000-0000-0000-000000000001.jpg"
    assert _face("a:b*c?").img_path == Path("data") / "sets" / "abc" / "00000000-0000-0000-0000-000000000001.jpg"
    assert (
        _face("  spaced.  ").img_path == Path("data") / "sets" / "spaced" / "00000000-0000-0000-0000-000000000001.jpg"
    )
    assert _face("::").img_path == Path("data") / "sets" / "_" / "00000000-0000-0000-0000-000000000001.jpg"
    assert _face("MH2").uuid == UUID("00000000-0000-0000-0000-000000000001")
