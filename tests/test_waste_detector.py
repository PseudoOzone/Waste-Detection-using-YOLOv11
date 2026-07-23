from PIL import Image
import pytest

from src.waste_detector import disposal_category, validate_image


def test_disposal_category_matches_keyword():
    mapping = {"plastic": "Dry recyclable"}
    assert disposal_category("plastic bottle", mapping) == "Dry recyclable"


def test_disposal_category_has_safe_fallback():
    assert disposal_category("unknown object", {}) == "Check local waste guidance"


def test_validate_image_converts_to_rgb():
    image = Image.new("L", (10, 10))
    assert validate_image(image).mode == "RGB"


def test_validate_image_rejects_large_input():
    image = Image.new("RGB", (100, 100))
    with pytest.raises(ValueError):
        validate_image(image, max_pixels=50)
