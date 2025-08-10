import pytest

from text_utils import extract_text_from_upload

def test_extract_text_from_txt():
    data = b"hello world"
    assert extract_text_from_upload("file.txt", data) == "hello world"

def test_extract_text_from_unknown_extension():
    with pytest.raises(ValueError):
        extract_text_from_upload("file.xyz", b"data")
