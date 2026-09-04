import pytest
from parsing import parse_model_json, ParseError

def test_plain_json():
    assert parse_model_json('{"a": 1}') == {"a": 1}

def test_fenced_block_stripped():
    assert parse_model_json('```json\n{"a": 1}\n```') == {"a": 1}

def test_broken_json_repaired():
    assert parse_model_json('{"a": 1, "b": [1,2,],}') == {"a": 1, "b": [1, 2]}

def test_irreparable_raises():
    with pytest.raises(ParseError):
        parse_model_json("not json at all <<<")
