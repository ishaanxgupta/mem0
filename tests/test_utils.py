import pytest
from mem0.memory.utils import remove_code_blocks, extract_json

def test_remove_code_blocks():
    # Test with language tag
    content1 = "```python\nprint('hello')\n```"
    assert remove_code_blocks(content1) == "print('hello')"

    # Test without language tag
    content2 = "```\njust some text\n```"
    assert remove_code_blocks(content2) == "just some text"

    # Test without code block markers
    content3 = "plain text"
    assert remove_code_blocks(content3) == "plain text"

    # Test with <think> tags
    content4 = "<think>some thoughts</think>actual content"
    assert remove_code_blocks(content4) == "actual content"

    # Test mixed content: code block AND <think> tags
    content5 = "```\n<think>internal thought</think>code content\n```"
    assert remove_code_blocks(content5) == "code content"

    # Test with whitespace around markers
    content6 = "  ```python\nprint('hello')\n```  "
    assert remove_code_blocks(content6) == "print('hello')"

    # Test where code block is not the whole string (should return as is but stripped and <think> removed)
    content7 = "Some text before\n```\ncode\n```"
    assert remove_code_blocks(content7) == "Some text before\n```\ncode\n```"

def test_extract_json():
    # JSON in ```json block
    json_str1 = '```json\n{"key": "value"}\n```'
    assert extract_json(json_str1) == '{"key": "value"}'

    # JSON in ``` block
    json_str2 = '```\n{"key": "value"}\n```'
    assert extract_json(json_str2) == '{"key": "value"}'

    # Raw JSON
    json_str3 = '{"key": "value"}'
    assert extract_json(json_str3) == '{"key": "value"}'

    # JSON with surrounding text (re.search)
    json_str4 = 'Here is the data: ```json\n{"a": 1}\n``` and more text.'
    assert extract_json(json_str4) == '{"a": 1}'

    # JSON with different whitespace
    json_str5 = '```json {"b": 2} ```'
    assert extract_json(json_str5) == '{"b": 2}'
