from unittest.mock import Mock, patch

import pytest

from mem0.configs.llms.base import BaseLlmConfig
from mem0.llms import litellm


@pytest.fixture
def mock_litellm():
    with patch("mem0.llms.litellm.litellm") as mock_litellm:
        yield mock_litellm


def test_generate_response_without_tools(mock_litellm):
    config = BaseLlmConfig(model="gpt-4.1-nano-2025-04-14", temperature=0.7, max_tokens=100, top_p=1)
    llm = litellm.LiteLLM(config)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"},
    ]

    mock_response = Mock()
    mock_response.choices = [Mock(message=Mock(content="I'm doing well, thank you for asking!"))]
    mock_litellm.completion.return_value = mock_response

    response = llm.generate_response(messages)

    mock_litellm.completion.assert_called_once_with(
        model="gpt-4.1-nano-2025-04-14", messages=messages, temperature=0.7, max_tokens=100, top_p=1.0
    )
    assert response == "I'm doing well, thank you for asking!"


def test_generate_response_with_tools(mock_litellm):
    config = BaseLlmConfig(model="gpt-4.1-nano-2025-04-14", temperature=0.7, max_tokens=100, top_p=1)
    llm = litellm.LiteLLM(config)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Add a new memory: Today is a sunny day."},
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "add_memory",
                "description": "Add a memory",
                "parameters": {
                    "type": "object",
                    "properties": {"data": {"type": "string", "description": "Data to add to memory"}},
                    "required": ["data"],
                },
            },
        }
    ]

    mock_response = Mock()
    mock_message = Mock()
    mock_message.content = "I've added the memory for you."
    mock_response.choices = [Mock(message=mock_message)]
    mock_litellm.completion.return_value = mock_response

    response = llm.generate_response(messages, tools=tools)

    # tools should be ignored and not passed to litellm.completion
    mock_litellm.completion.assert_called_once_with(
        model="gpt-4.1-nano-2025-04-14", messages=messages, temperature=0.7, max_tokens=100, top_p=1.0
    )

    assert response == "I've added the memory for you."
