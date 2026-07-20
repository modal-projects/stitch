from stitch.routing.tokens import extract_token_ids


class FakeTokenizer:
    def encode(self, text):
        return [ord(c) % 1000 for c in text]

    def apply_chat_template(self, messages, *, add_generation_prompt, tokenize):
        assert add_generation_prompt and not tokenize
        return "|".join(m["content"] for m in messages) + "<gen>"


def test_generate_input_ids_passthrough():
    ids, total = extract_token_ids("generate", {"input_ids": [1, 2, 3]}, None)
    assert ids == [1, 2, 3] and total == 3


def test_generate_batched_input_ids():
    ids, total = extract_token_ids("/generate", {"input_ids": [[1, 2], [3, 4, 5]]}, None)
    assert ids == [1, 2]  # first sequence keys the trie
    assert total == 5  # batch sum drives the load counter


def test_generate_text_with_tokenizer():
    ids, total = extract_token_ids("generate", {"text": "hi"}, FakeTokenizer())
    assert ids == [ord("h"), ord("i")] and total == 2


def test_generate_text_byte_fallback():
    ids, total = extract_token_ids("generate", {"text": "hi"}, None)
    assert ids == list(b"hi") and total == 2
    assert all(0 <= t < 256 for t in ids)


def test_generate_batched_text():
    ids, total = extract_token_ids("generate", {"text": ["ab", "cde"]}, None)
    assert ids == list(b"ab") and total == 5


def test_chat_completions_uses_chat_template():
    payload = {"messages": [{"role": "user", "content": "hello"}]}
    ids, total = extract_token_ids("v1/chat/completions", payload, FakeTokenizer())
    assert total == len("hello<gen>")


def test_chat_completions_byte_fallback_with_blocks():
    payload = {
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "hi"}, {"type": "image_url"}]}
        ]
    }
    ids, total = extract_token_ids("v1/chat/completions", payload, None)
    assert ids == list(b"hi") and total == 2


def test_completions_prompt():
    ids, total = extract_token_ids("v1/completions", {"prompt": "abc"}, None)
    assert ids == list(b"abc") and total == 3


def test_non_routed_and_empty():
    assert extract_token_ids("server_info", {"x": 1}, None) == ([], 0)
    assert extract_token_ids("generate", None, None) == ([], 0)
    assert extract_token_ids("generate", {}, None) == ([], 0)


def test_tokenizer_failure_degrades_to_empty():
    class Broken:
        def encode(self, text):
            raise RuntimeError("boom")

    assert extract_token_ids("generate", {"text": "hi"}, Broken()) == ([], 0)
