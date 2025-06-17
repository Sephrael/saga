import pytest

from utils import text_processing


class DummySpan:
    def __init__(self, text: str, start: int, end: int) -> None:
        self.text = text
        self.start_char = start
        self.end_char = end


class DummyNLP:
    def __call__(self, text: str):
        class Doc:
            def __init__(self, t: str):
                self.sents = [DummySpan(t, 0, len(t))]

        return Doc(text)


@pytest.mark.asyncio
async def test_find_quote_offsets_no_model(monkeypatch):
    monkeypatch.setattr(text_processing.spacy_manager, "_nlp", None)
    monkeypatch.setattr(text_processing.spacy_manager, "load", lambda: None)
    result = await text_processing.find_quote_and_sentence_offsets_with_spacy(
        "doc", "quote"
    )
    assert result is None


@pytest.mark.asyncio
async def test_find_quote_offsets_direct(monkeypatch):
    monkeypatch.setattr(text_processing.spacy_manager, "_nlp", DummyNLP())
    monkeypatch.setattr(text_processing.spacy_manager, "load", lambda: None)
    result = await text_processing.find_quote_and_sentence_offsets_with_spacy(
        "Hello world", "world"
    )
    assert result == (6, 11, 0, len("Hello world"))
