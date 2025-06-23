import spacy

import utils
from processing.gptism_cleanup import replace_gptisms


def test_replace_gptisms_no_model(monkeypatch):
    monkeypatch.setattr(utils.spacy_manager, "_nlp", None)
    monkeypatch.setattr(utils, "load_spacy_model_if_needed", lambda: None)
    text = "As an AI language model, I cannot help with that."
    cleaned, count = replace_gptisms(text, threshold=80)
    assert cleaned == text
    assert count == 0


def test_replace_gptisms_with_model(monkeypatch):
    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")
    monkeypatch.setattr(utils.spacy_manager, "_nlp", nlp)
    monkeypatch.setattr(utils, "load_spacy_model_if_needed", lambda: None)
    text = "As an AI language model, I cannot help with that."
    cleaned, count = replace_gptisms(text, threshold=80)
    assert count == 1
    assert "as an ai language model" not in cleaned.lower()
