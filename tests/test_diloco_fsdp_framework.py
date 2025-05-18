import torch
import pytest

import diloco_fsdp_framework as dff


class DummyDataset:
    def __init__(self, samples):
        self.samples = samples

    def take(self, n):
        for i in range(min(n, len(self.samples))):
            yield self.samples[i]

    def __iter__(self):
        return iter(self.samples)


class DummyTokenizer:
    def __init__(self):
        self.called_with = None

    def __call__(self, texts, max_length, truncation, padding, return_tensors):
        self.called_with = {
            "texts": texts,
            "max_length": max_length,
            "truncation": truncation,
            "padding": padding,
            "return_tensors": return_tensors,
        }
        return {"input_ids": torch.tensor([[1, 2], [3, 4]])}


def make_trainer(cfg):
    trainer = dff.DilocoFSDPTrainer.__new__(dff.DilocoFSDPTrainer)
    trainer.config = cfg
    trainer.tokenizer = DummyTokenizer()
    return trainer


def test_get_dataset_detect_text(monkeypatch):
    dataset = DummyDataset([{"text": "hello"}])

    def fake_load_dataset(name, name=None, split=None, streaming=None):
        return dataset

    monkeypatch.setattr(dff, "load_dataset", fake_load_dataset)
    cfg = dff.TrainerConfig(
        model_name="m",
        dataset_name="ds",
        subset="sub",
        output_dir="out",
    )
    trainer = make_trainer(cfg)

    ds, field = dff.DilocoFSDPTrainer._get_dataset(trainer)
    assert ds is dataset
    assert field == "text"


def test_get_dataset_invalid_field(monkeypatch):
    dataset = DummyDataset([{"other": "hi"}])

    monkeypatch.setattr(dff, "load_dataset", lambda *a, **k: dataset)
    cfg = dff.TrainerConfig(
        model_name="m",
        dataset_name="ds",
        subset="sub",
        output_dir="out",
    )
    trainer = make_trainer(cfg)

    with pytest.raises(ValueError):
        dff.DilocoFSDPTrainer._get_dataset(trainer)


def test_get_dataset_explicit_field(monkeypatch):
    dataset = DummyDataset([{"content": "hi"}])
    monkeypatch.setattr(dff, "load_dataset", lambda *a, **k: dataset)
    cfg = dff.TrainerConfig(
        model_name="m",
        dataset_name="ds",
        subset="sub",
        output_dir="out",
        text_field="content",
    )
    trainer = make_trainer(cfg)

    ds, field = dff.DilocoFSDPTrainer._get_dataset(trainer)
    assert field == "content"
    assert ds is dataset


def test_tokenize_batch(monkeypatch):
    cfg = dff.TrainerConfig(
        model_name="m",
        dataset_name="ds",
        subset="sub",
        output_dir="out",
    )
    trainer = make_trainer(cfg)

    examples = [{"text": "a"}, {"text": "b"}]
    result = dff.DilocoFSDPTrainer._tokenize_batch(trainer, examples, "text", seq_len=5)

    assert isinstance(result, dict)
    assert torch.equal(result["labels"], result["input_ids"])
    called = trainer.tokenizer.called_with
    assert called["texts"] == ["a", "b"]
    assert called["max_length"] == 5
    assert called["truncation"] is True
    assert called["padding"] == "max_length"
    assert called["return_tensors"] == "pt"
