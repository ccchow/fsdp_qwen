import os
import types
import torch
from types import SimpleNamespace

import pytest

import diloco_fsdp_framework as df


class DummyDataset:
    def __init__(self, samples):
        self.samples = list(samples)

    def take(self, n):
        return (self.samples[i] for i in range(min(n, len(self.samples))))

    def __iter__(self):
        return iter(self.samples)


def make_trainer(config):
    trainer = df.DilocoFSDPTrainer.__new__(df.DilocoFSDPTrainer)
    trainer.config = config
    return trainer


class DummyTokenizer:
    def __init__(self):
        self.called_with = None
        self.pad_token = None
        self.eos_token = 0

    def __call__(self, texts, max_length, truncation, padding, return_tensors):
        self.called_with = {
            'texts': texts,
            'max_length': max_length,
            'truncation': truncation,
            'padding': padding,
            'return_tensors': return_tensors,
        }
        ids = torch.ones((len(texts), max_length), dtype=torch.long)
        return {'input_ids': ids}


class DummyAccelerator:
    is_main_process = True

    def wait_for_everyone(self):
        pass

    def unwrap_model(self, model):
        return model

    def get_state_dict(self, model):
        return {'dummy': torch.tensor(1)}


def test_get_dataset_detects_text(monkeypatch):
    ds = DummyDataset([{'text': 'a'}])

    def fake_load_dataset(*args, **kwargs):
        return ds

    monkeypatch.setattr(df, 'load_dataset', fake_load_dataset)
    cfg = df.TrainerConfig('model', 'name', 'subset', 'out')
    trainer = make_trainer(cfg)
    dataset, field = df.DilocoFSDPTrainer._get_dataset(trainer)
    assert dataset is ds
    assert field == 'text'


def test_get_dataset_detects_content(monkeypatch):
    ds = DummyDataset([{'content': 'a'}])

    def fake_load_dataset(*args, **kwargs):
        return ds

    monkeypatch.setattr(df, 'load_dataset', fake_load_dataset)
    cfg = df.TrainerConfig('model', 'name', 'subset', 'out')
    trainer = make_trainer(cfg)
    dataset, field = df.DilocoFSDPTrainer._get_dataset(trainer)
    assert field == 'content'


def test_get_dataset_custom_field(monkeypatch):
    ds = DummyDataset([{'foo': 'a'}])

    def fake_load_dataset(*args, **kwargs):
        return ds

    monkeypatch.setattr(df, 'load_dataset', fake_load_dataset)
    cfg = df.TrainerConfig('model', 'name', 'subset', 'out', text_field='foo')
    trainer = make_trainer(cfg)
    dataset, field = df.DilocoFSDPTrainer._get_dataset(trainer)
    assert field == 'foo'


def test_get_dataset_invalid_field(monkeypatch):
    ds = DummyDataset([{'foo': 'a'}])

    def fake_load_dataset(*args, **kwargs):
        return ds

    monkeypatch.setattr(df, 'load_dataset', fake_load_dataset)
    cfg = df.TrainerConfig('m', 'n', 's', 'o', text_field='bar')
    trainer = make_trainer(cfg)
    with pytest.raises(ValueError):
        df.DilocoFSDPTrainer._get_dataset(trainer)


def test_tokenize_batch():
    tokenizer = DummyTokenizer()
    trainer = make_trainer(df.TrainerConfig('m', 'n', 's', 'o'))
    trainer.tokenizer = tokenizer
    examples = [{'txt': 'hello'}, {'txt': 'world'}]
    tokens = df.DilocoFSDPTrainer._tokenize_batch(trainer, examples, text_field='txt', seq_len=3)
    assert tokens['input_ids'].shape == (2, 3)
    assert torch.equal(tokens['labels'], tokens['input_ids'])
    assert tokenizer.called_with['max_length'] == 3


def test_save_final(tmp_path):
    trainer = make_trainer(df.TrainerConfig('m', 'n', 's', str(tmp_path)))
    trainer.accelerator = DummyAccelerator()
    trainer.model = types.SimpleNamespace()
    trainer.tokenizer = types.SimpleNamespace()
    trainer.is_main = True
    trainer.accelerator.unwrap_model = lambda m: trainer.model
    trainer.accelerator.get_state_dict = lambda m: {'k': 1}
    trainer.model.save_pretrained = lambda path, state_dict: open(os.path.join(path, 'model.saved'), 'w').write('ok')
    trainer.tokenizer.save_pretrained = lambda path: open(os.path.join(path, 'tokenizer.saved'), 'w').write('ok')

    df.DilocoFSDPTrainer._save_final(trainer)

    assert (tmp_path / 'model.saved').exists()
    assert (tmp_path / 'tokenizer.saved').exists()


def test_init_compressed_prev_vector(monkeypatch):
    ds = DummyDataset([{'text': 'a'}])

    monkeypatch.setattr(df, 'load_dataset', lambda *a, **k: ds)
    monkeypatch.setattr(df.AutoTokenizer, 'from_pretrained', lambda *a, **k: DummyTokenizer())

    def fake_model(*args, **kwargs):
        model = torch.nn.Linear(2, 2)
        model.config = types.SimpleNamespace(use_cache=False)
        model.gradient_checkpointing_enable = lambda: None
        return model

    monkeypatch.setattr(df.AutoModelForCausalLM, 'from_pretrained', fake_model)

    dummy_acc = SimpleNamespace(
        device=torch.device('cpu'),
        is_main_process=True,
        num_processes=1,
        mixed_precision='no',
        state=SimpleNamespace(fsdp_plugin=None),
        prepare=lambda *args: args,
    )

    cfg = df.TrainerConfig('m', 'd', 's', 'out', max_steps=1)
    trainer = df.DilocoFSDPTrainer(cfg, accelerator=dummy_acc)
    assert trainer.prev_vector.dtype == torch.float16
