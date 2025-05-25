import os
import types
import torch
from types import SimpleNamespace

import pytest

import diloco_fsdp_framework as df
from dataset_utils import DatasetLoader
from outer_optimizer import OuterOptimizer


class DummyDataset:
    def __init__(self, samples):
        self.samples = list(samples)
        self.shuffled_with = None

    def take(self, n):
        return (self.samples[i] for i in range(min(n, len(self.samples))))

    def __iter__(self):
        return iter(self.samples)

    def shuffle(self, buffer_size, seed):
        self.shuffled_with = (buffer_size, seed)
        return self


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

    import dataset_utils
    monkeypatch.setattr(dataset_utils, 'load_dataset', fake_load_dataset)
    loader = DatasetLoader('name', 'subset', shuffle_buffer=10, seed=42)
    dataset, field = loader.load()
    assert dataset is ds
    assert field == 'text'
    assert ds.shuffled_with == (loader.shuffle_buffer, loader.seed)


def test_get_dataset_detects_content(monkeypatch):
    ds = DummyDataset([{'content': 'a'}])

    def fake_load_dataset(*args, **kwargs):
        return ds

    import dataset_utils
    monkeypatch.setattr(dataset_utils, 'load_dataset', fake_load_dataset)
    loader = DatasetLoader('name', 'subset')
    dataset, field = loader.load()
    assert field == 'content'
    assert ds.shuffled_with == (loader.shuffle_buffer, loader.seed)


def test_get_dataset_custom_field(monkeypatch):
    ds = DummyDataset([{'foo': 'a'}])

    def fake_load_dataset(*args, **kwargs):
        return ds

    import dataset_utils
    monkeypatch.setattr(dataset_utils, 'load_dataset', fake_load_dataset)
    loader = DatasetLoader('name', 'subset', text_field='foo')
    dataset, field = loader.load()
    assert field == 'foo'
    assert ds.shuffled_with == (loader.shuffle_buffer, loader.seed)


def test_get_dataset_invalid_field(monkeypatch):
    ds = DummyDataset([{'foo': 'a'}])

    def fake_load_dataset(*args, **kwargs):
        return ds

    import dataset_utils
    monkeypatch.setattr(dataset_utils, 'load_dataset', fake_load_dataset)
    loader = DatasetLoader('name', 'subset', text_field='bar')
    with pytest.raises(ValueError):
        loader.load()
    assert ds.shuffled_with == (loader.shuffle_buffer, loader.seed)


def test_trainer_config_validation():
    with pytest.raises(ValueError):
        df.TrainerConfig('m', 'n', 's', 'o', diloco_loops=0)
    with pytest.raises(ValueError):
        df.TrainerConfig('m', 'n', 's', 'o', outer_momentum=-0.1)
    with pytest.raises(ValueError):
        df.TrainerConfig('m', 'n', 's', 'o', outer_momentum=1.1)
    with pytest.raises(ValueError):
        df.TrainerConfig('m', 'n', 's', 'o', lr=0)
    with pytest.raises(ValueError):
        df.TrainerConfig('m', 'n', 's', 'o', outer_lr=0)


def test_tokenize_batch():
    tokenizer = DummyTokenizer()
    examples = [{'txt': 'hello'}, {'txt': 'world'}]
    tokens = DatasetLoader.tokenize_batch(
        tokenizer, examples, text_field='txt', seq_len=3, dynamic_batch=False
    )
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


class Bf16Accelerator:
    device = torch.device('cpu')
    is_main_process = True
    mixed_precision = 'bf16'
    num_processes = 1
    sync_gradients = True

    def accumulate(self, model):
        class Ctx:
            def __enter__(self_):
                pass

            def __exit__(self_, exc_type, exc, tb):
                pass

        return Ctx()

    def backward(self, loss):
        loss.backward()


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.config = SimpleNamespace(use_cache=True)

    def gradient_checkpointing_enable(self):
        pass

    def forward(self, input_ids, labels=None):
        loss = (input_ids.float().mean() * self.weight)
        return SimpleNamespace(loss=loss)


def test_train_bf16_dtype():
    cfg = df.TrainerConfig('m', 'n', 's', 'o', max_steps=1)
    trainer = make_trainer(cfg)
    trainer.accelerator = Bf16Accelerator()
    trainer.device = trainer.accelerator.device
    trainer.is_main = True
    trainer.device_mesh = None
    trainer.model = DummyModel().to(torch.bfloat16)
    trainer.optimizer = torch.optim.AdamW(trainer.model.parameters(), lr=0.1)
    trainer.lr_scheduler = SimpleNamespace(step=lambda: None)
    trainer.outer_opt = OuterOptimizer(
        trainer.model,
        torch.optim.SGD(trainer.model.parameters(), lr=0.1),
        device=trainer.device,
        device_mesh=None,
    )
    trainer.outer_optimizer = trainer.outer_opt.optimizer
    batch = {
        'input_ids': torch.ones(1, 2, dtype=torch.long),
        'labels': torch.ones(1, 2, dtype=torch.long),
    }
    trainer.dataloader = [batch]
    trainer._save_final = lambda: None

    df.DilocoFSDPTrainer.train(trainer)

def test_init_compressed_prev_vector(monkeypatch):
    ds = DummyDataset([{'text': 'a'}])

    import dataset_utils
    monkeypatch.setattr(dataset_utils, 'load_dataset', lambda *a, **k: ds)
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
    assert trainer.outer_opt.prev_vector.dtype == torch.float16


def test_checkpoint_save_load(tmp_path):
    trainer = make_trainer(df.TrainerConfig('m', 'n', 's', 'o'))
    trainer.accelerator = DummyAccelerator()
    trainer.accelerator.get_state_dict = lambda m: m.state_dict()
    trainer.accelerator.unwrap_model = lambda m: m
    trainer.is_main = True
    trainer.model = torch.nn.Linear(1, 1)
    trainer.optimizer = torch.optim.AdamW(trainer.model.parameters())
    trainer.outer_opt = OuterOptimizer(
        trainer.model,
        torch.optim.SGD(trainer.model.parameters(), lr=0.1),
        device=torch.device('cpu'),
        device_mesh=None,
        momentum=0.0,
    )
    trainer.outer_optimizer = trainer.outer_opt.optimizer
    trainer.outer_opt.momentum_buffer = torch.tensor([1.0])
    orig_params = [p.detach().clone() for p in trainer.model.parameters()]
    orig_inner = trainer.optimizer.state_dict()
    orig_outer = trainer.outer_optimizer.state_dict()
    orig_mb = trainer.outer_opt.momentum_buffer.clone()

    ckpt = tmp_path / 'ckpt.pt'
    df.DilocoFSDPTrainer.save_checkpoint(trainer, ckpt, step=5)
    assert ckpt.exists()

    for p in trainer.model.parameters():
        p.data.add_(1)
    trainer.optimizer.param_groups[0]['lr'] = 2.0
    trainer.outer_optimizer.param_groups[0]['lr'] = 3.0
    trainer.outer_opt.momentum_buffer.add_(2)

    step = df.DilocoFSDPTrainer.load_checkpoint(trainer, ckpt)
    assert step == 5
    for p, o in zip(trainer.model.parameters(), orig_params):
        assert torch.equal(p, o)
    assert trainer.optimizer.state_dict()['param_groups'][0]['lr'] == orig_inner['param_groups'][0]['lr']
    assert trainer.outer_optimizer.state_dict()['param_groups'][0]['lr'] == orig_outer['param_groups'][0]['lr']
    assert torch.equal(trainer.outer_opt.momentum_buffer, orig_mb)


def test_outer_optimizer_step():
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.zero_()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    outer = OuterOptimizer(model, opt, device=torch.device('cpu'), device_mesh=None)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)
    delta_norm = outer.step()
    assert pytest.approx(delta_norm, rel=1e-5) == 1.0
    for p in model.parameters():
        assert torch.allclose(p, torch.tensor([[0.9]]), atol=1e-4)
