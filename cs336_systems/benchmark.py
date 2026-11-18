import argparse
import json
import statistics
from pathlib import Path
from timeit import default_timer

import torch

from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW
from cs336_basics.model import TransformerLM

from .config import load_yaml, load_model_config

def benchmark(config):
    settings = config['benchmark']
    model_config = config['model']
    device = torch.device(settings['device'])
    torch.manual_seed(settings['seed'])
    model = TransformerLM(**model_config, device=device, dtype=getattr(torch, settings['dtype']))
    model.train()
    tokens = torch.randint(model_config['vocab_size'], (settings['batch_size'], model_config['context_length']), device=device)
    targets = torch.randint(model_config['vocab_size'], tokens.shape, device=device)
    optimizer = AdamW(model.parameters(), **config['optimizer']) if settings['mode'] == 'train' else None
    
    def synchronize():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
    
    def step():
        model.zero_grad()
        logits = model(tokens)
        if settings['mode'] != 'forward':
            loss = cross_entropy(logits, targets)
            loss.backward()
            if optimizer is not None:
                optimizer.step()
    
    synchronize()
    
    for _ in range(settings['warmup_steps']):
        step()
        synchronize()
    timing_ms = []
    for _ in range(settings['measurement_steps']):
        start = default_timer()
        step()
        synchronize()
        timing_ms.append((default_timer() - start) * 1000)
    
    return{
        'config': config,
        'torch_version': torch.__version__,
        'device_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu',
        'parameter_count': sum(p.numel() for p in model.parameters()),
        'timing_ms': timing_ms,
        'mean_ms': statistics.mean(timing_ms),
        'std_ms': statistics.pstdev(timing_ms),
    }
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/benchmark.yaml')
    args = parser.parse_args()
    config_path = Path(args.config)
    config = load_yaml(config_path)
    config['model'] = load_model_config(config_path.parent / config['model_sizes'], config['model_size'], config['model'])
    result = benchmark(config)
    records = json.dumps(result, indent=2)
    print(records)
    if config['benchmark']['output'] is not None:
        output = Path(config['benchmark']['output'])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(records + '\n', encoding='utf-8')

if __name__ == '__main__':
    main()