import os, sys
import argparse
import contextlib

import torch
from fvcore.nn import FlopCountAnalysis
from transformers import AutoModelForCausalLM, AutoTokenizer

a = argparse.ArgumentParser()
a.add_argument('model_arch', default='resnet18', nargs='+')
a.add_argument('--csv-path', default=None)
a.add_argument('--batch-size', default=64, type=int)
a.add_argument('--resolution', default=256, type=int)
a.add_argument('--channels-last', default=False, action='store_true')
args = a.parse_args()

batch_size_multiplier = {
    'distilgpt2': 4,
}

op_names = ['addmm', 'conv', 'batch_norm', 'layer_norm', 'linear', 'matmul', 'adaptive_avg_pool2d']
nn_layer_names = [
    ln.lower() for ln in 
    ['Linear', 'Conv1D', 'Conv2d', 'Dropout', 'LayerNorm', 'BatchNorm2d', 'ReLU', 'MaxPool2d', 'AdaptiveAvgPool2d', 'NewGELUActivation', 'Embedding']
]

memory_format = torch.channels_last if args.channels_last else torch.contiguous_format

class LMWrapper(torch.nn.Module):
    def __init__(self, causal_lm):
        super().__init__()
        self.causal_lm = causal_lm

    def forward(self, input_ids):
        return self.causal_lm.generate(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            pad_token_id=self.causal_lm.config.eos_token_id,
            max_length=30,
        )

def layer_count(module, layers={}):
    for c in module.children():
        layer_name = c.__class__.__name__.lower()
        if not list(c.children()):
            layers[layer_name] = layers.get(layer_name, 0) + 1
        else:
            pass
        layer_count(c, layers)
    return layers

out = open(args.csv_path, 'a') if args.csv_path else sys.stdout
require_header = True

for model_arch in args.model_arch:
    if model_arch.startswith('resnet'):
        batch_dims = (args.batch_size, 3, args.resolution, args.resolution)
        inputs = torch.randn(batch_dims).to(memory_format=memory_format)
        model = torch.hub.load(f'pytorch/vision:v0.13.0', model_arch, pretrained=True, verbose=False)
    elif 'gpt' in model_arch or 'bert' in model_arch:
        inputs = AutoTokenizer.from_pretrained(model_arch)('\n', return_tensors='pt')
        inputs = inputs['input_ids'].expand(
            batch_size_multiplier.get(model_arch, 1) * args.batch_size,
            -1
        ).to(memory_format=memory_format)
        model = LMWrapper(AutoModelForCausalLM.from_pretrained(model_arch))

    flops = FlopCountAnalysis(model.cpu(), inputs.cpu())
    with contextlib.redirect_stderr(None):
        assert (set(flops.by_operator().keys()) - set(op_names)) == set()

    # print(model)
    print({f'model-flops-{op}': flops.by_operator()[op] for op in op_names})
    model_layer_counts = layer_count(model)

    datum = {
        'model-arch': model_arch,
        'model-flops-total': flops.total(),
        **{f'model-flops-{op}': flops.by_operator()[op] for op in op_names},
        **{f'model-layers-{ln}': model_layer_counts.get(ln, 0) for ln in nn_layer_names},
    }
    if require_header:
        out.write(','.join(datum) + '\n')
        require_header = False
    out.write(','.join([str(v) for k, v in datum.items()]) + '\n')
    out.flush()

