import os, sys
import argparse
import time
import contextlib
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning)
import re
from functools import reduce
from operator import add, mul

import torch
from fvcore.nn import FlopCountAnalysis
from transformers import AutoModelForCausalLM, AutoTokenizer

a = argparse.ArgumentParser()
a.add_argument('--model-arch', default='resnet18')
a.add_argument('--exec-device', default='cuda')
a.add_argument('--csv-path', default=None)
a.add_argument('--batch-size', default=64, type=int)
a.add_argument('--resolution', default=256, type=int)
a.add_argument('--iterations', default=25, type=int)
a.add_argument('--dim', nargs='*', default=[])
a.add_argument('--inference-mode', default=False, action='store_true')
a.add_argument('--no-grad', default=False, action='store_true')
a.add_argument('--cudnn-benchmark', default=False, action='store_true')
a.add_argument('--autocast-dtype', default='float32')
a.add_argument('--channels-last', default=False, action='store_true')
a.add_argument('--pessimization', default=False, action='store_true')
a.add_argument('--jit-script', default=False, action='store_true')
a.add_argument('--eval', default=False, action='store_true')
a.add_argument('--compile', default=False, action='store_true')
a.add_argument('--compile-mode', default=None)
a.add_argument('--optimize', default=False, action='store_true')
a.add_argument('--set-to-none', default=False, action='store_true')
a.add_argument('--bnb', default=False, action='store_true')
args = a.parse_args()

batch_size_multiplier = {
    'distilgpt2': 4,
}

if args.compile and not hasattr(torch, 'compile'):
    raise RuntimeError('torch.compile not present')

op_names = ['addmm', 'conv', 'batch_norm', 'layer_norm', 'linear', 'matmul', 'adaptive_avg_pool2d']

require_header = not args.csv_path or not os.path.exists(args.csv_path)

exec_device = torch.device(args.exec_device)
init_device = torch.device(args.exec_device if not args.pessimization else 'cpu')
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

if args.model_arch.startswith('resnet'):
    batch_dims = (args.batch_size, 3, args.resolution, args.resolution)
    inputs = torch.randn(batch_dims, device=init_device).to(memory_format=memory_format, device=exec_device)
    model = torch.hub.load(f'pytorch/vision:v0.13.0', args.model_arch, pretrained=True, verbose=False)
elif 'gpt' in args.model_arch or 'bert' in args.model_arch:
    if args.channels_last:
        raise RuntimeError('channels_last incompatible with GPT models')
    inputs = AutoTokenizer.from_pretrained(args.model_arch)('\n', return_tensors='pt')
    inputs = inputs['input_ids'].expand(
        batch_size_multiplier.get(args.model_arch, 1) * args.batch_size,
        -1
    ).to(memory_format=memory_format, device=exec_device)
    model = LMWrapper(AutoModelForCausalLM.from_pretrained(args.model_arch))

flops = FlopCountAnalysis(model.cpu(), inputs.cpu())
with contextlib.redirect_stderr(None):
    assert (set(flops.by_operator().keys()) - set(op_names)) == set()

if args.eval:
    model = model.eval()
else:
    model = model.train()
num_model_params = reduce(add, [reduce(mul, p.size()) for n, p in model.named_parameters()])

model = model.to(exec_device, memory_format=memory_format)
if args.jit_script:
    model = torch.jit.script(model, optimize=True)
if args.compile:
    if not hasattr(torch, 'compile'):
        raise RuntimeError('torch.compile not present')
    if args.channels_last:
        raise RuntimeError('channels_last incompatible with torch.compile')
    import torch._dynamo
    torch._dynamo.reset()
    model = torch.compile(model, mode=args.compile_mode)

grad_context_mgr = torch.no_grad if args.no_grad else contextlib.suppress

if args.optimize:
    if args.bnb:
        with contextlib.redirect_stdout(None):
            import bitsandbytes as bnb
        optimizer = bnb.optim.Adam8bit(model.parameters())
    else:
        optimizer = torch.optim.Adam(model.parameters())

with torch.autocast(exec_device.type, enabled=(args.autocast_dtype != 'float32'), dtype=getattr(torch, args.autocast_dtype)):
    if args.cudnn_benchmark:
        torch.backends.cudnn.benchmark = True

    first_batch_s = None

    # JIT and warmup
    with torch.inference_mode(args.inference_mode):
        with grad_context_mgr():
            for i in range(10):
                before = time.time()
                result = model(inputs)

                if args.optimize:
                    loss = result.sum()
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=args.set_to_none)

                if first_batch_s == None:
                    torch.cuda.synchronize()
                    first_batch_s = time.time() - before

    torch.cuda.reset_peak_memory_stats()
    before = time.time()

    with torch.inference_mode(args.inference_mode):
        with grad_context_mgr():
            for i in range(args.iterations):
                result = model(inputs)
                if args.pessimization:
                    for batch_result in result:
                        with contextlib.redirect_stdout(None): # sync to host but don't actually embarrass ourselves on stdout
                            print(batch_result.sum().item())

                if args.optimize:
                    loss = result.sum()
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=args.set_to_none)

    torch.cuda.synchronize()
    elapsed_s = time.time() - before

datum = {
    'torch-version': re.search('\d+\.\d+\.\d+', torch.__version__).group(),
    'cuda-version': torch.version.cuda,
    'cudnn-version': f'{torch.backends.cudnn.version() / 1000:.3f}',
    'model-arch': args.model_arch,
    'optimize': args.optimize,
    'batch-size': args.batch_size,
    'resolution-h': args.resolution,
    'resolution-w': args.resolution,
    'pixels': args.resolution * args.resolution,
    'iterations': args.iterations,
    'inference-mode': args.inference_mode,
    'no-grad': args.no_grad,
    'cudnn-benchmark': args.cudnn_benchmark,
    'autocast-dtype': args.autocast_dtype,
    'channels-last': args.channels_last,
    'pessimization': args.pessimization,
    'jit-script': args.jit_script,
    'eval': args.eval,
    'compile': args.compile,
    'compile-mode': args.compile_mode,
    'set-to-none': args.set_to_none,
    'bnb': args.bnb,
    'model-params': num_model_params,
    'first-batch (ms)': round(1000 * first_batch_s),
    'throughput (it/s)': round(args.iterations * args.batch_size / elapsed_s),
    'peak-memory (GB)': round(torch.cuda.max_memory_allocated(device=exec_device) / 1024 / 1024 / 1024, 3),
    'model-flops-total': flops.total(),
    **{f'model-flops-{op}': flops.by_operator()[op] for op in op_names},
    **{k: v for k, v in [d.split('=') for d in args.dim]}
}

out = open(args.csv_path, 'a') if args.csv_path else sys.stdout
if require_header:
    out.write(','.join(datum) + '\n')
out.write(','.join([str(v) for k, v in datum.items()]) + '\n')
out.flush()
