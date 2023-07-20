import os, sys
import argparse
import time
import contextlib
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning)
import re
from functools import reduce, partial
from operator import add, mul

import torch
from torch.utils.data import DataLoader
from torch.profiler import profile, record_function, ProfilerActivity
from fvcore.nn import FlopCountAnalysis
from transformers import AutoModelForSequenceClassification, AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from gpu_mem_track import MemTracker
from checkpoint_monkey import ckpt_monkey

a = argparse.ArgumentParser()
a.add_argument('--model-arch', default='resnet18')
a.add_argument('--exec-device', default='cuda')
a.add_argument('--csv-path', default=None)
a.add_argument('--batch-size', default=32, type=int)
a.add_argument('--resolution', default=256, type=int)
a.add_argument('--iterations', default=25, type=int)
a.add_argument('--exit-wait-s', default=0, type=float)
a.add_argument('--dim', nargs='*', default=[])
a.add_argument('--inference-mode', default=False, action='store_true')
a.add_argument('--no-grad', default=False, action='store_true')
a.add_argument('--cudnn-benchmark', default=False, action='store_true')
a.add_argument('--autocast-dtype', default='float32')
a.add_argument('--model-cast-dtype', default=None)
a.add_argument('--channels-last', default=False, action='store_true')
a.add_argument('--pessimization', default=False, action='store_true')
a.add_argument('--jit-script', default=False, action='store_true')
a.add_argument('--eval', default=False, action='store_true')
a.add_argument('--compile', default=False, action='store_true')
a.add_argument('--compile-mode', default=None)
a.add_argument('--optimize', default=False, action='store_true')
a.add_argument('--lr', default=0.001, type=float)
a.add_argument('--optimizer', default='Adam')
a.add_argument('--optimizer-arg', nargs='*', default=[])
a.add_argument('--set-to-none', default=None, type=lambda v: v.lower() == 'true')
a.add_argument('--bnb', default=False, action='store_true')
a.add_argument('--paged', default=False, action='store_true')
a.add_argument('--print', default=False, action='store_true')
a.add_argument('--mem-ckpt', default=None)
args = a.parse_args()

if args.compile and not hasattr(torch, 'compile'):
    raise RuntimeError('torch.compile not present')

op_names = ['addmm', 'conv', 'batch_norm', 'layer_norm', 'linear', 'matmul', 'adaptive_avg_pool2d']

require_header = not args.csv_path or not os.path.exists(args.csv_path)

exec_device = torch.device(args.exec_device)
init_device = torch.device(args.exec_device if not args.pessimization else 'cpu')
memory_format = torch.channels_last if args.channels_last else torch.contiguous_format
model_cast_dtype = getattr(torch, args.model_cast_dtype) if args.model_cast_dtype else None
autocast_dtype = getattr(torch, args.autocast_dtype)

# mem_tracker = MemTracker()

def trace_memory(label, trace):
    prev_mem_gb = trace[-1][1] if trace else 0
    mem_gb = round(torch.cuda.max_memory_allocated(device=exec_device) / 1024 / 1024 / 1024, 3)
    trace.append((label, mem_gb, mem_gb - prev_mem_gb))
    torch.cuda.reset_peak_memory_stats()
    # mem_tracker.track(label)
    return trace

mem_trace_gb = []


class LMWrapper(torch.nn.Module):
    def __init__(self, causal_lm):
        super().__init__()
        self.causal_lm = causal_lm

    def forward(self, input_dict):
        return self.causal_lm(**input_dict)


if args.model_arch.startswith('resnet'):
    batch_dims = (args.batch_size, 3, args.resolution, args.resolution)
    input_dtype = model_cast_dtype if model_cast_dtype else autocast_dtype
    inputs = torch.randn(batch_dims, device=init_device, dtype=input_dtype).to(memory_format=memory_format, device=exec_device)
    mem_trace_gb = trace_memory('inputs-created', mem_trace_gb)
    model = torch.hub.load(f'pytorch/vision:v0.13.0', args.model_arch, pretrained=True, verbose=False)

elif 'gpt' in args.model_arch or 'bert' in args.model_arch:
    if args.channels_last:
        raise RuntimeError('channels_last incompatible with GPT models')
    tokenizer = AutoTokenizer.from_pretrained(args.model_arch, pad_token='[PAD]')
    # tokenizer.pad_token = '\n'#tokenizer.eos_token
    # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    dataset = load_dataset('yelp_review_full').filter(lambda e, i: i < 1000, with_indices=True)

    def tokenize_function(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True)

    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True
    ).remove_columns(['text', 'label'])
    tokenized_datasets.set_format('torch')

    train_dataloader = DataLoader(tokenized_datasets['train'], shuffle=False, batch_size=args.batch_size)
    for batch in train_dataloader:
        inputs = {k: v.to(exec_device) for k, v in batch.items()}
        inputs['labels'] = inputs['input_ids']
        break

    # inputs = AutoTokenizer.from_pretrained(args.model_arch)('\n', return_tensors='pt')
    # input_dtype = torch.long
    # inputs = inputs['input_ids'].expand(
    #     batch_size_multiplier.get(args.model_arch, 1) * args.batch_size,
        # -1
    # ).to(dtype=input_dtype, memory_format=memory_format, device=exec_device)
    mem_trace_gb = trace_memory('inputs-created', mem_trace_gb)
    model = LMWrapper(AutoModelForCausalLM.from_pretrained(args.model_arch))

if args.print:
    print(model)
    import sys; sys.exit()


if args.mem_ckpt:
    # if args.model_arch.startswith('resnet'):
    model = ckpt_monkey(model, re.compile(args.mem_ckpt))
    # else:
    #     raise NotImplemented()

if args.eval:
    model = model.eval()
else:
    model = model.train()
num_model_params = reduce(add, [reduce(mul, p.size()) for n, p in model.named_parameters()])

model = model.to(dtype=model_cast_dtype, memory_format=memory_format, device=exec_device)

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

mem_trace_gb = trace_memory('model-ready', mem_trace_gb)

if args.optimize:
    optim_module = torch.optim
    optim_class_name = args.optimizer
    if args.bnb:
        with contextlib.redirect_stdout(None):
            import bitsandbytes as bnb
        optim_module = bnb.optim
        optim_class_name += '8bit'
        if args.paged:
            optim_class_name = 'Paged' + optim_class_name

    optim_class = getattr(optim_module, optim_class_name)
    optim_args = {k: float(v) for k, v in [a.split('=') for a in args.optimizer_arg]}
    optimizer = optim_class(model.parameters(), lr=args.lr, **optim_args)

loss = None

with torch.autocast(exec_device.type, enabled=(args.autocast_dtype != 'float32'), dtype=autocast_dtype):
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
                    loss = result.loss if hasattr(result, 'loss') else result.sum()
                    loss.backward()
                    optimizer.step()
                    if args.set_to_none == None:
                        optimizer.zero_grad()
                    else:
                        optimizer.zero_grad(set_to_none=args.set_to_none)

                if first_batch_s == None:
                    torch.cuda.synchronize()
                    first_batch_s = time.time() - before

    torch.cuda.reset_peak_memory_stats() # don't want to record peak of prev section, just instantaneous pre-main-loop
    # mem_trace_gb = trace_memory('pre-main-loop', mem_trace_gb)

    loss = None
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
                    loss = result.loss if hasattr(result, 'loss') else result.sum()
                    loss.backward()
                    optimizer.step()
                    if args.set_to_none == None:
                        optimizer.zero_grad()
                    else:
                        optimizer.zero_grad(set_to_none=args.set_to_none)

    mem_trace_gb = trace_memory('peak-main-loop', mem_trace_gb)
    torch.cuda.synchronize()
    elapsed_s = time.time() - before

    torch.cuda.reset_peak_memory_stats() # instantaneous, not peak
    mem_trace_gb = trace_memory('post-main-loop', mem_trace_gb)


data = [{
    'torch-version': re.search('\d+\.\d+\.\d+', torch.__version__).group(),
    'cuda-version': torch.version.cuda,
    'cudnn-version': f'{torch.backends.cudnn.version() / 1000:.3f}',
    'model-arch': args.model_arch,
    'optimize': args.optimize,
    'optimizer': args.optimizer,
    'batch-size': args.batch_size,
    'resolution-h': args.resolution,
    'resolution-w': args.resolution,
    'pixels': args.resolution * args.resolution,
    'iterations': args.iterations,
    'inference-mode': args.inference_mode,
    'no-grad': args.no_grad,
    'cudnn-benchmark': args.cudnn_benchmark,
    'autocast-dtype': args.autocast_dtype,
    'model-cast-dtype': args.model_cast_dtype,
    'channels-last': args.channels_last,
    'pessimization': args.pessimization,
    'jit-script': args.jit_script,
    'eval': args.eval,
    'compile': args.compile,
    'compile-mode': args.compile_mode,
    'set-to-none': args.set_to_none,
    'mem-ckpt': args.mem_ckpt,
    'bnb': args.bnb,
    'paged': args.paged,
    'model-params': num_model_params,
    'first-batch (ms)': round(1000 * first_batch_s, 2),
    'throughput (it/s)': round(args.iterations * args.batch_size / elapsed_s, 2),
    'peak-memory (GB)': peak_mem_gb,
    'delta-memory (GB)': delta_mem_gb,
    'stage': stage,
    **{k: v for k, v in [d.split('=') for d in args.dim]}
} for stage, peak_mem_gb, delta_mem_gb in mem_trace_gb]

out = open(args.csv_path, 'a') if args.csv_path else sys.stdout
if require_header:
    out.write(','.join(data[0]) + '\n')
for datum in data:
    out.write(','.join([str(v) for k, v in datum.items()]) + '\n')
out.flush()

if args.exit_wait_s:
    print(f'chill for {args.exit_wait_s:.0f} secs...')
    time.sleep(args.exit_wait_s)
