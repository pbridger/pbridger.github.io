torch.backends.cudnn.benchmark = True
output = model(input) # benchmarking occurs JIT during first execution
output = model(input) # subsequent executions use the fastest available kernels
