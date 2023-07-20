from torch.utils import checkpoint as ckpt
# where module is part of a model:
result = module(*args, **kwargs) # regular invocation with default activation caching
result = ckpt.checkpoint(module, *args, **kwargs) # checkpointed invocation
