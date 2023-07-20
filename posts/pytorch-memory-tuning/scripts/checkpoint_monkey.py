import torch

class CheckpointModule(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return torch.utils.checkpoint.checkpoint(self.module, *args, **kwargs)

def ckpt_monkey(module, pattern_re, ancestor_vn=''):
    for vn, v in module._modules.items():
        full_vn = f'{ancestor_vn}.{vn}' if ancestor_vn else vn
        v = ckpt_monkey(v, pattern_re, full_vn)
        if pattern_re.match(full_vn):
            print('monkey-patching', full_vn)
            setattr(module, vn, CheckpointModule(v))
        else:
            setattr(module, vn, v)
    return module
