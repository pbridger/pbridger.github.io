model = torch.hub.load(...)
model = ckpt_monkey(model, re.compile('layer1$')) # checkpoint all modules named "layer1"
