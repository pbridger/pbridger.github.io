model = model.to(memory_format=torch.channels_last)
with torch.autocast('cuda'):
    output = model(input.to(memory_format=torch.channels_last))
