model = model.to(dtype=torch.float16) # cast network parameters to float16
for input in dataloader:
    output = model(input.to(dtype=torch.float16))
    loss = loss_fn(output, target)
    loss.backward()
    optimizer.step()
