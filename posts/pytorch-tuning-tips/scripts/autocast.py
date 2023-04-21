with torch.autocast("cuda"):
    output = model(input)
    loss = loss_fn(output, target)
loss.backward()
optimizer.step()
