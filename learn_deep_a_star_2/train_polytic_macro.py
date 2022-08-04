import torch
from ppo_net import PPOActor_macro as PPOActor
from dataset import H5Dataset
import numpy as np

device = "cuda:0" if torch.cuda.is_available() else "cpu"

model = PPOActor().to(device)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

batch_size = 100
dataloader_train = torch.utils.data.DataLoader(H5Dataset(num_chunk=200, h5_path='train_3.hdf5'),
                                               shuffle=True, batch_size=batch_size)
dataloader_test = torch.utils.data.DataLoader(H5Dataset(num_chunk=50, h5_path='test_3.hdf5'),
                                              shuffle=False, batch_size=batch_size)
size = len(dataloader_train.dataset)
size_dataset = len(dataloader_test.dataset)
print('len train', size)
print('len test', size_dataset)

epochs = 5
# hidden_state = torch.zeros((batch_size, 1, 64)).to(device)
best_acc = 0

for t in range(epochs):
    print(f"Epoch {t + 1}")
    model.train()
    sum = 0
    for i, batch in enumerate(dataloader_train):
        s, a = batch['s'].to(device).to(torch.float32), \
                      batch['a'].to(device).to(torch.long)

        if len(s) != batch_size:
            continue

        # Compute prediction error
        pred, _ = model(s, device)
        loss = loss_fn(pred, a)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        y2 = np.argmax(pred.detach().cpu().numpy(), 1)

        sum += np.sum(y2 - a.cpu().numpy() == 0)
        if i % 1000 == 0 and i != 0:
            loss, current = loss.item(), i * len(s)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]", end=' ')

            acc = 100 * sum / current
            print("accuracy train {:.03f}".format(acc))
            torch.save(model.state_dict(), "model_end_macro.pth")
    acc = 100 * sum / size
    print("accuracy train {:.03f}".format(acc))


    model.eval()
    sum = 0
    for i, batch in enumerate(dataloader_test):
        s, a = batch['s'].to(device).to(torch.float32), \
                      batch['a'].to(device).to(torch.long)
        if len(s) != batch_size:
            continue

        # Compute prediction error
        pred = model(s)
        y2 = np.argmax(pred.detach().cpu().numpy(), 1)

        sum += np.sum(y2 - a.cpu().numpy() == 0)

    acc = 100 * sum / size_dataset
    if best_acc < acc:
        best_acc = acc
        torch.save(model.state_dict(), "model_best_macro.pth")
    print("accuracy val {:.03f}".format(acc))
    print(f"------------------------------")
