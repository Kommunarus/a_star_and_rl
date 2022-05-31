import torch
from ppo_net import PPOActor
from dataset import H5Dataset
import numpy as np

device = "cuda:1" if torch.cuda.is_available() else "cpu"

model = PPOActor(input_shape=3, dop_input_shape=5, rnn_hidden_dim=64, n_actions=5).to(device)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

batch_size = 1000
dataloader_train = torch.utils.data.DataLoader(H5Dataset(num_chunk=499, h5_path='train.hdf5'),
                                               shuffle=True, batch_size=batch_size)
dataloader_test = torch.utils.data.DataLoader(H5Dataset(num_chunk=100, h5_path='test.hdf5'),
                                              shuffle=False, batch_size=batch_size)
size = len(dataloader_train.dataset)
size_dataset = len(dataloader_test.dataset)
print('len train', size)
print('len test', size_dataset)

epochs = 50
hidden_state = torch.zeros((batch_size, 1, 64)).to(device)
best_acc = 0

for t in range(epochs):
    print(f"Epoch {t + 1}")
    model.train()
    sum = 0
    for i, batch in enumerate(dataloader_train):
        s, a, a_old = batch['s'].to(device).to(torch.float32), \
                      batch['a'].to(device), \
                      batch['a_old'].to(device)

        if len(s) != batch_size:
            continue

        # Compute prediction error
        pred, hidden_state = model(s, a_old, hidden_state)
        loss = loss_fn(pred, a)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        y2 = np.argmax(pred.detach().cpu().numpy(), 1)

        sum += np.sum(y2 - a.cpu().numpy() == 0)
        if i % 1000 == 0:
                loss, current = loss.item(), i * len(s)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    acc = 100 * sum / size
    print("accuracy train {:.03f}".format(acc))


    model.eval()
    sum = 0
    for i, batch in enumerate(dataloader_test):
        s, a, a_old = batch['s'].to(device).to(torch.float32), \
                      batch['a'].cpu().numpy(), \
                      batch['a_old'].to(device)
        if len(s) != batch_size:
            continue

        # Compute prediction error
        pred, hidden_state = model(s, a_old, hidden_state)
        y2 = np.argmax(pred.detach().cpu().numpy(), 1)

        sum += np.sum(y2 - a == 0)

    acc = 100 * sum / size_dataset
    if best_acc < acc:
        best_acc = acc
        torch.save(model.state_dict(), "model_best.pth")
    print("accuracy val {:.03f}".format(acc))
    print(f"------------------------------")

    torch.save(model.state_dict(), "model_end.pth")