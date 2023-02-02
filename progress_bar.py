import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


def progress_bar(train_function, num_epochs, train_dataloader):
    for epoch in range(num_epochs):
        loop = tqdm(train_dataloader)

        for idx, (x, y) in enumerate(loop):
            loss, acc = train_function(x, y)

            loop.set_description(f"Epoch [{epoch}/{num_epochs}]")
            loop.set_postfix(loss=loss, acc=acc)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn((1000, 1000))
    y = torch.randint(0, 9, size=(1000, 1))

    ds = TensorDataset(x, y)
    loader = DataLoader(ds, batch_size=64)

    model = nn.Sequential(
        nn.Linear(1000, 10),
        nn.ReLU(),
        nn.Linear(10, 10)
    ).to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_func = nn.CrossEntropyLoss()

    def test_train_function(x, y):
        x, y = x.to(device), y.to(device)

        preds = model(x)

        preds = preds.unsqueeze(-1)

        loss = loss_func(preds, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc = (torch.argmax(preds, dim=1) == y).sum()/len(x)*100

        return loss.item(), acc.item()

    num_epochs = 50

    progress_bar(test_train_function, num_epochs, loader)
