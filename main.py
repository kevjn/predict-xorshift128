import argparse

import torch
import numpy as np

from tqdm import trange

def xorshift128(seed):
    '''xorshift
    https://en.wikipedia.org/wiki/Xorshift
    '''
    state = seed
    while True:
        t = state[0] ^ state[0] << np.uint32(11)
        t = t ^ t >> 8
        state = np.roll(state, -1)
        state[-1] = t ^ state[-2] ^ state[-2] >> np.uint32(19)
        yield state[-1]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=int, default=2_000_000, help='number of training samples')
    args = parser.parse_args()

    g = xorshift128(np.array([123456789, 362436069, 521288629, 88675123], dtype=np.uint32))

    # initialize weights
    w0, b0 = torch.empty(128, 1024, requires_grad=True), torch.zeros(1024, requires_grad=True)
    w0 = torch.nn.init.xavier_uniform_(w0) # xavier uniform is much better than random
    w1, b1 = torch.empty(1024, 32, requires_grad=True), torch.zeros(32, requires_grad=True)
    w1 = torch.nn.init.xavier_uniform_(w1)

    # generate training data
    train = np.array([next(g) for _ in trange(args.train, desc="Generating RNG samples")])
    train_bits = (train[:,None] & (2 ** np.arange(32, dtype="uint32")) > 0).astype(int)
    train_frames = np.lib.stride_tricks.sliding_window_view(train_bits, (5,32))[:, 0, :, :]

    # convert to (X,y) train data
    X, y = train_frames[:, :-1, :], train_frames[:, -1, :]
    X = X.reshape(-1, 4*32)
    X, y = torch.from_numpy(X.copy()).float(), torch.from_numpy(y.copy()).float()

    optim = torch.optim.Adam([w0, b0, w1, b1])
    loss_fn = torch.nn.BCELoss(reduction='sum')

    # train model
    for epoch in (t := trange(100, desc=f"Training model (epoch 1)")):
        perm = torch.randperm(X.size()[0])

        for i in range(0, X.size()[0], 512):
            optim.zero_grad()
            indices = perm[i:i+512]
            batch_x, batch_y = X[indices], y[indices]

            out = torch.sigmoid(torch.relu(batch_x @ w0 + b0) @ w1 + b1)
            loss = loss_fn(out, batch_y)

            loss.backward()
            optim.step()

        acc = (out.round() == batch_y).float().mean()
        t.set_description(f"Training model (epoch {epoch+1})")
        t.set_postfix_str(s=f"{acc * 100:.2f}% bitwise accuracy")

    seed = np.random.randint(2**32 - 1, size=4, dtype=np.uint32)
    print(f"Testing trained model, new random seed: {str(seed)}")
    g = xorshift128(seed)
    
    correct = 0
    for _ in range(100):
        X = ((np.array([next(g) for _ in range(4)])[:,None] & (2 ** np.arange(32))) > 0).astype(np.float32)
        X = torch.from_numpy(X.ravel())
        y = torch.sigmoid(torch.relu(X @ w0 + b0) @ w1 + b1).round()
        pred = (y.detach().numpy().astype(int) * (2**np.arange(32, dtype="uint32"))).sum()

        correct += pred == next(g)

    print(f"Correctly predicted {correct}/100 samples")