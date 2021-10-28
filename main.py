import torch
import numpy as np

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
    g = xorshift128(np.array([123456789, 362436069, 521288629, 88675123], dtype=np.uint32))

    # initialize weights
    w0, b0 = torch.empty(128, 1024, requires_grad=True), torch.zeros(1024, requires_grad=True)
    w0 = torch.nn.init.xavier_uniform(w0) # xavier uniform is much better than random
    w1, b1 = torch.empty(1024, 32, requires_grad=True), torch.zeros(32, requires_grad=True)
    w1 = torch.nn.init.xavier_uniform(w1)

    # generate training data (1000 RNG samples)
    train = np.array([next(g) for _ in range(1_0000)])
    train_bits = (train[:,None] & (2 ** np.arange(32, dtype="uint32")) > 0).astype(int)
    train_frames = np.lib.stride_tricks.sliding_window_view(train_bits, (5,32))[:, 0, :, :]

    # convert to (X,y) train data
    X, y = train_frames[:, :-1, :], train_frames[:, -1, :]
    X = X.reshape(-1, 4*32)
    X, y = torch.from_numpy(X).float(), torch.from_numpy(y).float()

    optim = torch.optim.Adam([w0, b0, w1, b1])

    loss_fn = torch.nn.BCELoss(reduction='sum')

    # train model
    for epoch in range(1000):
        optim.zero_grad()
        out = torch.sigmoid(torch.relu(X @ w0 + b0) @ w1 + b1)
        loss = loss_fn(out, y)
        loss.backward()
        optim.step()

        acc = (out.round() == y).float().mean()
        print(f"acc: {acc * 100} %")

    # TODO: test model (use a different seed)
    seed = np.random.randint(2**32 - 1, size=4, dtype=np.uint32)
    g = xorshift128(seed)