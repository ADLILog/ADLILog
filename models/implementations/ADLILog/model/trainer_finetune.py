import numpy as np
import torch
import time

fp = open("experiment_output.txt", "a+")


def run_train(dataloader, model, loss_compute, step_size=10):
    """Standard Training and Logging Function"""
    start = time.time()
    total_loss = 0
    i = 1
    for i, batch in enumerate(dataloader):
        b_input, b_labels = batch

        out = model.forward(b_input.cuda(), b_labels.cuda(), None, None)
#         out_p = model.generator(out)
#         dist = torch.sum((out[:, 0, :] - model.c) ** 2, dim=1)
#         loss = loss_compute(out, b_labels.cuda(), dist)
        dist = 0
        loss = loss_compute(out, b_labels.cuda(), dist)
        total_loss += loss

        if i % step_size == 0:
            print("Epoch Train Step: %d / %d Loss: %f" %
                  (i, len(dataloader), loss), end='\r')

    print("Epoch train Loss: %f" %
          (total_loss / i), " " * 50)
    print("Epoch train Loss: %f" %
          (total_loss / i), " " * 50, file=fp)
    fp.flush()

    return total_loss


def run_test(dataloader, model, loss_compute, step_size=10):
    """Standard Training and Logging Function"""

    preds = []
    distances = []
    total_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            b_input, b_labels = batch

            out = model.forward(b_input.cuda(), b_labels.cuda(),
                                None, None)
            out_p = model.generator(out)
#             dist = torch.sum((out[:, 0, :] - model.c) ** 2, dim=1)
#             loss = loss_compute(out, b_labels.cuda(), dist)
            dist = 0
            loss = loss_compute(out, b_labels.cuda(), dist)
            total_loss += loss
            if i % step_size == 0:
                print("Epoch Test Step: %d / %d Loss: %f" %
                      (i, len(dataloader), loss), end='\r')
            tmp = out_p.cpu().numpy()
            preds += list(np.argmax(tmp, axis=1))
#             distances += list(dist.cpu().numpy())
            distances += list(np.array(dist).reshape(-1, 1))

    print("Epoch test Loss: %f" %
          (total_loss / i), " " * 50, file=fp)
    print(file=fp)
    print()
    fp.flush()

    return preds, distances
