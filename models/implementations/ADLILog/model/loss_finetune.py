import torch


class SimpleLossCompute:
    """A simple loss compute and train function."""

    def __init__(self, model, criterion, opt=None, is_test=False, generator=False):
        self.model = model
        self.criterion = criterion
        self.opt = opt
        self.is_test = is_test
        self.bce = torch.nn.CrossEntropyLoss(weight=torch.tensor([0.3, 1.0]).cuda())
        self._lambda = 0.05
        self.generator = generator

    def __call__(self, x, y, dist):
#         loss = torch.mean((1 - y) * torch.sqrt(dist) - (y) * torch.log(1 - torch.exp(-torch.sqrt(dist))))
        if self.generator:
            out = self.model.generator(x)
            loss = self.bce(out, y)
            

        if not self.is_test:
            loss.backward()
            if self.opt is not None:
                self.opt.step()
                self.opt.zero_grad()

        return loss.item()

 