import torch


class NoamOpt:
    def __init__(self,
                 model_size,  # d_model
                 factor,  # 2
                 warmup,  # 4000
                 optimizer,
                 last_state=None,
                 ):
        self.optimizer = optimizer
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        if last_state is not None:
            self.n_step = last_state['n_step']
            self.p_rate = last_state['p_rate']
        else:
            self.n_step = 0
            self.p_rate = 0

    def step(self):
        self.n_step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self.p_rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        if step is None:
            step = self.n_step
        f = min(step ** (-0.5), step * self.warmup ** (-1.5))
        return self.factor * (self.model_size ** (-0.5) * f)

    def state_dict(self):
        return {'n_step': self.n_step, 'p_rate': self.p_rate}


class SimpleLossCompute:
    def __init__(self, generator, criterion=None, opt=None):
        self.generator = generator
        if criterion is None:
            if generator.is_classifier:
                criterion = torch.nn.CrossEntropyLoss()
                # criterion = torch.nn.NLLLoss()
            else:
                criterion = torch.nn.MSELoss()
        self.criterion = criterion
        self.opt = opt

    def __call__(self,
                 # (batch,) for regression
                 # or (batch,n) # for classification
                 x: torch.Tensor,
                 y: torch.Tensor,  # (batch,)
                 norm: torch.Tensor  # batch.n_seqs
                 ):
        loss = self.criterion(x, y) / norm
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        nloss = loss.item() * norm
        if self.generator.is_classifier:
            return nloss
        return nloss * 10000.0


def build_adam_opt(model,
                   last_state=None,
                   ):
    opt = torch.optim.Adam(model.parameters(),
                           lr=0,
                           betas=(0.9, 0.98),
                           eps=1e-9
                           )
    if last_state is not None:
        opt.load_state_dict(last_state)
    return opt


def build_train_val_loss_compute(model,
                                 model_config,
                                 optimizer_state=None,
                                 noam_state=None,
                                 ):
    optimizer = build_adam_opt(model, last_state=optimizer_state)
    model_opt = NoamOpt(model_config.d_model, 1, 4000, optimizer, last_state=noam_state)
    train_loss_compute = SimpleLossCompute(model.generator, None, model_opt)
    val_loss_compute = SimpleLossCompute(model.generator, None, None)
    return optimizer, train_loss_compute, val_loss_compute


def build_val_loss_compute(model):
    val_loss_compute = SimpleLossCompute(model.generator, None, None)
    return val_loss_compute
