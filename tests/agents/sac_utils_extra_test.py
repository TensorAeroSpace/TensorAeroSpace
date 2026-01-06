import torch

from tensoraerospace.agent.sac import utils


class _DummyNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(2, 2, bias=False)


def test_soft_update_and_hard_update():
    src = _DummyNet()
    tgt = _DummyNet()

    with torch.no_grad():
        src.lin.weight.fill_(1.0)
        tgt.lin.weight.fill_(0.0)

    utils.soft_update(tgt, src, tau=0.5)
    assert torch.allclose(tgt.lin.weight, torch.full_like(tgt.lin.weight, 0.5))

    utils.hard_update(tgt, src)
    assert torch.allclose(tgt.lin.weight, torch.full_like(tgt.lin.weight, 1.0))


def test_logsumexp_none_dim_matches_manual():
    x = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
    out = utils.logsumexp(x)
    # compare with torch.logsumexp over all elements
    ref = torch.logsumexp(x.view(-1), dim=0)
    assert torch.allclose(out, ref)
