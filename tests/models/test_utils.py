#pylint: disable=not-callable

import torch

from publicplan.models.utils import ccr, pad_tensors


def test_ccr_shape():
    x = torch.randn(100, 10)
    x_red, fp = ccr(x)
    assert fp.size() == (10, )
    assert x_red.size() == x.size()


def test_ccr_value():
    x = torch.diag(torch.tensor([3.0, 2.0, 1.0]))
    x_red, fp = ccr(x)
    assert (fp == torch.tensor([1, 0, 0])).all()
    assert (x_red == torch.diag(torch.tensor([0.0, 2.0, 1.0]))).all()


def test_pad_tensors():
    torch.manual_seed(0)
    t1 = torch.randn(2, 1, 3)
    t2 = torch.randn(2, 2, 1)

    result = pad_tensors([t1, t2], dims=[1, 2])
    assert result[0].shape == (2, 2, 3)
    assert result[1].shape == (2, 2, 3)

    assert (result[0][:, :1] == t1).all()
    assert (result[0][:, 1:] == 0).all()
    assert (result[1][:, :, :1] == t2).all()
    assert (result[1][:, :, 1:] == 0).all()
