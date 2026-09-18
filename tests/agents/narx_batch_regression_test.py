"""Independent examples must remain independent in standalone NARX learning."""

import copy

import pytest
import torch

from tensoraerospace.agent.narx.model import NARX


@pytest.mark.parametrize("shape", [(5,), (2, 3)])
def test_batched_forward_and_gradients_match_individual_examples(shape):
    torch.manual_seed(11)
    model = NARX(3, 8, 2).double()
    inputs = torch.randn(*shape, 3, dtype=torch.float64, requires_grad=True)
    previous = torch.randn(*shape, 2, dtype=torch.float64, requires_grad=True)
    expected = torch.stack(
        [model(x, y) for x, y in zip(inputs.reshape(-1, 3), previous.reshape(-1, 2))]
    ).reshape(*shape, 2)
    actual = model(inputs, previous)
    torch.testing.assert_close(actual, expected)
    variables = (inputs, previous, *model.parameters())
    actual_gradients = torch.autograd.grad(
        actual.square().mean(), variables, retain_graph=True
    )
    expected_gradients = torch.autograd.grad(expected.square().mean(), variables)
    for a, b in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(a, b)


def test_batched_training_matches_scalar_reference_optimizer_step():
    torch.manual_seed(29)
    model = NARX(1, 8, 2)
    reference = copy.deepcopy(model)
    inputs, previous, target = torch.randn(7, 1), torch.randn(7, 2), torch.randn(7, 2)
    expected = torch.stack([reference(x, y) for x, y in zip(inputs, previous)])
    reference_loss = reference.train(expected, target)
    loss = model.train(model(inputs, previous), target)
    assert loss == pytest.approx(reference_loss, rel=1e-6)
    for a, b in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(a, b)


@pytest.mark.parametrize("target_shape", [(3,), (1, 1), (3, 2)])
def test_loss_rejects_broadcasting_before_mutating_parameters(target_shape):
    model = NARX(1, 8, 1)
    predicted = torch.stack(
        [model(torch.tensor([x]), torch.zeros(1)) for x in (1.0, 2.0, 3.0)]
    )
    before = [p.detach().clone() for p in model.parameters()]
    with pytest.raises(ValueError, match="shape"):
        model.train(predicted, torch.ones(target_shape))
    assert not model.optimizer.state
    for a, b in zip(model.parameters(), before):
        torch.testing.assert_close(a, b)


def test_single_example_and_module_modes_remain_compatible():
    model = NARX(3, 8, 2)
    assert model.train(False) is model
    assert not model.training
    actual = model(torch.ones(3), torch.zeros(2))
    expected = model.output_layer(
        torch.tanh(model.input_layer(torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0])))
    )
    torch.testing.assert_close(actual, expected)
    assert model.train(True) is model
    assert model.training
    assert model.eval() is model


def test_checkpoint_preserves_batch_inference_and_optimizer_continuation(tmp_path):
    torch.manual_seed(47)
    model = NARX(1, 8, 2)
    inputs, previous, target = torch.randn(8, 1), torch.randn(8, 2), torch.randn(8, 2)
    model.train(model(inputs, previous), target)
    directory = model.save(tmp_path, save_gradients=True)
    loaded = NARX.load(directory, load_gradients=True)
    torch.testing.assert_close(loaded(inputs, previous), model(inputs, previous))
    a = model.train(model(inputs, previous), target)
    b = loaded.train(loaded(inputs, previous), target)
    assert a == pytest.approx(b)
    for p, q in zip(model.parameters(), loaded.parameters()):
        torch.testing.assert_close(p, q)
