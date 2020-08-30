# Original referece: https://github.com/janfreyberg/pytorch-revgrad
# Only weight paramter is added. 

import torch

class Func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, weight):
        ctx.save_for_backward(input_)
        ctx.weight = weight
        output = input_
        return output

    @staticmethod
    def backward(ctx, grad_output):  # pragma: no cover
        grad_input = None
        if ctx.needs_input_grad[0]:
            grad_input = -ctx.weight * grad_output
        return grad_input, None

class GRL(torch.nn.Module):
    def __init__(self, weight= 1.0):
        """
        A gradient reversal layer.
        This layer has no parameters, and simply reverses the gradient
        in the backward pass.
        """
        super(GRL, self).__init__()
        self.weight = weight

    def forward(self, input_):
        return Func.apply(
            input_,
            torch.FloatTensor([self.weight]).to(device= input_.device)
            )