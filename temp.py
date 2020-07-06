import torch

class LayerNorm(torch.nn.Module):
  def __init__(self, channels, eps=1e-4):
      super().__init__()
      self.channels = channels
      self.eps = eps

      self.gamma = torch.nn.Parameter(torch.ones(channels))
      self.beta = torch.nn.Parameter(torch.zeros(channels))

  def forward(self, x):
    n_dims = len(x.shape)
    mean = torch.mean(x, 1, keepdim=True)
    variance = torch.mean((x -mean)**2, 1, keepdim=True)

    x = (x - mean) * torch.rsqrt(variance + self.eps)

    shape = [1, -1] + [1] * (n_dims - 2)
    x = x * self.gamma.view(*shape) + self.beta.view(*shape)
    return x

if __name__ == "__main__":
    conv = torch.nn.Conv1d(16, 8, 3, 1, 1)
    ln1 = torch.nn.LayerNorm(8, eps= 1e-4)
    ln2 = LayerNorm(8)

    x = torch.randn(3, 16, 5)
    x = conv(x)
    y1 = ln1(x.transpose(2,1)).transpose(2,1)
    y2 = ln2(x)

    print(y1 - y2)

    print(y1.shape)
    print(y2.shape)

