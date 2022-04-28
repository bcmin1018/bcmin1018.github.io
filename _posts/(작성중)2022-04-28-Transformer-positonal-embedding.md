---
layout: single
title: "[딥러닝] Transformer positional encoding"
categories: deeplearning
tag: [python, deeplearning, transformer]
---
# Transforemr positional encoding-pytorch(파이토치)

Transformer 구조 중 가장 먼저 input에 대해 알아보았다.
Transformer는 RNN과 달리 순차적으로 데이터를 처리하지 않기 때문에 따로 위치에 대한 정보를 입력 값에 적용해야한다.
이때 위치에 대한 정보는 positional encoding이라고 하고 해당 값을 token 임배딩 벡터에 더해주면된다. 

![image](https://user-images.githubusercontent.com/101251439/165789362-8a70e693-51b1-4625-9280-6d30009d040b.png)

positional encoding은 다음과 같은 수식으로 구할 수 있다. encoding을 4차원으로 한다면 [0, 1, 2, 3] 홀수 자리에 위치한 0, 2 는 cos를 적용하고 짝수는 sin을 적용한다.
![image](https://user-images.githubusercontent.com/101251439/165788843-c1491f37-baf5-46cf-8510-4864116e1102.png)

```python
class PositionalEncoding(nn.Module):
  def __init__(self, max_len, dim_model):
    super(PositionalEncoding, self).__init__()
    self.pos_encoding = torch.zeros(max_len, dim_model)
    pos = torch.arange(0, max_len, dtype = torch.float).view(-1, 1)
    
    _2i = torch.arange(0, dim_model, step=2).float()

    self.pos_encoding[:, 0 :: 2] = torch.sin(pos/(10000**(_2i/dim_model)))
    self.pos_encoding[:, 1 :: 2] = torch.cos(pos/(10000**(_2i/dim_model)))

  def forward(self, token_embedding):
    return token_embedding + self.pos_encoding
```