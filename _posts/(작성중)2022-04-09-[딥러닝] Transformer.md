---
use_math: true
---
## 1. 입력 임베딩
트렌스포머의 인코더, 디코더는 512차원의 벡터를 다룬다. 따라서 각 던어를 512차원의 벡터로 변환해야한다.

## 2. Positional Encoding (Positional Encoding)
트랜스포머의 self-Attention 기법에서 순차적으로 데이터를 다루지 않기 때문에 문장 안에서의 단어 순서 정보를 인코딩해야한다. 순서 정보를 전달해야 단어의 앞뒤 관계 즉 문맥에 대한 정보를 고려한 모델을 생성할 수 있기 때문이다.

각 포지션 임베딩 값을 구하는 식은 아래와 같다.
$$
PE(pos, 2i) = sin(pos/10000^{2i/d_{model}})
$$
$$
PE(pos, 2i+1) = cos(pos/10000^{2i/d_{model}})
$$
수식 1 Positional Encoding

그리고 input, positional 인코딩 값을 더하여 Encoder의 입력값을 만든다.

## 3. Encoder
2번에서 만들어진 벡터를 Encoder에 입력하는 단계이다.
![image](https://user-images.githubusercontent.com/101251439/162553630-a7695c7d-c4dd-4a67-8585-d1a1d1040d7f.png)
그림1 Transformer Encoder 구조

그림1과 같이 입력값은 Multi-Head Attention을 거쳐 FeedForward Neural Network 과정을 거쳐 Output을 만든다. (Self-Attention, FeedForward 위에는 Add & Norm 레이어가 있다.)

## - 3-1. Muti-Head Attention
Multi-Head Attention이란 Self-Attension과 같은 의미로 해석되며 Self-Attension 동작에 대해 먼저 알아야한다. Multi-Head Attention은 아래의 수식을 통해 나타낼 수 있다.

$$
Attention(Q,K,V) = softmax(\frac{QK^T}{\sqrt{d_{k}}})V
$$
수식2. Scaled Dot-Product Attention

$$
head_{i} = Attention(QW_{i}^{Q},KW_{i}^{K},VW_{i}^{V})
$$
수식3. 한개의 head

$$
MultiHead(Q,K,V) = Concat(head_{1},...,head_{h})W^{0}
$$
수식4. MultiHead Attention


### Self-Attention
Self-Attension 은 "I want to be Data Scientist." 라는 문장이 있다고 가정하면, 단어 한개를 선택하여 문장의 다른 단어와의 관계를 보는 것이다. I를 선택했을 때 wnat, to, be, Data, Scientist 단어 각각과의 관계를 본다.


![image](https://user-images.githubusercontent.com/101251439/162554179-5796f541-5e1f-4019-a2a1-f1512cda81ba.png)
그림2 Scaled Dot-Product Attention

문장의 한단어를 선택해서 다른 단어와의 관계를 구하는 방법은 그림2의 Scaled Dot-Product Attention을 사용한다. Attention의 수식은 아래의 수식2와 같다.

### Query, Key, Value
앞의 수식2에서 Q, K, V에 대해 알아보자.
Q는 Query이므로 다른 단어와의 관계를 알고 싶은 문장내 한단어이다.
K는 Key로 Q와의 관계를 계산할 대상이다.
예를 들어 "I am a boy" 라는 문장에서 I의 단어와 am 단어의 관계를 계산하고자 할때 I가 Query이고 am이 Key이다.
V는 Value로 Q, K와의 가중치를 의미한다.


 "I am Min" 이라는 단어를 토큰화하고 앞서말한 input, position 인코등 값을 만들어 아래와 같은 벡터가 만들어 졌다고 가정하자.
