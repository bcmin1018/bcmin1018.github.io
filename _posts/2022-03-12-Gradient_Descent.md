---
layout: single
title: "[딥러닝] 경사하강법"
use_math: true
---

딥러닝에서 가장 좋은 그래프는 로스의 합이 가장 작은 그래프이다. 그렇다면 가장 좋은 그래프는 기울기를 변경하면서 찾아낼 수 있다. 기울기는 weight(w)이며 에러가 작은 w를 구하는 방법을 'Gradient descent' 라고 한다.


"w가 0에 가까울 수록 로스가 적다."

아래의 그림에서 Cost는 전체 로스이고 랜덤한 시작점 부터 w가 0에 가까운 지점을 찾는 방식이다. 0에 가까울 수록 전체 로스가 적은 그래프에 가까운 w이라고 할 수 있다.

![image](https://user-images.githubusercontent.com/101251439/158021715-c5174c4d-2a7a-49ae-b2af-b1f7a6b4fc6d.png)
[gradient descent] 출처: https://morioh.com/p/b0fefc78f330

여기서 Learning step 은 흔히 Learning Rate(lr)라고 불리며 lr이 너무 크면 최저점을 지나칠 수도 있으므로 작게 설정하여 w가 최소인 지점을 찾아야한다. 이를 위한 식은

w^{1} = w^{0} - lr\frac{\partial E }{\partial W}\left.\begin{matrix}
 \\  
\end{matrix}\right| w = w^{0}