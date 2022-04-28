---
layout: single
title: "[CV] 합성곱 Featuremap 코딩구현"
categories: [deeplearning, cv]
tag: [python, deeplearning, cv]
toc: true
---

# 합성곱 코딩 구현
3개의 이미지 배열과 3개의 kernel을 갖고 

```python
import numpy as np
channel1 = np.array([[0,1,1,1,0], [1,1,0,0,1], [1,1,0,0,1], [0,1,1,1,0], [1,0,1,1,0]])
channel2 = np.array([[0,1,1,1,0], [1,1,1,0,1], [1,1,0,0,1], [1,1,1,1,1], [0,0,1,1,0]])
channel3 = np.array([[0,1,1,1,1], [1,1,0,0,1], [1,1,0,0,1], [0,1,1,1,1], [0,0,1,1,0]])

kernel1 = np.array([[-1, -1, 1], [-1, 1, 0], [1, 0, 0]])
kernel2 = np.array([[0, 0, 1], [0, 1, -1], [1, -1, -1]])
kernel3 = np.array([[-1, -1, 1], [-1, 1, -1], [1, -1, -1]])
```
```python
def convolution(channel, kernel, padding = True):
  feature_map = []

  cx, cy = np.shape(channel)
  kx, ky = np.shape(kernel)

  #padding
  if padding is True:
    channel = np.insert(channel, 0, np.zeros(cx), axis = 0)
    channel = np.insert(channel, cy+1, np.zeros(cx), axis = 0)
    channel = np.insert(channel, 0, 0, axis = 1)
    channel = np.insert(channel, cx+1, 0, axis = 1)
    
  new_cx, new_cy = np.shape(channel)
  for i in range(new_cx - kx +1):
    for j in range(new_cy - ky +1):
      feature_map.append((channel[i:i+kx, j:j+ky]*kernel).sum())
  result = np.array(feature_map).reshape(cx,cy)
  print("피처맵 : \n {}".format(result))
  return result
  ```

```python
cv1 = convolution(channel1, kernel1, padding=True)
cv2 = convolution(channel2, kernel2, padding=True)
cv3 = convolution(channel3, kernel3, padding=True)
```