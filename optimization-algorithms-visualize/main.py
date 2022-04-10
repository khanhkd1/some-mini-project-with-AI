# import thư viện
import numpy as np
import matplotlib.pyplot as plt
import math
from PIL import Image
import os


# khai báo hàm f(x)
def f(x):
    return x**5 - x**4 - 3*(x**3) + 2*(x**2) + x


# khai báo đạo hàm (gradient)
def gradient(x):
    return 5*(x**4) - 4*(x**3) - 9*(x**2) + 4*x + 1


# Vẽ đồ thị hàm f(x)
X = np.linspace(-1.5, 2, 100)  # vẽ đồ thị trong khoảng x chạy từ -4 -> 2, lấy 100 số
y = f(X)
fig = plt.figure(figsize=(10, 7))

# Tiến hành tối ưu bằng phương pháp Gradient Descent
x = 0.9  # điểm bắt đầu của x
learning_rate = 0.01
epsilon = math.pow(10, -6)  # giá trị 10^(-6) cực nhỏ để làm điều kiện dừng
grad_f = gradient(x)
i = 1


while True:
    y_pre = f(x)
    x = x - learning_rate*grad_f  # cập nhật x

    if abs(f(x) - y_pre) < epsilon or i > 300:  # giá trị y cập nhật rất nhỏ thì dừng lại
        break

    plt.clf()
    plt.title('x = {0:.3f}'.format(x), loc='left')
    plt.title('i = {}'.format(i), loc='right')
    plt.title('learning_rate = {}'.format(learning_rate), loc='center')
    plt.plot(X, y)
    plt.scatter(x, f(x))
    plt.savefig(f'img-{i}.png')

    grad_f = gradient(x)
    i += 1

# Ghép các ảnh thành 1 video gif
images = [Image.open(f'img-{n}.png') for n in range(1, i)]
images[0].save(f'video-lr_{learning_rate}.gif', save_all=True, append_images=images[1:], duration=2, loop=0)

# Xoá ảnh sau khi có video
[os.remove(file) for file in os.listdir('/Users/khnhkd/WorkSpace/some-mini-project-with-AI/optimization-algorithms'
                                        '-visualize') if file.endswith('.png')]
