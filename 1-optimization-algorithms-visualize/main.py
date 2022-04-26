
import numpy as np
import matplotlib.pyplot as plt
import math
from PIL import Image
import os


def f(x):
    return 4 * x**2 - 10*x + 1
    # return x ** 5 - x ** 4 - 3 * (x ** 3) + 2 * (x ** 2) + x


def gradient(x):
    return 8*x - 10
    # return 5 * (x ** 4) - 4 * (x ** 3) - 9 * (x ** 2) + 4 * x + 1


def plot(x0=0.9, learning_rate=0.01, x_start=-2.0, x_end=2.0, step=100, fig_size=(10, 7), epsilon=math.pow(10, -6)):
    X = np.linspace(x_start, x_end, step)
    y = f(X)
    plt.figure(figsize=fig_size)

    grad_f = gradient(x0)
    i = 1

    while True:
        y_pre = f(x0)
        x0 = x0 - learning_rate * grad_f

        if abs(f(x0) - y_pre) < epsilon or i > 140:
            break

        plt.clf()
        plt.title('       x = {0:.3f}'.format(x0), loc='left')
        plt.title('i = {}'.format(i), loc='right')
        plt.title('learning_rate = {}'.format(learning_rate), loc='center')
        plt.plot(X, y)
        plt.scatter(x0, f(x0))
        plt.savefig(f'img-{i}.png')

        grad_f = gradient(x0)
        i += 1

    images = [Image.open(f'img-{n}.png') for n in range(1, i)]
    images[0].save(f'video-lr_{learning_rate}.gif', save_all=True, append_images=images[1:], duration=2, loop=0)

    [os.remove(file) for file in
     os.listdir('/Users/khnhkd/WorkSpace/some-mini-project-with-AI/optimization-algorithms'
                '-visualize') if file.endswith('.png')]


if __name__ == '__main__':
    plot(learning_rate=0.26, x0=3, x_start=-100000, x_end=100000)
