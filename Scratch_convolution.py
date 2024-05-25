import numpy as np
from scipy import misc
import matplotlib.pyplot as plt

i = misc.face(gray=True)
plt.grid(False)
plt.gray()
plt.axis('off')
plt.imshow(i)
plt.show()

i_new = np.copy(i)
size_x = i_new.shape[0]  # width
size_y = i_new.shape[1]  # height

Filter = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
for x in range(1, size_x - 1):
    for y in range(1, size_y - 1):
        convolution = 0.0
        convolution = convolution + (Filter[0][0] * i[x - 1][y - 1])
        convolution = convolution + (Filter[0][1] * i[x - 1][y])
        convolution = convolution + (Filter[0][2] * i[x - 1][y + 1])
        convolution = convolution + (Filter[1][0] * i[x][y - 1])
        convolution = convolution + (Filter[1][1] * i[x][y])
        convolution = convolution + (Filter[1][2] * i[x][y + 1])
        convolution = convolution + (Filter[2][0] * i[x + 1][y - 1])
        convolution = convolution + (Filter[2][1] * i[x + 1][y])
        convolution = convolution + (Filter[2][2] * i[x + 1][y + 1])
        if convolution < 0:
            convolution = 0
        if convolution > 255:
            convolution = 255
        i_new[x][y] = convolution
        # print(x, y, i_new[x][y])

plt.grid()
plt.grid(False)
plt.imshow(i_new)
plt.show()

# Max pool

new_x = int(size_x / 2)
new_y = int(size_y / 2)
new_img = np.zeros((new_x, new_y))
for x in range(0, size_x, 2):
    for y in range(0, size_y, 2):
        pixels = []
        pixels.append(i_new[x, y])
        pixels.append(i_new[x+1, y])
        pixels.append(i_new[x, y+1])
        pixels.append(i_new[x+1, y+1])
        new_img[int(x/2)][int(y/2)] = max(pixels)

# plot the image, note the size of the axes -- now 256 pixels instead of 512
plt.gray()
plt.grid(False)
plt.imshow(new_img)
plt.show()

