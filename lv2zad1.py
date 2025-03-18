
import numpy as np
import matplotlib.pyplot as plt 

x = np.array([1, 3, 3, 2, 1], dtype=np.float64)
y = np.array([1, 1, 2, 2, 1], dtype=np.float64)

print("x:", x)
print("y:", y)

plt.plot(x, y, color='b', linestyle='-', linewidth=1, marker='.', markersize=7, markerfacecolor='blue')

plt.xlim(0, 5)
plt.ylim(0, 5)

plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.title('Matplotlib')

plt.savefig('graf.png')

plt.show()