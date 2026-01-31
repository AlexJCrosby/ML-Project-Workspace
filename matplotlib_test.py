import matplotlib
import matplotlib.pyplot as plt

print("Matplotlib version:", matplotlib.__version__)

plt.plot([0, 1, 2], [0, 1, 4])
plt.title("Matplotlib test")
plt.savefig("matplotlib_test.png")  # saves instead of showing a window
print("Saved matplotlib_test.png")
