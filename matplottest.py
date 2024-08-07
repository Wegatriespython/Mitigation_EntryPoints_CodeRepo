import matplotlib.pyplot as plt
import numpy as np

def test_matplotlib():
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    
    plt.figure()
    plt.plot(x, y)
    plt.title("Test Plot")
    plt.savefig("test_plot.png")
    plt.close()
    print("Test plot saved successfully.")

if __name__ == "__main__":
    test_matplotlib()