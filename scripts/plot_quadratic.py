import matplotlib.pyplot as plt
import numpy as np

# Generate data for the quadratic equation y = x^2
x = np.linspace(-10, 10, 400)
y = x**2

# Create the plot
plt.figure(figsize=(8, 6))
plt.plot(x, y, label='y = x^2')
plt.title('Quadratic Plot')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.legend()

# Save the plot to /tmp
output_path = './tmp/quadratic_plot.png'
plt.savefig(output_path)
print(f"Plot saved to {output_path}")
