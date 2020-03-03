import json
import matplotlib.pyplot as plt

# Put colors in a dictionary (collection of variables)
colors_out = {
    'grey': '#cdd0d0',
    'blue': '#20547C',
    'orange': '#C07F28'
}

# Save the colours to a (human-readable) file
with open('colors.json', 'w') as f:
    json.dump(colors_out, f, indent=2)

# Plot the colours so we know which is which
plt.plot(1, 1, color=colors_out['orange'], label='orange', marker='o')
plt.plot(2, 1, color=colors_out['grey'], label='grey', marker='o')
plt.plot(3, 1, color=colors_out['blue'], label='blue', marker='o')

plt.legend()
plt.show()

# Example of loading in some colours
with open('colors.json', 'r') as f:
    colors_in = json.load(f)

plt.plot(1, 2, color=colors_in['orange'], label='orange', marker='o')
plt.plot(2, 2, color=colors_in['grey'], label='grey', marker='o')
plt.plot(3, 2, color=colors_in['blue'], label='blue', marker='o')

plt.legend()
plt.show()

# Some fancy python way of plotting all the colors:
for i, (key, value) in enumerate(colors_in.items()):
    plt.plot(i, 3, label=key, color=value, marker='o')
plt.legend()
plt.show()