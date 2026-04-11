import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

fig, ax = plt.subplots(figsize=(14, 3))
ax.axis('off')

steps = [
    {'text': 'Endoszkópos\nvideójel', 'color': '#f0f0f0'},
    {'text': 'Képkocka\nkiemelése\n(Trigger)', 'color': '#fff2cc'},
    {'text': 'DeepLabV3+\nSzegmentáció', 'color': '#e6f2ff'},
    {'text': 'Bounding box\nkivágás', 'color': '#e6ffe6'},
    {'text': 'ResNet50V2\nKlasszifikáció', 'color': '#e6f2ff'},
    {'text': 'Eredmény\nmegjelenítése\n(UI)', 'color': '#f2e6ff'}
]

x_start = 0.02
y_start = 0.3
box_width = 0.12
box_height = 0.4
spacing = 0.16

for i, step in enumerate(steps):
    x = x_start + i * spacing
    
    # Draw box
    box = mpatches.FancyBboxPatch((x, y_start), box_width, box_height,
                                  boxstyle="round,pad=0.03",
                                  ec="#333333", fc=step['color'], lw=1.5)
    ax.add_patch(box)
    
    # Add text
    ax.text(x + box_width/2, y_start + box_height/2, step['text'],
            ha='center', va='center', fontsize=11, fontweight='bold')

    # Draw arrow
    if i < len(steps) - 1:
        ax.annotate('', xy=(x + box_width + 0.03, y_start + box_height/2),
                    xytext=(x + box_width, y_start + box_height/2),
                    arrowprops=dict(arrowstyle="->,head_width=0.5,head_length=0.8", color="#333333", lw=2.5))

plt.tight_layout()
plt.savefig('tdk/pipeline_diagram.png', dpi=300, bbox_inches='tight')
print('Pipeline diagram saved to tdk/pipeline_diagram.png')
