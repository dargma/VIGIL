"""Generate Phase 4 GDPO comparison bar chart."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'font.size': 12})

# Data
methods = ['Baseline', 'GDPO\n(no-LSR)', 'GDPO\n(with-LSR)', 'Phase 2\nGRPO-LSR']
pope_acc = [91.7, 93.3, 91.7, 95.0]
blind_gap = [40.0, 42.0, 40.0, 44.0]

x = np.arange(len(methods))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))

bars1 = ax.bar(x - width/2, pope_acc, width, label='POPE Accuracy (%)',
               color='#4C72B0', edgecolor='white', linewidth=0.8)
bars2 = ax.bar(x + width/2, blind_gap, width, label='Blind Gap (pp)',
               color='#DD8452', edgecolor='white', linewidth=0.8)

# Value labels
for bar in bars1:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., h + 0.3,
            f'{h:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

for bar in bars2:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., h + 0.3,
            f'{h:.1f}pp', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Highlight best
bars1[3].set_edgecolor('#2d4a7a')
bars1[3].set_linewidth(2.5)
bars2[3].set_edgecolor('#a05520')
bars2[3].set_linewidth(2.5)

ax.set_ylabel('Score', fontsize=13)
ax.set_title('Phase 4 GDPO vs Phase 2 GRPO-LSR Comparison\n(Qwen3-VL-2B-Thinking, 50-step training)',
             fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(methods, fontsize=12)
ax.set_ylim(0, 105)
ax.legend(loc='upper left', fontsize=12)

# Annotations
ax.annotate('Best: +3.3pp', xy=(3 - width/2, 95.0), xytext=(3 - width/2 - 0.6, 100),
            arrowprops=dict(arrowstyle='->', color='#2d4a7a', lw=1.5),
            fontsize=10, color='#2d4a7a', fontweight='bold')

ax.annotate('No improvement\n(LSR dilutes signal)',
            xy=(2, 92.5), xytext=(1.0, 98),
            arrowprops=dict(arrowstyle='->', color='#cc3333', lw=1.5),
            fontsize=9, color='#cc3333', fontstyle='italic')

plt.tight_layout()
plt.savefig('/content/drive/MyDrive/VIGIL/lab/reports/phase4_gdpo/comparison/phase4_comparison.png',
            dpi=150, bbox_inches='tight')
print("Plot saved to phase4_comparison.png")
