import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch, Circle
import matplotlib.lines as mlines

fig, ax = plt.subplots(figsize=(10, 7.5))
ax.set_xlim(0, 10)
ax.set_ylim(0, 8)
ax.axis("off")

ax.text(5, 7.6, "VisionFlow -- Use Case Diagram", ha="center", fontsize=14, weight="bold")

# System boundary
from matplotlib.patches import FancyBboxPatch
boundary = FancyBboxPatch((2.0, 0.8), 6.6, 6.3, boxstyle="square,pad=0.0",
                           linewidth=1.4, edgecolor="#444444", facecolor="none", zorder=1)
ax.add_patch(boundary)
ax.text(5.3, 6.85, "VisionFlow CLI System", ha="center", fontsize=10.5, style="italic", color="#444444")

def actor(x, y, label):
    # simple stick figure
    ax.add_patch(Circle((x, y+0.75), 0.18, edgecolor="black", facecolor="#f2d98d", zorder=3))
    ax.plot([x, x], [y+0.57, y], color="black", linewidth=1.6, zorder=3)
    ax.plot([x-0.22, x+0.22], [y+0.42, y+0.42], color="black", linewidth=1.6, zorder=3)
    ax.plot([x, x-0.2], [y, y-0.35], color="black", linewidth=1.6, zorder=3)
    ax.plot([x, x+0.2], [y, y-0.35], color="black", linewidth=1.6, zorder=3)
    ax.text(x, y-0.55, label, ha="center", fontsize=9.5, weight="bold")

def use_case(x, y, w, h, text):
    e = Ellipse((x, y), w, h, edgecolor="#2c4a6e", facecolor="#dce6f2", linewidth=1.3, zorder=2)
    ax.add_patch(e)
    ax.text(x, y, text, ha="center", va="center", fontsize=8.7, zorder=3)
    return (x, y)

def link(p1, p2):
    line = mlines.Line2D([p1[0], p2[0]], [p1[1], p2[1]], color="#333333", linewidth=1.2, zorder=1)
    ax.add_line(line)

actor(0.9, 3.6, "Student /\nOperator")

uc1 = use_case(4.3, 6.2, 2.6, 0.85, "Download pretrained\nYOLOv4-tiny model")
uc2 = use_case(4.3, 5.0, 2.6, 0.85, "Generate synthetic\ndemo video")
uc3 = use_case(4.3, 3.8, 2.6, 0.9, "Run detection +\ntracking + analytics\npipeline on a video")
uc4 = use_case(7.6, 4.6, 2.4, 0.85, "View annotated\noutput video")
uc5 = use_case(7.6, 3.4, 2.4, 0.85, "Inspect CSV / JSON\ndetection report")
uc6 = use_case(7.6, 2.2, 2.4, 0.85, "View motion heatmap\n& edge snapshot")
uc7 = use_case(4.3, 2.4, 2.6, 0.85, "Run automated\nunit test suite")

link((1.15, 3.9), uc1)
link((1.15, 3.75), uc2)
link((1.15, 3.6), uc3)
link((1.15, 3.2), uc7)

link(uc3, uc4)
link(uc3, uc5)
link(uc3, uc6)

# include relations
ax.text(5.6, 4.35, "<<include>>", fontsize=7.5, color="#555555", style="italic")
ax.text(5.85, 3.85, "<<include>>", fontsize=7.5, color="#555555", style="italic")
ax.text(6.0, 2.85, "<<include>>", fontsize=7.5, color="#555555", style="italic")

plt.tight_layout()
plt.savefig("/home/claude/visionflow/docs/diagrams/use_case.png", dpi=165, bbox_inches="tight")
print("saved use_case.png")
