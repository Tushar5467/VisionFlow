import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Polygon

fig, ax = plt.subplots(figsize=(9, 11))
ax.set_xlim(0, 9.5)
ax.set_ylim(0, 15.5)
ax.axis("off")

def rounded(x, y, w, h, text, color="#3b6ea5", fontsize=10):
    patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.15",
                            linewidth=1.4, edgecolor="#22344a", facecolor=color, zorder=2)
    ax.add_patch(patch)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fontsize,
            color="white", weight="bold", zorder=3)

def diamond(x, y, w, h, text, color="#c9a227", fontsize=9.5):
    pts = [(x, y+h/2), (x+w/2, y+h), (x+w, y+h/2), (x+w/2, y)]
    poly = Polygon(pts, closed=True, linewidth=1.4, edgecolor="#5a4a10", facecolor=color, zorder=2)
    ax.add_patch(poly)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fontsize,
            color="black", weight="bold", zorder=3)

def start_end(x, y, w, h, text, color="#5b8c5a"):
    patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.4",
                            linewidth=1.4, edgecolor="#1f3a1f", facecolor=color, zorder=2)
    ax.add_patch(patch)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=10.5,
            color="white", weight="bold", zorder=3)

def arrow(x1, y1, x2, y2, text=None, tx_off=0.25):
    arr = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=15,
                           linewidth=1.5, color="#333333", zorder=1)
    ax.add_patch(arr)
    if text:
        ax.text(x2 + tx_off, (y1 + y2) / 2, text, fontsize=8.5, color="#333333")

cx = 2.5   # centre x for main column
bx = 6.3   # centre x for end-of-stream branch column
w_box = 3.6

ax.text(3.3, 15.1, "VisionFlow -- Workflow / Process Flow", ha="center", fontsize=14, weight="bold")

start_end(cx-1.3, 14.0, 3.3, 0.8, "Start: python main.py run")
arrow(cx+0.35, 14.0, cx+0.35, 13.6)

rounded(cx-1.5, 12.8, w_box, 0.8, "Open video source\n(file / webcam index)")
arrow(cx+0.3, 12.8, cx+0.3, 12.4)

rounded(cx-1.5, 11.6, w_box, 0.8, "Read next frame")
arrow(cx+0.3, 11.6, cx+0.3, 11.2)

diamond(cx-1.5, 10.3, w_box, 0.9, "Frame read OK?")
ax.text(cx+2.3, 10.9, "no", fontsize=9, color="#333333")
arrow(cx+2.1, 10.75, bx-1.25, 10.35)

rounded(cx-1.5, 9.1, w_box, 0.8, "Resize frame\n(utils.resize_frame)")
ax.text(cx+2.0, 10.05, "yes", fontsize=9, color="#333333")
arrow(cx+0.3, 9.1, cx+0.3, 8.7)

rounded(cx-1.5, 7.9, w_box, 0.8, "YOLOv4-tiny detection\n(detector.detect)")
arrow(cx+0.3, 7.9, cx+0.3, 7.5)

rounded(cx-1.5, 6.7, w_box, 0.8, "Centroid tracking +\nline-crossing update")
arrow(cx+0.3, 6.7, cx+0.3, 6.3)

rounded(cx-1.5, 5.5, w_box, 0.8, "Canny edges + MOG2\nmotion / heatmap update")
arrow(cx+0.3, 5.5, cx+0.3, 5.1)

rounded(cx-1.5, 4.3, w_box, 0.8, "Draw annotations,\nlog frame to reporter")
arrow(cx+0.3, 4.3, cx+0.3, 3.9)

rounded(cx-1.5, 3.1, w_box, 0.8, "Write frame to\noutput video")
arrow(cx+0.3, 3.1, cx-1.2, 2.3)
ax.annotate("", xy=(cx-1.2, 11.6+0.4), xytext=(cx-1.2, 2.3),
            arrowprops=dict(arrowstyle="-|>", color="#333333", lw=1.5,
                             connectionstyle="arc3,rad=-0.0"))
ax.text(cx-2.1, 7.0, "loop", fontsize=9, rotation=90, color="#333333")

# End-of-stream branch (offset well clear of the main loop column)
bx = 6.3
rounded(bx-1.25, 9.9, 2.8, 0.9, "Release resources,\nwrite heatmap frame")
arrow(bx+0.15, 9.9, bx+0.15, 9.4)
rounded(bx-1.4, 8.6, 3.05, 0.9, "Finalize CSV + JSON\n(ReportGenerator)")
arrow(bx+0.15, 8.6, bx+0.15, 8.1)
rounded(bx-1.4, 7.4, 3.05, 0.9, "Print run summary\nto console")
arrow(bx+0.15, 7.4, bx+0.15, 6.9)
start_end(bx-1.2, 6.1, 2.4, 0.8, "End")

plt.tight_layout()
plt.savefig("/home/claude/visionflow/docs/diagrams/workflow.png", dpi=170, bbox_inches="tight")
print("saved workflow.png")
