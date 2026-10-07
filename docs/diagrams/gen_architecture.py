import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(11, 7))
ax.set_xlim(0, 11)
ax.set_ylim(0, 7.5)
ax.axis("off")

def box(x, y, w, h, text, color="#3b6ea5", fontsize=10, textcolor="white"):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.08,rounding_size=0.12",
        linewidth=1.4, edgecolor="#22344a", facecolor=color, zorder=2,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
             fontsize=fontsize, color=textcolor, weight="bold", zorder=3, wrap=True)
    return patch

def arrow(x1, y1, x2, y2, text=""):
    arr = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=16,
                           linewidth=1.6, color="#444444", zorder=1)
    ax.add_patch(arr)
    if text:
        ax.text((x1 + x2) / 2 + 0.15, (y1 + y2) / 2, text, fontsize=8.5, color="#333333", style="italic")

ax.text(5.5, 7.1, "VisionFlow -- System Architecture", ha="center", fontsize=15, weight="bold")

# Input layer
box(0.3, 5.7, 2.2, 1.0, "Video Source\n(file / webcam)", color="#5b8c5a")

# CLI
box(3.0, 5.7, 2.4, 1.0, "CLI (main.py)\nargparse", color="#5b8c5a")

# Core pipeline orchestrator
box(3.0, 4.0, 2.4, 1.0, "Pipeline\nOrchestrator", color="#3b6ea5")

# Detector
box(0.3, 2.2, 2.3, 1.0, "YOLODetector\n(OpenCV DNN,\nYOLOv4-tiny)", color="#3b6ea5")

# Tracker
box(2.9, 2.2, 2.3, 1.0, "CentroidTracker +\nLineCrossingCounter", color="#3b6ea5")

# Analytics
box(5.5, 2.2, 2.3, 1.0, "EdgeDetector +\nMotionAnalyzer", color="#3b6ea5")

# Report generator
box(8.1, 2.2, 2.5, 1.0, "ReportGenerator\n(CSV / JSON)", color="#3b6ea5")

# Config / utils shared
box(6.0, 4.0, 4.6, 1.0, "config.py  /  utils.py\n(shared configuration, logging, helpers)", color="#8a5a9e")

# Outputs
box(0.3, 0.3, 2.3, 1.1, "Annotated Video\n(output/*.mp4)", color="#b5651d")
box(2.9, 0.3, 2.3, 1.1, "detections.csv\nsummary.json", color="#b5651d")
box(5.5, 0.3, 2.3, 1.1, "Motion Heatmap /\nEdge Frame (.jpg)", color="#b5651d")
box(8.1, 0.3, 2.5, 1.1, "Log File\n(output/logs/)", color="#b5651d")

arrow(1.4, 5.7, 1.4, 5.5)  # not used
arrow(2.5, 6.2, 3.0, 6.2)  # source -> CLI
arrow(4.2, 5.7, 4.2, 5.0)  # CLI -> pipeline
arrow(4.0, 4.0, 1.5, 3.2, "frame")
arrow(4.2, 4.0, 4.0, 3.2)
arrow(4.6, 4.0, 6.5, 3.2)
arrow(4.2, 4.0, 9.3, 3.2, "events")
arrow(6.0, 4.5, 5.3, 4.5)

arrow(1.4, 2.2, 1.4, 1.4)
arrow(4.0, 2.2, 4.0, 1.4)
arrow(6.6, 2.2, 6.6, 1.4)
arrow(9.3, 2.2, 9.3, 1.4)

plt.tight_layout()
plt.savefig("/home/claude/visionflow/docs/diagrams/architecture.png", dpi=170, bbox_inches="tight")
print("saved architecture.png")
