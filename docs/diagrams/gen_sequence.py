import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

fig, ax = plt.subplots(figsize=(11, 7.5))
ax.set_xlim(0, 12)
ax.set_ylim(0, 9)
ax.axis("off")

ax.text(6, 8.7, "VisionFlow -- Sequence Diagram (per frame)", ha="center", fontsize=14, weight="bold")

actors = [
    ("CLI", 0.8),
    ("Pipeline", 2.9),
    ("YOLODetector", 5.1),
    ("CentroidTracker", 7.2),
    ("MotionAnalyzer", 9.1),
    ("ReportGenerator", 10.9),
]

top_y = 8.0
bottom_y = 0.6

for name, x in actors:
    ax.add_patch(Rectangle((x-0.75, top_y), 1.5, 0.5, edgecolor="#2c4a6e",
                            facecolor="#dce6f2", linewidth=1.2, zorder=3))
    ax.text(x, top_y+0.25, name, ha="center", va="center", fontsize=8.8, weight="bold", zorder=4)
    ax.plot([x, x], [top_y, bottom_y], color="#888888", linewidth=1.1, linestyle="--", zorder=1)

def msg(x1, x2, y, text, dashed=False, fontsize=8.2):
    style = "-|>" if not dashed else "-|>"
    ls = "dashed" if dashed else "solid"
    arr = FancyArrowPatch((x1, y), (x2, y), arrowstyle=style, mutation_scale=12,
                           linewidth=1.2, color="#222222", linestyle=ls, zorder=2)
    ax.add_patch(arr)
    mid = (x1 + x2) / 2
    ax.text(mid, y+0.12, text, ha="center", fontsize=fontsize, zorder=4)

cli_x, pipe_x, det_x, trk_x, mot_x, rep_x = [a[1] for a in actors]

y = 7.3
msg(cli_x, pipe_x, y, "run(source)")
y -= 0.55
msg(pipe_x, det_x, y, "detect(frame)")
y -= 0.5
msg(det_x, pipe_x, y, "[Detection, ...]", dashed=True)
y -= 0.55
msg(pipe_x, trk_x, y, "update(detections)")
y -= 0.5
msg(trk_x, pipe_x, y, "tracked_boxes: dict", dashed=True)
y -= 0.55
msg(pipe_x, trk_x, y, "line_counter.update(boxes)")
y -= 0.55
msg(pipe_x, mot_x, y, "process(frame)")
y -= 0.5
msg(mot_x, pipe_x, y, "fg_mask, heatmap", dashed=True)
y -= 0.55
msg(pipe_x, rep_x, y, "log_frame(idx, rows)")
y -= 0.6
ax.text((pipe_x+rep_x)/2, y+0.35, "-- loop over all frames --", fontsize=8, style="italic", color="#555555")
y -= 0.3
msg(pipe_x, rep_x, y, "finalize(extra_stats)")
y -= 0.5
msg(rep_x, pipe_x, y, "summary: dict", dashed=True)
y -= 0.55
msg(pipe_x, cli_x, y, "summary", dashed=True)

plt.tight_layout()
plt.savefig("/home/claude/visionflow/docs/diagrams/sequence.png", dpi=165, bbox_inches="tight")
print("saved sequence.png")
