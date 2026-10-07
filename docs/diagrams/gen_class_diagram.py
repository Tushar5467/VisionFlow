import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(13, 8.5))
ax.set_xlim(0, 13)
ax.set_ylim(0, 9.5)
ax.axis("off")

def uml_class(x, y, w, h, title, attrs, methods, color="#e9eef5", title_color="#2c4a6e"):
    box = FancyBboxPatch((x, y), w, h, boxstyle="square,pad=0.0",
                          linewidth=1.3, edgecolor="#2c4a6e", facecolor=color, zorder=2)
    ax.add_patch(box)
    title_h = 0.5
    ax.plot([x, x+w], [y+h-title_h, y+h-title_h], color="#2c4a6e", linewidth=1.2, zorder=3)
    ax.text(x+w/2, y+h-title_h/2, title, ha="center", va="center", fontsize=10.5,
            weight="bold", color=title_color, zorder=4)

    n_attr = len(attrs)
    attr_block_h = 0.28 * max(n_attr, 1)
    ax.plot([x, x+w], [y+h-title_h-attr_block_h, y+h-title_h-attr_block_h],
            color="#2c4a6e", linewidth=1.0, zorder=3)
    for i, a in enumerate(attrs):
        ax.text(x+0.15, y+h-title_h-0.18-0.28*i, a, ha="left", va="center", fontsize=8.3, zorder=4)

    for i, m in enumerate(methods):
        ax.text(x+0.15, y+h-title_h-attr_block_h-0.22-0.28*i, m, ha="left", va="center",
                fontsize=8.3, zorder=4, style="italic")

def arrow(x1, y1, x2, y2, style="-|>", label=""):
    arr = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=14,
                           linewidth=1.3, color="#333333", zorder=1,
                           connectionstyle="arc3,rad=0.0")
    ax.add_patch(arr)
    if label:
        ax.text((x1+x2)/2, (y1+y2)/2 + 0.15, label, fontsize=7.8, color="#333333")

ax.text(6.5, 9.2, "VisionFlow -- Simplified Class Diagram", ha="center", fontsize=14, weight="bold")

uml_class(0.3, 5.6, 3.0, 3.2, "VisionFlowPipeline",
          ["- detector: YOLODetector", "- tracker: CentroidTracker",
           "- edge_detector: EdgeDetector", "- motion_analyzer: MotionAnalyzer",
           "- reporter: ReportGenerator", "- line_counter: LineCrossingCounter"],
          ["+ run(): dict", "- _init_video_io()"])

uml_class(4.2, 7.2, 2.7, 1.6, "YOLODetector",
          ["- net: cv2.dnn.Net", "- class_names: list",
           "- confidence_threshold: float"],
          ["+ detect(frame): list[Detection]"])

uml_class(4.2, 5.3, 2.7, 1.4, "Detection",
          ["+ box: tuple", "+ class_id: int",
           "+ class_name: str", "+ confidence: float"],
          [])

uml_class(7.3, 7.2, 2.9, 1.9, "CentroidTracker",
          ["- objects: dict", "- boxes: dict",
           "- disappeared: dict", "- max_distance: int"],
          ["+ update(detections): dict", "- _register()", "- _deregister()"])

uml_class(7.3, 5.3, 2.9, 1.4, "LineCrossingCounter",
          ["- line_y: int", "- count_up: int", "- count_down: int"],
          ["+ update(tracked_boxes)"])

uml_class(10.5, 5.9, 2.2, 1.8, "EdgeDetector",
          ["- low: int", "- high: int"],
          ["+ process(frame): ndarray"])

uml_class(10.5, 3.6, 2.2, 2.0, "MotionAnalyzer",
          ["- bg_subtractor", "- heatmap: ndarray"],
          ["+ process(frame): ndarray", "+ get_heatmap_overlay()"])

uml_class(4.2, 2.9, 3.4, 2.1, "ReportGenerator",
          ["- csv_path: str", "- json_path: str",
           "- _class_counts: Counter"],
          ["+ log_frame(idx, rows)", "+ finalize(extra): dict"])

uml_class(0.3, 2.7, 3.4, 2.2, "config / utils (module)",
          ["config.* (paths, thresholds)",
           "utils.get_logger()", "utils.draw_box()",
           "utils.centroid_of_box()"],
          [])

# Relationships
arrow(3.3, 7.6, 4.2, 7.9, label="uses")
arrow(3.3, 7.0, 4.2, 6.0, label="uses")
arrow(3.3, 6.6, 7.3, 7.8, label="uses")
arrow(3.3, 6.3, 7.3, 5.9, label="uses")
arrow(3.3, 6.0, 10.5, 6.8, label="uses")
arrow(3.3, 5.8, 10.5, 4.6, label="uses")
arrow(3.3, 6.9, 5.9, 5.0, label="writes via")
arrow(5.6, 7.2, 5.6, 6.7, label="creates")
arrow(8.6, 7.2, 8.6, 6.7, label="feeds")
arrow(1.9, 5.6, 1.9, 4.9, label="reads")

plt.tight_layout()
plt.savefig("/home/claude/visionflow/docs/diagrams/class_diagram.png", dpi=165, bbox_inches="tight")
print("saved class_diagram.png")
