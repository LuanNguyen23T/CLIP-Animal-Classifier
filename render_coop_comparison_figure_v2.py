import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from PIL import Image

# Publication style
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "mathtext.fontset": "dejavusans",
    "font.size": 10,
    "axes.linewidth": 0.8,
})

def load_and_square_crop(path, size=400):
    img = Image.open(path).convert('RGB')
    w, h = img.size
    min_dim = min(w, h)
    left = (w - min_dim) // 2
    top = (h - min_dim) // 2
    img = img.crop((left, top, left + min_dim, top + min_dim))
    img = img.resize((size, size), Image.Resampling.LANCZOS)
    return np.array(img)

# Breeds data with exact computed accuracies
breeds_data = [
    {
        "id": "a",
        "title": "Bengal (Cat)",
        "img_path": "dataset/oxford-iiit-pet/images/Bengal_10.jpg",
        "prompts": [
            {"text": "a [CLASS].", "acc": "35.00", "style": "blue"},
            {"text": "a photo of [CLASS].", "acc": "84.00", "style": "blue"},
            {"text": "a photo of a [CLASS].", "acc": "86.00", "style": "blue"},
            {"text": "a photo of a [CLASS], a type of cat.", "acc": "86.00", "style": "blue"},
            {"text": r"$[\mathbf{V}]_1 [\mathbf{V}]_2 \dots [\mathbf{V}]_M$ [CLASS]. (CoOp thuần)", "acc": "30.00", "style": "gray"},
            {"text": r"$[\mathbf{V}]_1 [\mathbf{V}]_2 \dots [\mathbf{V}]_M$ [CLASS] [$\hat{a}_c$]. (Ours)", "acc": "90.00", "style": "green"}
        ]
    },
    {
        "id": "b",
        "title": "Beagle (Dog)",
        "img_path": "dataset/oxford-iiit-pet/images/beagle_1.jpg",
        "prompts": [
            {"text": "a [CLASS].", "acc": "86.00", "style": "blue"},
            {"text": "a photo of [CLASS].", "acc": "75.00", "style": "blue"},
            {"text": "a photo of a [CLASS].", "acc": "81.00", "style": "blue"},
            {"text": "a photo of a [CLASS], a type of dog.", "acc": "94.00", "style": "blue"},
            {"text": r"$[\mathbf{V}]_1 [\mathbf{V}]_2 \dots [\mathbf{V}]_M$ [CLASS]. (CoOp thuần)", "acc": "94.00", "style": "gray"},
            {"text": r"$[\mathbf{V}]_1 [\mathbf{V}]_2 \dots [\mathbf{V}]_M$ [CLASS] [$\hat{a}_c$]. (Ours)", "acc": "97.00", "style": "green"}
        ]
    },
    {
        "id": "c",
        "title": "Great Pyrenees (Dog)",
        "img_path": "dataset/oxford-iiit-pet/images/great_pyrenees_10.jpg",
        "prompts": [
            {"text": "a [CLASS].", "acc": "94.00", "style": "blue"},
            {"text": "a photo of [CLASS].", "acc": "85.00", "style": "blue"},
            {"text": "a photo of a [CLASS].", "acc": "87.00", "style": "blue"},
            {"text": "a photo of a [CLASS], a type of dog.", "acc": "88.00", "style": "blue"},
            {"text": r"$[\mathbf{V}]_1 [\mathbf{V}]_2 \dots [\mathbf{V}]_M$ [CLASS]. (CoOp thuần)", "acc": "87.00", "style": "gray"},
            {"text": r"$[\mathbf{V}]_1 [\mathbf{V}]_2 \dots [\mathbf{V}]_M$ [CLASS] [$\hat{a}_c$]. (Ours)", "acc": "95.00", "style": "green"}
        ]
    }
]

# Create Figure
fig, ax = plt.subplots(figsize=(18.0, 6.6), dpi=300)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')
fig.patch.set_facecolor('#FFFFFF')

# Metric layout parameters
col_w = 0.312
col_gap = 0.022
margin_left = 0.012

# Vertical coordinates
top_title_y = 0.94
header_y = 0.89
table_top_y = 0.855
row_h = 0.076
row_gap = 0.014
table_bottom_y = table_top_y - 6 * row_h - 5 * row_gap
subcap_y = 0.22
caption_y = 0.10

for idx, item in enumerate(breeds_data):
    cx = margin_left + idx * (col_w + col_gap)
    
    # 1. Category title on top
    ax.text(
        cx + col_w / 2.0, top_title_y, item["title"],
        ha='center', va='center', fontsize=13, fontweight='bold', color='#0F172A'
    )
    
    # 2. Left Image (square thumbnail)
    # Total table height = 6 * row_h + 5 * row_gap
    total_tbl_h = 6 * row_h + 5 * row_gap
    img_w = 0.082
    img_h = total_tbl_h * 0.72 # balanced square aspect
    # To keep 1:1 aspect in figure units: fig aspect is 18.0 / 6.6 = 2.727
    # So dx = dy / 2.727
    img_dy = total_tbl_h * 0.76
    img_dx = img_dy / (18.0 / 6.6)
    
    img_x1 = cx
    img_x2 = img_x1 + img_dx
    img_y1 = table_bottom_y + (total_tbl_h - img_dy) / 2.0
    img_y2 = img_y1 + img_dy
    
    im_arr = load_and_square_crop(item["img_path"], size=400)
    ax.imshow(im_arr, extent=[img_x1, img_x2, img_y1, img_y2], zorder=3)
    
    # Image border
    rect = plt.Rectangle(
        (img_x1, img_y1), img_dx, img_dy,
        facecolor='none', edgecolor='#64748B', linewidth=1.1, zorder=4
    )
    ax.add_patch(rect)
    
    # 3. Table Column Headers
    tbl_start_x = img_x2 + 0.012
    tbl_total_w = (cx + col_w) - tbl_start_x
    acc_box_w = 0.046
    p_box_w = tbl_total_w - acc_box_w - 0.007
    
    ax.text(tbl_start_x + p_box_w / 2.0, header_y, "Prompt", ha='center', va='center', fontsize=11, fontweight='bold', color='#1E293B')
    ax.text(tbl_start_x + p_box_w + 0.007 + acc_box_w / 2.0, header_y, "Accuracy", ha='center', va='center', fontsize=11, fontweight='bold', color='#1E293B')
    
    # 4. Table Rows
    for r_i, p in enumerate(item["prompts"]):
        ry = table_top_y - (r_i + 1) * row_h - r_i * row_gap
        
        if p["style"] == "blue":
            p_bg = '#E0F2FE'       # Crisp Ice Blue (radford style)
            p_edge = '#BAE6FD'
            p_col = '#0369A1'
            acc_bg = '#E0F2FE'
            acc_edge = '#BAE6FD'
            acc_col = '#0369A1'
            fw = 'normal'
            acc_fw = 'bold'
        elif p["style"] == "gray":
            p_bg = '#F1F5F9'       # Neutral slate
            p_edge = '#CBD5E1'
            p_col = '#475569'
            acc_bg = '#F1F5F9'
            acc_edge = '#CBD5E1'
            acc_col = '#334155'
            fw = 'normal'
            acc_fw = 'bold'
        else: # green (Ours)
            p_bg = '#DCFCE7'       # Vibrant Emerald Green (CoOp highlight style)
            p_edge = '#86EFAC'
            p_col = '#15803D'
            acc_bg = '#BBF7D0'
            acc_edge = '#4ADE80'
            acc_col = '#14532D'
            fw = 'bold'
            acc_fw = 'bold'
            
        # Prompt box
        p_patch = FancyBboxPatch(
            (tbl_start_x, ry), p_box_w, row_h,
            boxstyle="round,pad=0,rounding_size=0.004",
            facecolor=p_bg, edgecolor=p_edge, linewidth=0.8, zorder=2
        )
        ax.add_patch(p_patch)
        
        # Prompt text
        ax.text(
            tbl_start_x + 0.005, ry + row_h / 2.0, p["text"],
            ha='left', va='center', fontsize=9.2, fontweight=fw, color=p_col, zorder=5
        )
        
        # Accuracy box
        acc_x = tbl_start_x + p_box_w + 0.007
        acc_patch = FancyBboxPatch(
            (acc_x, ry), acc_box_w, row_h,
            boxstyle="round,pad=0,rounding_size=0.004",
            facecolor=acc_bg, edgecolor=acc_edge, linewidth=0.8, zorder=2
        )
        ax.add_patch(acc_patch)
        
        # Accuracy number
        ax.text(
            acc_x + acc_box_w / 2.0, ry + row_h / 2.0, p["acc"],
            ha='center', va='center', fontsize=10.0, fontweight=acc_fw, color=acc_col, zorder=5
        )
        
    # 5. Subcaption (a), (b), (c)
    ax.text(
        cx + col_w / 2.0, subcap_y, f"({item['id']})",
        ha='center', va='center', fontsize=12, fontweight='bold', color='#0F172A'
    )

# 6. Overall Publication Caption at Bottom
cap_header = r"$\mathbf{Fig.\ 1\ Prompt\ engineering\ vs.\ Context\ Optimization\ (CoOp)\ vs.\ Ours\ (Gemma-CoOp).}$"
cap_text = (
    " Hand-crafted prompts (blue) require empirical manual words tuning with unpredictable sensitivity across breeds.\n"
    r"Standard CoOp (gray) learns continuous context vectors $[\mathbf{V}]_1 \dots [\mathbf{V}]_M$ directly from training data but suffers from semantic drift on fine-grained visual categories"
    "\n(e.g., Bengal collapses to 30.00% due to loss of feline semantic grounding). "
    r"In contrast, our proposed method (Gemma-CoOp, green) pairs continuous soft prompt tokens with"
    "\n"
    r"compact morphological trait anchors $[\hat{a}_c]$ from Google Gemma 4 SLM, reliably stabilizing backpropagation and achieving record per-class accuracy across species."
)

ax.text(
    0.5, caption_y, cap_header + cap_text,
    ha='center', va='center', fontsize=9.6, color='#1E293B', linespacing=1.35
)

# Output paths
out_png = "enhanced/figures/fig1_prompt_engineering_vs_coop_vs_ours.png"
out_pdf = "enhanced/figures/fig1_prompt_engineering_vs_coop_vs_ours.pdf"
artifact_dir = "/Users/nguyenhuuhoangluan/.gemini/antigravity-ide/brain/3fd5905b-77c3-45f5-8d83-7d6a7341b347"
out_artifact = os.path.join(artifact_dir, "fig1_prompt_engineering_vs_coop_vs_ours.png")

plt.savefig(out_png, format='png', dpi=300, bbox_inches='tight', pad_inches=0.08)
plt.savefig(out_pdf, format='pdf', bbox_inches='tight', pad_inches=0.08)
plt.savefig(out_artifact, format='png', dpi=300, bbox_inches='tight', pad_inches=0.08)
print("Updated figure rendered successfully!")
