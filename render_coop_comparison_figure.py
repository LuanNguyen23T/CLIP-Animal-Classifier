import os
import cv2
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from PIL import Image

# Publication style
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "mathtext.fontset": "dejavusans",
    "font.size": 10,
    "axes.linewidth": 0.8,
})

# Image loading and center square cropping
def load_and_square_crop(path, size=320):
    img = Image.open(path).convert('RGB')
    w, h = img.size
    min_dim = min(w, h)
    left = (w - min_dim) // 2
    top = (h - min_dim) // 2
    img = img.crop((left, top, left + min_dim, top + min_dim))
    img = img.resize((size, size), Image.Resampling.LANCZOS)
    return np.array(img)

# Breeds data
breeds_data = [
    {
        "id": "a",
        "name": "Bengal (Cat)",
        "img_path": "dataset/oxford-iiit-pet/images/Bengal_10.jpg",
        "prompts": [
            {"text": "a [CLASS].", "acc": "35.00", "type": "handcrafted"},
            {"text": "a photo of [CLASS].", "acc": "84.00", "type": "handcrafted"},
            {"text": "a photo of a [CLASS].", "acc": "86.00", "type": "handcrafted"},
            {"text": "a photo of a [CLASS], a type of [SPECIES].", "acc": "86.00", "type": "handcrafted"},
            {"text": r"$[\mathbf{V}]_1 [\mathbf{V}]_2 \dots [\mathbf{V}]_M$ [CLASS]. (CoOp thuần)", "acc": "30.00", "type": "coop"},
            {"text": r"$[\mathbf{V}]_1 [\mathbf{V}]_2 \dots [\mathbf{V}]_M$ [CLASS] [TRAIT]. (Ours)", "acc": "90.00", "type": "ours"}
        ]
    },
    {
        "id": "b",
        "name": "Beagle (Dog)",
        "img_path": "dataset/oxford-iiit-pet/images/beagle_1.jpg",
        "prompts": [
            {"text": "a [CLASS].", "acc": "86.00", "type": "handcrafted"},
            {"text": "a photo of [CLASS].", "acc": "75.00", "type": "handcrafted"},
            {"text": "a photo of a [CLASS].", "acc": "81.00", "type": "handcrafted"},
            {"text": "a photo of a [CLASS], a type of [SPECIES].", "acc": "94.00", "type": "handcrafted"},
            {"text": r"$[\mathbf{V}]_1 [\mathbf{V}]_2 \dots [\mathbf{V}]_M$ [CLASS]. (CoOp thuần)", "acc": "94.00", "type": "coop"},
            {"text": r"$[\mathbf{V}]_1 [\mathbf{V}]_2 \dots [\mathbf{V}]_M$ [CLASS] [TRAIT]. (Ours)", "acc": "97.00", "type": "ours"}
        ]
    },
    {
        "id": "c",
        "name": "Great Pyrenees (Dog)",
        "img_path": "dataset/oxford-iiit-pet/images/great_pyrenees_10.jpg",
        "prompts": [
            {"text": "a [CLASS].", "acc": "94.00", "type": "handcrafted"},
            {"text": "a photo of [CLASS].", "acc": "85.00", "type": "handcrafted"},
            {"text": "a photo of a [CLASS].", "acc": "87.00", "type": "handcrafted"},
            {"text": "a photo of a [CLASS], a type of [SPECIES].", "acc": "88.00", "type": "handcrafted"},
            {"text": r"$[\mathbf{V}]_1 [\mathbf{V}]_2 \dots [\mathbf{V}]_M$ [CLASS]. (CoOp thuần)", "acc": "87.00", "type": "coop"},
            {"text": r"$[\mathbf{V}]_1 [\mathbf{V}]_2 \dots [\mathbf{V}]_M$ [CLASS] [TRAIT]. (Ours)", "acc": "95.00", "type": "ours"}
        ]
    }
]

# Set up figure
fig, ax = plt.subplots(figsize=(16.8, 6.2), dpi=300)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')

# Background
fig.patch.set_facecolor('#FFFFFF')

# Panel metrics
panel_w = 0.315
gap = 0.02
left_margin = 0.015
bottom_y = 0.16
panel_h = 0.76

for idx, b_info in enumerate(breeds_data):
    px = left_margin + idx * (panel_w + gap)
    py = bottom_y
    
    # Outer card box (sleek border)
    card_bg = FancyBboxPatch(
        (px, py), panel_w, panel_h,
        boxstyle="round,pad=0,rounding_size=0.012",
        facecolor='#F8FAFC', edgecolor='#E2E8F0', linewidth=1.2, zorder=1
    )
    ax.add_patch(card_bg)
    
    # Title at top of card
    title_y = py + panel_h - 0.045
    ax.text(
        px + panel_w / 2.0, title_y, b_info["name"],
        ha='center', va='center', fontsize=12.5, fontweight='bold', color='#0F172A', zorder=5
    )
    
    # Layout inside card: Left = Image, Right = Prompts table
    # Image coordinates
    img_size = 0.088
    img_x = px + 0.014
    img_y = py + (panel_h - 0.08) / 2.0 - 0.01
    
    # Render image
    im_arr = load_and_square_crop(b_info["img_path"], size=320)
    img_ax = fig.add_axes([img_x, img_y, img_size, img_size])
    img_ax.imshow(im_arr)
    img_ax.axis('off')
    for spine in img_ax.spines.values():
        spine.set_edgecolor('#94A3B8')
        spine.set_linewidth(1.0)
    
    # Right side: Table header
    tbl_x = img_x + img_size + 0.016
    tbl_w = (px + panel_w - 0.012) - tbl_x
    
    # Table headers
    hdr_y = py + panel_h - 0.09
    ax.text(tbl_x + tbl_w * 0.38, hdr_y, "Prompt", ha='center', va='center', fontsize=10.5, fontweight='bold', color='#334155', zorder=5)
    ax.text(tbl_x + tbl_w - 0.022, hdr_y, "Accuracy", ha='center', va='center', fontsize=10.5, fontweight='bold', color='#334155', zorder=5)
    
    # Draw row items
    prompts = b_info["prompts"]
    num_rows = len(prompts)
    row_h = 0.085
    start_row_y = hdr_y - 0.035
    
    for r_i, p in enumerate(prompts):
        ry = start_row_y - r_i * (row_h + 0.015) - row_h
        
        # Determine styling
        if p["type"] == "handcrafted":
            bg_col = '#E0F2FE'       # Light sky blue
            border_col = '#BAE6FD'
            txt_col = '#0369A1'
            acc_bg = '#E0F2FE'
            acc_border = '#BAE6FD'
            acc_col = '#0C4A6E'
            bold = False
        elif p["type"] == "coop":
            bg_col = '#F1F5F9'       # Light gray/slate
            border_col = '#CBD5E1'
            txt_col = '#475569'
            acc_bg = '#F1F5F9'
            acc_border = '#CBD5E1'
            acc_col = '#334155'
            bold = False
        else: # ours
            bg_col = '#DCFCE7'       # Emerald green
            border_col = '#86EFAC'
            txt_col = '#15803D'
            acc_bg = '#BBF7D0'
            acc_border = '#4ADE80'
            acc_col = '#14532D'
            bold = True
        
        # Prompt box
        p_box_w = tbl_w - 0.052
        p_patch = FancyBboxPatch(
            (tbl_x, ry), p_box_w, row_h,
            boxstyle="round,pad=0,rounding_size=0.006",
            facecolor=bg_col, edgecolor=border_col, linewidth=0.9, zorder=2
        )
        ax.add_patch(p_patch)
        
        # Prompt text
        fw = 'bold' if bold else 'normal'
        ax.text(
            tbl_x + 0.007, ry + row_h / 2.0, p["text"],
            ha='left', va='center', fontsize=8.4, fontweight=fw, color=txt_col, zorder=4
        )
        
        # Accuracy box
        acc_box_x = tbl_x + p_box_w + 0.006
        acc_box_w = tbl_w - p_box_w - 0.006
        acc_patch = FancyBboxPatch(
            (acc_box_x, ry), acc_box_w, row_h,
            boxstyle="round,pad=0,rounding_size=0.006",
            facecolor=acc_bg, edgecolor=acc_border, linewidth=0.9, zorder=2
        )
        ax.add_patch(acc_patch)
        
        # Accuracy text
        ax.text(
            acc_box_x + acc_box_w / 2.0, ry + row_h / 2.0, p["acc"],
            ha='center', va='center', fontsize=9.2, fontweight='bold', color=acc_col, zorder=4
        )
        
    # Subcaption (a), (b), (c)
    subcap_y = py - 0.035
    ax.text(
        px + panel_w / 2.0, subcap_y, f"({b_info['id']})",
        ha='center', va='center', fontsize=12.0, fontweight='bold', color='#1E293B', zorder=5
    )

# Overall publication caption below
caption_y = 0.045
cap_title = r"$\mathbf{Fig.\ 1\ Prompt\ Engineering\ vs.\ Context\ Optimization\ (CoOp)\ vs.\ Ours\ (Gemma-CoOp).}$"
cap_body = (
    " Hand-crafted prompts require manual phrasing search with erratic sensitivity. Standard CoOp learns continuous\n"
    "context vectors directly from data but suffers from semantic drift on fine-grained visual categories (e.g., Bengal collapses to 30.00%).\n"
    "In contrast, our proposed method (Gemma-CoOp 6-prompt morphological trait conditioning) anchors soft prompt learning with fine-grained anatomical priors,\n"
    "achieving peak per-class accuracy across fine-grained animal species on the Oxford-IIIT Pet dataset."
)
ax.text(
    0.5, caption_y, cap_title + cap_body,
    ha='center', va='center', fontsize=9.5, color='#1E293B', linespacing=1.45, zorder=5
)

# Output paths
out_png = "enhanced/figures/fig1_prompt_engineering_vs_coop_vs_ours.png"
out_pdf = "enhanced/figures/fig1_prompt_engineering_vs_coop_vs_ours.pdf"
artifact_dir = "/Users/nguyenhuuhoangluan/.gemini/antigravity-ide/brain/3fd5905b-77c3-45f5-8d83-7d6a7341b347"
out_artifact = os.path.join(artifact_dir, "fig1_prompt_engineering_vs_coop_vs_ours.png")

plt.savefig(out_png, format='png', dpi=300, bbox_inches='tight', pad_inches=0.08)
plt.savefig(out_pdf, format='pdf', bbox_inches='tight', pad_inches=0.08)
plt.savefig(out_artifact, format='png', dpi=300, bbox_inches='tight', pad_inches=0.08)
print(f"Successfully generated:\n- {out_png}\n- {out_pdf}\n- {out_artifact}")
