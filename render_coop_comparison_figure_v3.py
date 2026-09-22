import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from PIL import Image

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "mathtext.fontset": "dejavusans",
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

breeds_data = [
    {
        "id": "a",
        "title": "Bengal",
        "img_path": "dataset/oxford-iiit-pet/images/Bengal_10.jpg",
        "prompts": [
            {"text": "a [CLASS].", "acc": "35.00", "type": "blue"},
            {"text": "a photo of [CLASS].", "acc": "84.00", "type": "blue"},
            {"text": "a photo of a [CLASS].", "acc": "86.00", "type": "blue"},
            {"text": "a photo of a [CLASS], a type of cat.", "acc": "86.00", "type": "blue"},
            {"text": r"$[\mathbf{V}]_1 [\mathbf{V}]_2 \dots [\mathbf{V}]_M$ [CLASS].", "acc": "30.00", "type": "gray"},
            {"text": r"$[\mathbf{V}]_1 [\mathbf{V}]_2 \dots [\mathbf{V}]_M$ [CLASS] [$\hat{\mathbf{a}}_c$].", "acc": "90.00", "type": "green"}
        ]
    },
    {
        "id": "b",
        "title": "Beagle",
        "img_path": "dataset/oxford-iiit-pet/images/beagle_1.jpg",
        "prompts": [
            {"text": "a [CLASS].", "acc": "86.00", "type": "blue"},
            {"text": "a photo of [CLASS].", "acc": "75.00", "type": "blue"},
            {"text": "a photo of a [CLASS].", "acc": "81.00", "type": "blue"},
            {"text": "a photo of a [CLASS], a type of dog.", "acc": "94.00", "type": "blue"},
            {"text": r"$[\mathbf{V}]_1 [\mathbf{V}]_2 \dots [\mathbf{V}]_M$ [CLASS].", "acc": "94.00", "type": "gray"},
            {"text": r"$[\mathbf{V}]_1 [\mathbf{V}]_2 \dots [\mathbf{V}]_M$ [CLASS] [$\hat{\mathbf{a}}_c$].", "acc": "97.00", "type": "green"}
        ]
    },
    {
        "id": "c",
        "title": "Great Pyrenees",
        "img_path": "dataset/oxford-iiit-pet/images/great_pyrenees_10.jpg",
        "prompts": [
            {"text": "a [CLASS].", "acc": "94.00", "type": "blue"},
            {"text": "a photo of [CLASS].", "acc": "85.00", "type": "blue"},
            {"text": "a photo of a [CLASS].", "acc": "87.00", "type": "blue"},
            {"text": "a photo of a [CLASS], a type of dog.", "acc": "88.00", "type": "blue"},
            {"text": r"$[\mathbf{V}]_1 [\mathbf{V}]_2 \dots [\mathbf{V}]_M$ [CLASS].", "acc": "87.00", "type": "gray"},
            {"text": r"$[\mathbf{V}]_1 [\mathbf{V}]_2 \dots [\mathbf{V}]_M$ [CLASS] [$\hat{\mathbf{a}}_c$].", "acc": "95.00", "type": "green"}
        ]
    }
]

# Set up figure with 3 subplots side by side
fig = plt.figure(figsize=(20.0, 5.8), dpi=300)
fig.patch.set_facecolor('#FFFFFF')

# Grid specs: 3 columns
gs = fig.add_gridspec(1, 3, left=0.03, right=0.97, bottom=0.22, top=0.92, wspace=0.18)

for idx, item in enumerate(breeds_data):
    ax = fig.add_subplot(gs[0, idx])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')
    
    # Title
    ax.text(50, 96, item["title"], ha='center', va='center', fontsize=13.5, fontweight='bold', color='#0F172A')
    
    # Representative Image on left
    # x: 0 to 32, y: 12 to 84 (aspect: 32x72)
    # Note: subplot aspect is roughly (20*0.94/3) / (5.8*0.70) = 6.26 / 4.06 = 1.54
    # To make image square: dy = 66, dx = dy / 1.54 = 42.8
    # Let's place image at x: 0 to 33, y: 14 to 82
    im_arr = load_and_square_crop(item["img_path"], size=400)
    img_x1, img_x2 = 0, 31
    img_y1, img_y2 = 12, 82
    ax.imshow(im_arr, extent=[img_x1, img_x2, img_y1, img_y2], zorder=3)
    
    # Border around image
    border = plt.Rectangle((img_x1, img_y1), img_x2 - img_x1, img_y2 - img_y1,
                           facecolor='none', edgecolor='#475569', linewidth=1.1, zorder=4)
    ax.add_patch(border)
    
    # Table headers
    tbl_x = 35
    p_w = 48
    acc_w = 15
    acc_x = tbl_x + p_w + 2
    
    ax.text(tbl_x + p_w / 2.0, 89, "Prompt", ha='center', va='center', fontsize=11, fontweight='bold', color='#1E293B')
    ax.text(acc_x + acc_w / 2.0, 89, "Accuracy", ha='center', va='center', fontsize=11, fontweight='bold', color='#1E293B')
    
    # 6 Rows
    row_h = 9.8
    row_gap = 2.4
    table_top = 84
    
    for r_i, p in enumerate(item["prompts"]):
        ry = table_top - (r_i + 1) * row_h - r_i * row_gap
        
        if p["type"] == "blue":
            p_bg = '#E0F2FE'       # Clean ice blue
            p_border = '#BAE6FD'
            p_col = '#0369A1'
            acc_bg = '#E0F2FE'
            acc_border = '#BAE6FD'
            acc_col = '#0C4A6E'
            fw = 'normal'
            acc_fw = 'bold'
        elif p["type"] == "gray":
            p_bg = '#F1F5F9'       # Neutral slate
            p_border = '#CBD5E1'
            p_col = '#475569'
            acc_bg = '#F1F5F9'
            acc_border = '#CBD5E1'
            acc_col = '#334155'
            fw = 'normal'
            acc_fw = 'bold'
        else: # green (Ours)
            p_bg = '#DCFCE7'       # Emerald green highlight
            p_border = '#86EFAC'
            p_col = '#15803D'
            acc_bg = '#BBF7D0'
            acc_border = '#4ADE80'
            acc_col = '#14532D'
            fw = 'bold'
            acc_fw = 'bold'
            
        # Prompt box
        p_patch = FancyBboxPatch((tbl_x, ry), p_w, row_h,
                                boxstyle="round,pad=0,rounding_size=1.2",
                                facecolor=p_bg, edgecolor=p_border, linewidth=0.9, zorder=2)
        ax.add_patch(p_patch)
        
        # Prompt text
        ax.text(tbl_x + 1.6, ry + row_h / 2.0, p["text"],
                ha='left', va='center', fontsize=8.6, fontweight=fw, color=p_col, zorder=5)
        
        # Accuracy box
        acc_patch = FancyBboxPatch((acc_x, ry), acc_w, row_h,
                                  boxstyle="round,pad=0,rounding_size=1.2",
                                  facecolor=acc_bg, edgecolor=acc_border, linewidth=0.9, zorder=2)
        ax.add_patch(acc_patch)
        
        # Accuracy text
        ax.text(acc_x + acc_w / 2.0, ry + row_h / 2.0, p["acc"],
                ha='center', va='center', fontsize=9.2, fontweight=acc_fw, color=acc_col, zorder=5)
        
    # Subcaption (a), (b), (c)
    ax.text(50, 0, f"({item['id']})", ha='center', va='center', fontsize=12, fontweight='bold', color='#0F172A')

# Bottom Caption on Figure level
caption_text = (
    r"$\mathbf{Fig.\ 1\ \ Prompt\ engineering\ vs\ Context\ Optimization\ (CoOp)\ vs\ Ours\ (Gemma-CoOp).}$ "
    "The former requires empirical words tuning with erratic sensitivity;\n"
    r"CoOp automates the process with continuous learnable context $[\mathbf{V}]_1 \dots [\mathbf{V}]_M$ but suffers severe semantic drift on fine-grained classes (e.g., Bengal drops to 30.00%);"
    "\n"
    r"our method conditions learnable soft prompt vectors with compact morphological trait anchors $[\hat{\mathbf{a}}_c]$ from Google Gemma 4 SLM, achieving top accuracy across all species."
)
fig.text(0.5, 0.08, caption_text, ha='center', va='center', fontsize=9.6, color='#1E293B', linespacing=1.35)

# Output paths
out_png = "enhanced/figures/fig1_prompt_engineering_vs_coop_vs_ours.png"
out_pdf = "enhanced/figures/fig1_prompt_engineering_vs_coop_vs_ours.pdf"
artifact_dir = "/Users/nguyenhuuhoangluan/.gemini/antigravity-ide/brain/3fd5905b-77c3-45f5-8d83-7d6a7341b347"
out_artifact = os.path.join(artifact_dir, "fig1_prompt_engineering_vs_coop_vs_ours.png")

plt.savefig(out_png, format='png', dpi=300, bbox_inches='tight', pad_inches=0.08)
plt.savefig(out_pdf, format='pdf', bbox_inches='tight', pad_inches=0.08)
plt.savefig(out_artifact, format='png', dpi=300, bbox_inches='tight', pad_inches=0.08)
print("V3 generated successfully!")
