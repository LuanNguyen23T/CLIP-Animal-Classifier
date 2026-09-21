import os
import cv2
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle

# ==============================================================================
# NATURE PORTFOLIO / CVPR PUBLICATION SETTINGS
# ==============================================================================
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "mathtext.fontset": "dejavusans",
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 8,
    "axes.linewidth": 0.8,
})

def generate_5_views(img_path):
    """Generate 5-view TTA specimen representations for visualization."""
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        v1 = np.ones((120, 120, 3), dtype=np.uint8) * 220
        return [v1, v1, v1, v1, v1]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    h, w, _ = img_rgb.shape
    min_dim = min(h, w)
    square = img_rgb[(h-min_dim)//2:(h-min_dim)//2+min_dim, (w-min_dim)//2:(w-min_dim)//2+min_dim]
    v1 = cv2.resize(square, (120, 120), interpolation=cv2.INTER_CUBIC)
    v2 = cv2.flip(v1, 1)
    v3 = cv2.bilateralFilter(v1, d=9, sigmaColor=75, sigmaSpace=75)
    
    lab = cv2.cvtColor(v1, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    v4 = cv2.cvtColor(cv2.merge([clahe.apply(l), a, b]), cv2.COLOR_LAB2RGB)
    
    crop_size = int(120 / 1.18)
    margin = (120 - crop_size) // 2
    v5 = cv2.resize(v1[margin:margin+crop_size, margin:margin+crop_size], (120, 120), interpolation=cv2.INTER_CUBIC)
    return [v1, v2, v3, v4, v5]

def draw_round_box(ax, x, y, w, h, bg='#FFFFFF', border='#1E293B', lw=0.9, rad=0.007, ls='-', zorder=3):
    """Utility to draw sleek, rounded publication cards."""
    patch = FancyBboxPatch((x, y), w, h,
                           boxstyle=f"round,pad=0,rounding_size={rad}",
                           facecolor=bg, edgecolor=border, linewidth=lw, linestyle=ls, zorder=zorder)
    ax.add_patch(patch)

def draw_op(ax, cx, cy, r=0.013, symbol=r'$\odot$', color='#1E293B', bg='#FFFFFF', lw=1.1, fontsize=9.0, zorder=7):
    """Draw circular mathematical operator nodes."""
    circ = Circle((cx, cy), r, facecolor=bg, edgecolor=color, linewidth=lw, zorder=zorder)
    ax.add_patch(circ)
    ax.text(cx, cy, symbol, ha='center', va='center', fontsize=fontsize, fontweight='bold', color=color, zorder=zorder+1)

def draw_arrow(ax, p1, p2, color='#1E293B', lw=1.1, style='->', mut=8, zorder=6):
    """Draw clean directional vector arrows."""
    arrow = FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=mut,
                            color=color, linewidth=lw, zorder=zorder)
    ax.add_patch(arrow)

def render_optimal_gemma_clip_architecture():
    """
    Sơ đồ kiến trúc mới theo đúng chuẩn bài báo tham khảo (Tip-Adapter / CVPR):
    - Tên khối ở trong hộp bo góc màu kem/vàng/xanh nhẹ
    - Đầu ra trên mũi tên chỉ mang ĐÚNG 1 KÝ HIỆU TOÁN HỌC (f_test, W_c^T, f_test W_c^T, p, y_hat)
    - Toàn bộ công thức chi tiết được giải thích ở bảng Legend Directory dưới đáy.
    """
    fig, ax = plt.subplots(figsize=(20.0, 10.0), dpi=300)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis('off')
    
    # -------------------------------------------------------------------------
    # BẢNG MÀU CHUẨN KHOA HỌC
    # -------------------------------------------------------------------------
    c_text = '#0F172A'          # Slate-900
    c_sub = '#475569'           # Slate-600
    c_arrow = '#334155'         # Slate-700
    
    # Màu khối hộp phong cách Tip-Adapter
    c_box_yellow = '#FEF3C7'    # Vàng kem
    c_box_yellow_border = '#D97706'
    
    c_box_pink = '#FFE4E6'      # Hồng phấn nhẹ
    c_box_pink_border = '#E11D48'
    
    c_purple = '#6D28D9'        # Violet
    c_purple_soft = '#FAF5FF'
    
    c_indigo = '#4338CA'        # Indigo
    c_indigo_soft = '#EEF2FF'
    
    c_green = '#047857'         # Emerald
    c_green_soft = '#F0FDF4'
    
    c_amber = '#B45309'         # Amber

    # -------------------------------------------------------------------------
    # TIÊU ĐỀ CHÍNH
    # -------------------------------------------------------------------------
    ax.text(0.025, 0.968, "Optimal Gemma-CLIP: Dual-Stream Morphological Ensemble Architecture", 
            fontsize=14.0, fontweight='bold', color=c_text, va='center')
    ax.text(0.025, 0.944, "Coupling Google Gemma 4 Morphological Prior with Learnable Soft Tokens and 5-View TTA (92.15% Top-1 SOTA)", 
            fontsize=8.8, color=c_sub, va='center')
    ax.plot([0.025, 0.975], [0.928, 0.928], color='#E2E8F0', lw=1.0, zorder=1)

    # -------------------------------------------------------------------------
    # TIÊU ĐỀ 4 PANEL (A, B, C, D)
    # -------------------------------------------------------------------------
    ax.text(0.025, 0.905, "a   Knowledge Extraction & Dual Text Classifiers", 
            fontsize=9.2, fontweight='bold', color=c_purple, va='center')
    ax.text(0.535, 0.905, "c   Ensemble Fusion Center", 
            fontsize=9.2, fontweight='bold', color=c_amber, va='center')
    ax.text(0.770, 0.905, "d   Prediction & Validation", 
            fontsize=9.2, fontweight='bold', color='#2563EB', va='center')
    ax.text(0.025, 0.448, "b   5-View Test-Time Augmentation (TTA) & Visual Representation", 
            fontsize=9.2, fontweight='bold', color=c_green, va='center')

    # =========================================================================
    # a. NHÁNH VĂN BẢN (GEMMA 4 -> PROMPT BANK & GEMMA-COOP)
    # =========================================================================
    # Khối 1: Gemma 4 SLM Box
    slm_x, slm_y, slm_w, slm_h = 0.025, 0.485, 0.155, 0.395
    draw_round_box(ax, slm_x, slm_y, slm_w, slm_h, bg=c_purple_soft, border=c_purple, lw=1.1, rad=0.007)
    
    hdr_h = 0.034
    draw_round_box(ax, slm_x, slm_y + slm_h - hdr_h, slm_w, hdr_h, bg='#EDE9FE', border=c_purple, lw=0.9, rad=0.007)
    ax.text(slm_x + slm_w/2.0, slm_y + slm_h - hdr_h/2.0, "Google Gemma 4 SLM", 
            ha='center', va='center', fontsize=8.0, fontweight='bold', color='#4C1D95')
    
    descriptors = [
        ("• Head/Ears:", "triangular erect ears"),
        ("• Eyes:", "deep-set dark almond"),
        ("• Coat/Fur:", "dense red double coat"),
        ("• Body:", "compact muscular frame"),
        ("• Dist. Mark:", "curled tail, white urajiro")
    ]
    for idx, (slot, val) in enumerate(descriptors):
        item_y = slm_y + slm_h - hdr_h - 0.045 - idx * 0.046
        ax.text(slm_x + 0.010, item_y, slot, fontsize=6.2, fontweight='bold', color='#4C1D95')
        ax.text(slm_x + 0.010, item_y - 0.016, val, fontsize=5.8, color=c_text)
        
    ax.text(slm_x + slm_w/2.0, slm_y + 0.020, "Local Offline (LM Studio Server)", 
            ha='center', va='center', fontsize=5.6, fontstyle='italic', color=c_sub)

    # Đầu ra từ Gemma 4 chỉ mang đúng 1 ký hiệu: A_c
    slm_out_x = slm_x + slm_w
    slm_out_y = slm_y + slm_h * 0.50
    fork_x = 0.200
    draw_arrow(ax, (slm_out_x, slm_out_y), (fork_x, slm_out_y), color=c_arrow, lw=1.1)
    ax.text((slm_out_x + fork_x)/2.0, slm_out_y + 0.016, r"$\mathcal{A}_c$", ha='center', va='center', fontsize=8.8, fontweight='bold', color=c_purple)
    
    y_sub1 = 0.780  # Nhánh Centroid
    y_sub2 = 0.590  # Nhánh Gemma-CoOp
    
    ax.plot([fork_x, fork_x], [y_sub2, y_sub1], color=c_arrow, lw=1.1)
    draw_arrow(ax, (fork_x, y_sub1), (0.222, y_sub1), color=c_purple, lw=1.1)
    draw_arrow(ax, (fork_x, y_sub2), (0.222, y_sub2), color=c_indigo, lw=1.1)

    # -------------------------------------------------------------------------
    # Nhánh 1: Hierarchical Prompt Bank -> Text Encoder -> W_cen
    # -------------------------------------------------------------------------
    p1_x, p1_y, p1_w, p1_h = 0.225, 0.700, 0.145, 0.155
    draw_round_box(ax, p1_x, p1_y, p1_w, p1_h, bg='#FFFFFF', border=c_purple, lw=0.9, rad=0.006)
    
    p1_hdr_h = 0.028
    draw_round_box(ax, p1_x, p1_y + p1_h - p1_hdr_h, p1_w, p1_hdr_h, bg='#EDE9FE', border=c_purple, lw=0.8, rad=0.006)
    ax.text(p1_x + p1_w/2.0, p1_y + p1_h - p1_hdr_h/2.0, "Prompt Bank (K=6)", 
            ha='center', va='center', fontsize=7.2, fontweight='bold', color='#4C1D95')
    
    p1_lines = [
        r'$T_{c,1}$: "a photo of a {breed}"',
        r'$T_{c,2}$: "pet with {face}"',
        r'$T_{c,3..6}$: "pet with {traits}"',
        "6 structured templates/class"
    ]
    for idx, l_txt in enumerate(p1_lines):
        item_c = c_sub if idx == 3 else c_text
        item_fs = 5.6 if idx == 3 else 6.0
        ax.text(p1_x + 0.008, p1_y + p1_h - p1_hdr_h - 0.026 - idx * 0.026, l_txt, fontsize=item_fs, color=item_c)

    # Mũi tên dẫn vào Text Encoder
    draw_arrow(ax, (p1_x + p1_w, y_sub1), (0.395, y_sub1), color=c_arrow, lw=1.1)
    
    # Hộp Text Encoder phong cách Tip-Adapter (Màu kem bo góc)
    tenc1_x, tenc1_y, tenc1_w, tenc1_h = 0.398, 0.730, 0.052, 0.100
    draw_round_box(ax, tenc1_x, tenc1_y, tenc1_w, tenc1_h, bg=c_box_yellow, border=c_box_yellow_border, lw=1.0, rad=0.005)
    ax.text(tenc1_x + tenc1_w/2.0, tenc1_y + tenc1_h*0.62, "Text", ha='center', va='center', fontsize=7.4, fontweight='bold', color='#78350F')
    ax.text(tenc1_x + tenc1_w/2.0, tenc1_y + tenc1_h*0.35, "Encoder", ha='center', va='center', fontsize=7.4, fontweight='bold', color='#78350F')

    # Mũi tên xuất ra ma trận W_cen
    draw_arrow(ax, (tenc1_x + tenc1_w, y_sub1), (0.470, y_sub1), color=c_arrow, lw=1.1)
    
    # Ma trận W_cen (Hiển thị ký hiệu ngắn gọn)
    wcen_x, wcen_y, wcen_w, wcen_h = 0.475, 0.710, 0.024, 0.140
    draw_round_box(ax, wcen_x, wcen_y, wcen_w, wcen_h, bg=c_purple_soft, border=c_purple, lw=1.0, rad=0.004)
    for r in range(5):
        ax.plot([wcen_x, wcen_x + wcen_w], [wcen_y + r*(wcen_h/5), wcen_y + r*(wcen_h/5)], color='#DDD6FE', lw=0.5, zorder=4)
    ax.text(wcen_x + wcen_w/2.0, wcen_y + wcen_h + 0.016, r"$\mathbf{W}_{\mathrm{cen}}^\top$", 
            ha='center', va='center', fontsize=8.5, fontweight='bold', color=c_purple)
    ax.text(wcen_x + wcen_w/2.0, wcen_y - 0.012, r"$C$", 
            ha='center', va='center', fontsize=6.5, color=c_purple)

    # -------------------------------------------------------------------------
    # Nhánh 2: Gemma-CoOp Soft Prompt -> Text Encoder -> W_coop
    # -------------------------------------------------------------------------
    p2_x, p2_y, p2_w, p2_h = 0.225, 0.505, 0.145, 0.165
    draw_round_box(ax, p2_x, p2_y, p2_w, p2_h, bg='#FFFFFF', border=c_indigo, lw=0.9, rad=0.006)
    
    p2_hdr_h = 0.028
    draw_round_box(ax, p2_x, p2_y + p2_h - p2_hdr_h, p2_w, p2_hdr_h, bg='#E0E7FF', border=c_indigo, lw=0.8, rad=0.006)
    ax.text(p2_x + p2_w/2.0, p2_y + p2_h - p2_hdr_h/2.0, "Gemma-CoOp (M=4)", 
            ha='center', va='center', fontsize=7.2, fontweight='bold', color='#312E81')
    
    # Soft Tokens Pills
    pills = [
        ("[V1]", 0.016, '#EEF2FF', '#6366F1', '#3730A3'),
        ("[V2]", 0.016, '#EEF2FF', '#6366F1', '#3730A3'),
        ("[V3]", 0.016, '#EEF2FF', '#6366F1', '#3730A3'),
        ("[V4]", 0.016, '#EEF2FF', '#6366F1', '#3730A3'),
        ("c", 0.015, '#EFF6FF', '#3B82F6', '#1D4ED8'),
        (r"$\hat{a}_c$", 0.017, '#ECFDF5', '#10B981', '#047857')
    ]
    pill_y = p2_y + 0.075
    pill_h = 0.024
    cur_px = p2_x + 0.006
    for txt, pw, bgc, bdc, fgc in pills:
        draw_round_box(ax, cur_px, pill_y, pw, pill_h, bg=bgc, border=bdc, lw=0.6, rad=0.003)
        ax.text(cur_px + pw/2.0, pill_y + pill_h/2.0, txt, 
                ha='center', va='center', fontsize=5.6, fontweight='bold', color=fgc)
        cur_px += pw + 0.003
        
    ax.text(p2_x + 0.008, p2_y + 0.040, r"• Learnable context: $M=4$", fontsize=5.8, color=c_indigo)
    ax.text(p2_x + 0.008, p2_y + 0.018, r"• $2{,}048$ tunable parameters", fontsize=5.6, fontstyle='italic', color=c_sub)

    # Mũi tên dẫn vào Text Encoder 2
    draw_arrow(ax, (p2_x + p2_w, y_sub2), (0.395, y_sub2), color=c_arrow, lw=1.1)

    # Hộp Text Encoder 2
    tenc2_x, tenc2_y, tenc2_w, tenc2_h = 0.398, 0.540, 0.052, 0.100
    draw_round_box(ax, tenc2_x, tenc2_y, tenc2_w, tenc2_h, bg=c_box_yellow, border=c_box_yellow_border, lw=1.0, rad=0.005)
    ax.text(tenc2_x + tenc2_w/2.0, tenc2_y + tenc2_h*0.62, "Text", ha='center', va='center', fontsize=7.4, fontweight='bold', color='#78350F')
    ax.text(tenc2_x + tenc2_w/2.0, tenc2_y + tenc2_h*0.35, "Encoder", ha='center', va='center', fontsize=7.4, fontweight='bold', color='#78350F')

    # Mũi tên xuất ra ma trận W_coop
    draw_arrow(ax, (tenc2_x + tenc2_w, y_sub2), (0.470, y_sub2), color=c_arrow, lw=1.1)

    # Ma trận W_coop
    wcoop_x, wcoop_y, wcoop_w, wcoop_h = 0.475, 0.520, 0.024, 0.140
    draw_round_box(ax, wcoop_x, wcoop_y, wcoop_w, wcoop_h, bg=c_indigo_soft, border=c_indigo, lw=1.0, rad=0.004)
    for r in range(5):
        ax.plot([wcoop_x, wcoop_x + wcoop_w], [wcoop_y + r*(wcoop_h/5), wcoop_y + r*(wcoop_h/5)], color='#C7D2FE', lw=0.5, zorder=4)
    ax.text(wcoop_x + wcoop_w/2.0, wcoop_y + wcoop_h + 0.016, r"$\mathbf{W}_{\mathrm{coop}}^\top$", 
            ha='center', va='center', fontsize=8.5, fontweight='bold', color=c_indigo)
    ax.text(wcoop_x + wcoop_w/2.0, wcoop_y - 0.012, r"$C$", 
            ha='center', va='center', fontsize=6.5, color=c_indigo)

    # =========================================================================
    # b. NHÁNH THỊ GIÁC: 5-VIEW TTA -> VISUAL ENCODER -> TTA FUSION -> f
    # =========================================================================
    shiba_bgr = "./dataset/oxford-iiit-pet/images/shiba_inu_122.jpg"
    views = generate_5_views(shiba_bgr)
    v_titles = [r"$I_1$", r"$I_2$", r"$I_3$", r"$I_4$", r"$I_5$"]
    v_subs = ["Base", "H-Flip", "Bilateral", "CLAHE", "Zoom"]
    
    card_w, card_h = 0.046, 0.190
    img_w, img_h = 0.040, 0.088
    start_v_x = 0.025
    v_gap = 0.007
    
    for i in range(5):
        cx = start_v_x + i * (card_w + v_gap)
        cy = 0.210
        draw_round_box(ax, cx, cy, card_w, card_h, bg='#FFFFFF', border='#CBD5E1', lw=0.7, rad=0.004)
        
        ix = cx + (card_w - img_w)/2.0
        iy = cy + card_h - img_h - 0.008
        ax.imshow(views[i], extent=[ix, ix + img_w, iy, iy + img_h], zorder=4)
        ax.add_patch(Rectangle((ix, iy), img_w, img_h, fill=False, edgecolor='#94A3B8', linewidth=0.5, zorder=5))
        
        ax.text(cx + card_w/2.0, cy + 0.055, v_titles[i], ha='center', va='center', fontsize=7.2, fontweight='bold', color=c_text)
        ax.text(cx + card_w/2.0, cy + 0.030, v_subs[i], ha='center', va='center', fontsize=5.8, color=c_sub)
        
    tta_end_x = start_v_x + 5 * card_w + 4 * v_gap
    draw_arrow(ax, (tta_end_x + 0.005, 0.305), (0.300, 0.305), color=c_arrow, lw=1.1)

    # Hộp Visual Encoder (Phong cách Tip-Adapter: Hộp vàng kem, KHÔNG chứa công thức dài)
    venc_x, venc_y, venc_w, venc_h = 0.305, 0.255, 0.056, 0.100
    draw_round_box(ax, venc_x, venc_y, venc_w, venc_h, bg=c_box_yellow, border=c_box_yellow_border, lw=1.0, rad=0.005)
    ax.text(venc_x + venc_w/2.0, venc_y + venc_h*0.62, "Visual", ha='center', va='center', fontsize=7.6, fontweight='bold', color='#78350F')
    ax.text(venc_x + venc_w/2.0, venc_y + venc_h*0.35, "Encoder", ha='center', va='center', fontsize=7.6, fontweight='bold', color='#78350F')

    # Mũi tên từ Visual Encoder sang TTA Fusion mang đúng ký hiệu {f_v}
    draw_arrow(ax, (venc_x + venc_w, 0.305), (0.395, 0.305), color=c_arrow, lw=1.1)
    ax.text((venc_x + venc_w + 0.395)/2.0, 0.320, r"$\{\mathbf{f}_v\}$", ha='center', va='center', fontsize=7.8, fontweight='bold', color='#047857')

    # Hộp TTA Fusion (Đổi thành hộp sạch sẽ, KHÔNG ghi công thức dài dòng, chỉ ghi tên khối)
    fus_x, fus_y, fus_w, fus_h = 0.400, 0.255, 0.056, 0.100
    draw_round_box(ax, fus_x, fus_y, fus_w, fus_h, bg=c_green_soft, border=c_green, lw=1.0, rad=0.005)
    ax.text(fus_x + fus_w/2.0, fus_y + fus_h*0.62, "TTA", ha='center', va='center', fontsize=7.6, fontweight='bold', color='#064E3B')
    ax.text(fus_x + fus_w/2.0, fus_y + fus_h*0.35, "Fusion", ha='center', va='center', fontsize=7.6, fontweight='bold', color='#064E3B')

    # Mũi tên xuất ra ĐÚNG 1 KÝ HIỆU VECTOR: f
    draw_arrow(ax, (fus_x + fus_w, 0.305), (0.485, 0.305), color=c_arrow, lw=1.2)
    ax.text(0.480, 0.322, r"$\mathbf{f}$", ha='center', va='center', fontsize=9.2, fontweight='bold', color='#047857')

    # Biểu diễn vector f (Thanh chữ nhật nhỏ màu xanh lá nhạt với C chiều)
    f_bar_x, f_bar_y, f_bar_w, f_bar_h = 0.488, 0.295, 0.035, 0.020
    draw_round_box(ax, f_bar_x, f_bar_y, f_bar_w, f_bar_h, bg='#ECFDF5', border='#10B981', lw=0.9, rad=0.003)
    ax.text(f_bar_x + f_bar_w/2.0, f_bar_y - 0.010, r"$C$", ha='center', va='center', fontsize=6.2, color='#047857')

    # =========================================================================
    # c. KHU VỰC TRUNG TÂM: SO KHỚP COSINE VÀ NÚT HỢP NHẤT SOFTMAX
    # =========================================================================
    bus_x = 0.540
    r_op = 0.013

    op1_cx, op1_cy = bus_x, y_sub1
    op2_cx, op2_cy = bus_x, y_sub2

    # Trục Bus dọc dẫn vector f lên 2 toán tử Cosine
    draw_arrow(ax, (f_bar_x + f_bar_w, 0.305), (bus_x, 0.305), color=c_arrow, lw=1.1)
    draw_arrow(ax, (bus_x, 0.305), (bus_x, op2_cy - r_op - 0.002), color=c_arrow, lw=1.1)
    draw_arrow(ax, (bus_x, op2_cy + r_op + 0.002), (bus_x, op1_cy - r_op - 0.002), color=c_arrow, lw=1.1)

    # Mũi tên từ W_cen và W_coop vào các toán tử Cosine
    draw_arrow(ax, (wcen_x + wcen_w + 0.005, y_sub1), (op1_cx - r_op - 0.002, y_sub1), color=c_purple, lw=1.1)
    draw_arrow(ax, (wcoop_x + wcoop_w + 0.005, y_sub2), (op2_cx - r_op - 0.002, y_sub2), color=c_indigo, lw=1.1)

    # Vẽ nút tròn so khớp Cosine
    draw_op(ax, op1_cx, op1_cy, r=r_op, symbol=r'$\odot$', color=c_purple, bg='#FFFFFF')
    draw_op(ax, op2_cx, op2_cy, r=r_op, symbol=r'$\odot$', color=c_indigo, bg='#FFFFFF')

    # Ký hiệu biểu thức đầu ra của phép nhân Cosine (Phong cách Tip-Adapter: f W_cen^T)
    ax.text(op1_cx + r_op + 0.010, op1_cy + 0.014, r"$\mathbf{f}\mathbf{W}_{\mathrm{cen}}^\top$", 
            ha='left', va='center', fontsize=7.8, fontweight='bold', color=c_purple)
    ax.text(op1_cx + r_op + 0.010, op1_cy - 0.012, r"$\mathbf{s}_{\mathrm{cen}}$", ha='left', va='center', fontsize=7.0, color=c_sub)

    ax.text(op2_cx + r_op + 0.010, op2_cy + 0.014, r"$\mathbf{f}\mathbf{W}_{\mathrm{coop}}^\top$", 
            ha='left', va='center', fontsize=7.8, fontweight='bold', color=c_indigo)
    ax.text(op2_cx + r_op + 0.010, op2_cy - 0.012, r"$\mathbf{s}_{\mathrm{coop}}$", ha='left', va='center', fontsize=7.0, color=c_sub)

    # Nút hợp nhất Linear Combination Operator (+)
    ens_x = 0.680
    ens_y = (y_sub1 + y_sub2) / 2.0  # 0.685

    draw_arrow(ax, (op1_cx + 0.075, y_sub1), (ens_x - 0.016, ens_y + 0.008), color=c_purple, lw=1.1)
    draw_arrow(ax, (op2_cx + 0.075, y_sub2), (ens_x - 0.016, ens_y - 0.008), color=c_indigo, lw=1.1)
    
    ax.text((op1_cx + ens_x)/2.0 + 0.010, ens_y + 0.060, r"$\times (1-\beta)$", ha='center', va='center', fontsize=6.8, color=c_purple)
    ax.text((op2_cx + ens_x)/2.0 + 0.010, ens_y - 0.060, r"$\times \beta$", ha='center', va='center', fontsize=6.8, color=c_indigo)
    
    draw_op(ax, ens_x, ens_y, r=0.016, symbol=r'$\oplus$', color=c_amber, bg='#FFFBEB', lw=1.2, fontsize=10.5)
    ax.text(ens_x, ens_y + 0.028, "Ensemble", ha='center', va='center', fontsize=7.2, fontweight='bold', color='#78350F')

    # Mũi tên từ nút hợp nhất sang biểu đồ dự đoán mang ĐÚNG 1 KÝ HIỆU: p
    chart_x = 0.760
    draw_arrow(ax, (ens_x + 0.016, ens_y), (chart_x - 0.012, ens_y), color=c_amber, lw=1.2)
    ax.text((ens_x + chart_x)/2.0, ens_y + 0.016, r"$\mathbf{p}$", ha='center', va='center', fontsize=9.2, fontweight='bold', color=c_amber)

    # =========================================================================
    # d. BIỂU ĐỒ CỘT DỰ ĐOÁN & HUY HIỆU SOTA (Phong cách Tip-Adapter)
    # =========================================================================
    chart_y = 0.260
    chart_w = 0.210
    chart_h = 0.500
    
    draw_round_box(ax, chart_x - 0.010, chart_y - 0.045, chart_w + 0.020, chart_h + 0.125, 
                   bg='#FFFFFF', border='#CBD5E1', lw=0.9, rad=0.006)
    ax.text(chart_x + chart_w/2.0, chart_y + chart_h + 0.060, r"$\hat{y} = \arg\max_c (\mathbf{p}_c)$", 
            ha='center', va='center', fontsize=8.2, fontweight='bold', color=c_text)
    
    # Trục tọa độ
    ax.annotate("", xy=(chart_x, chart_y + chart_h + 0.025), xytext=(chart_x, chart_y),
                arrowprops=dict(arrowstyle="->", color=c_text, lw=0.9))
    ax.annotate("", xy=(chart_x + chart_w + 0.005, chart_y), xytext=(chart_x, chart_y),
                arrowprops=dict(arrowstyle="->", color=c_text, lw=0.9))
    
    classes = ["cat", "dog (shiba)", "panda", "rabbit"]
    probs = [0.035, 0.9215, 0.025, 0.018]
    colors = ['#3B82F6', '#10B981', '#F97316', '#EAB308']
    
    num_bars = len(classes)
    bar_width = chart_w / (num_bars + 1.0)
    
    for idx, (cls_name, p, bar_c) in enumerate(zip(classes, probs, colors)):
        bx = chart_x + (idx + 0.55) * bar_width
        bh = p * (chart_h * 0.85)
        
        is_target = ("shiba" in cls_name)
        b_rect = Rectangle((bx, chart_y), bar_width * 0.70, bh, 
                           facecolor=bar_c, edgecolor='#334155', linewidth=0.8, zorder=5)
        ax.add_patch(b_rect)
        
        fw = 'bold' if is_target else 'normal'
        ax.text(bx + bar_width * 0.35, chart_y - 0.012, cls_name, 
                ha='right', va='top', rotation=30, fontsize=6.8, fontstyle='italic', fontweight=fw, color=c_text)
        
        if is_target:
            ax.text(bx + bar_width * 0.35, chart_y + bh + 0.012, "92.15%", 
                    ha='center', va='bottom', fontsize=7.8, fontweight='bold', color='#047857')
            
    # Huy hiệu SOTA
    badge_x, badge_y, badge_w, badge_h = chart_x - 0.010, 0.050, chart_w + 0.020, 0.135
    draw_round_box(ax, badge_x, badge_y, badge_w, badge_h, bg=c_green_soft, border=c_green, lw=1.0, rad=0.005)
    ax.text(badge_x + badge_w/2.0, badge_y + badge_h - 0.024, "CROWNED SOTA RESULT", 
            ha='center', va='center', fontsize=6.5, fontweight='bold', color='#065F46')
    ax.text(badge_x + badge_w/2.0, badge_y + badge_h - 0.052, r"$\hat{y} = \mathrm{Shiba\ Inu}$", 
            ha='center', va='center', fontsize=8.6, fontweight='bold', color='#064E3B')
    ax.text(badge_x + badge_w/2.0, badge_y + 0.044, "Top-1: 92.15%  |  Top-5: 99.81%", 
            ha='center', va='center', fontsize=6.8, fontweight='bold', color='#047857')
    ax.text(badge_x + badge_w/2.0, badge_y + 0.020, "2,048 Parameters (0.001% Tuned)", 
            ha='center', va='center', fontsize=5.8, color='#059669')

    # =========================================================================
    # BẢNG CHÚ THÍCH KÝ HIỆU TOÁN HỌC DƯỚI ĐÁY (MATHEMATICAL LEGEND STRIP)
    # =========================================================================
    leg_x, leg_y, leg_w, leg_h = 0.025, 0.045, 0.710, 0.135
    draw_round_box(ax, leg_x, leg_y, leg_w, leg_h, bg='#F8FAFC', border='#CBD5E1', lw=0.8, rad=0.005)
    
    ax.text(leg_x + 0.012, leg_y + leg_h - 0.020, "Mathematical Notation & Symbol Directory (Reading Guide):", 
            fontsize=7.0, fontweight='bold', color='#0F172A')
    
    col1_x = leg_x + 0.012
    col2_x = leg_x + 0.250
    col3_x = leg_x + 0.490
    
    # Cột 1: Ký hiệu văn bản
    ax.text(col1_x, leg_y + 0.082, r"$\mathcal{A}_c = \mathrm{Gemma4}(\mathrm{Prompt}(c))$ : 5 traits", fontsize=5.8, color=c_purple)
    ax.text(col1_x, leg_y + 0.056, r"$\mathbf{W}_{\mathrm{cen}} \in \mathbb{R}^{37 \times 512}$ : Prompt centroid bank", fontsize=5.8, color=c_purple)
    ax.text(col1_x, leg_y + 0.030, r"$\mathbf{W}_{\mathrm{coop}} \in \mathbb{R}^{37 \times 512}$ : Tunable soft context", fontsize=5.8, color=c_indigo)
    ax.text(col1_x, leg_y + 0.010, r"$\mathbf{w}_c = \mathrm{Norm}(\sum_{k=1}^6 \mathbf{g}(T_{c,k}))$", fontsize=5.6, color=c_sub)

    # Cột 2: Ký hiệu thị giác
    ax.text(col2_x, leg_y + 0.082, r"$I_v \in \mathbb{R}^{224 \times 224 \times 3}$ : 5 TTA specimen views", fontsize=5.8, color='#047857')
    ax.text(col2_x, leg_y + 0.056, r"$\mathbf{f}_v = \mathbf{z}_v / \|\mathbf{z}_v\|_2$ : Unit-sphere embeddings", fontsize=5.8, color='#047857')
    ax.text(col2_x, leg_y + 0.030, r"$\mathbf{f} = \mathrm{Norm}(\sum_{v=1}^5 \mathbf{f}_v) \in \mathbb{S}^{511}$ : Invariant visual feature", fontsize=5.8, color='#047857')
    ax.text(col2_x, leg_y + 0.010, r"$\|\mathbf{f}\|_2 = 1.0$ (Invariant to flip, contrast, zoom)", fontsize=5.6, color=c_sub)

    # Cột 3: Ký hiệu suy luận & hợp nhất
    ax.text(col3_x, leg_y + 0.082, r"$\odot$ : Cosine dot product ($\tau = 0.01$)", fontsize=5.8, color='#B45309')
    ax.text(col3_x, leg_y + 0.056, r"$\mathbf{s}_{\mathrm{cen}}, \mathbf{s}_{\mathrm{coop}} \in \mathbb{R}^{37}$ : Modality logits", fontsize=5.8, color='#B45309')
    ax.text(col3_x, leg_y + 0.030, r"$\mathbf{p} = \beta \cdot \mathrm{Softmax}(\mathbf{s}_{\mathrm{coop}}) + (1-\beta)\cdot\mathrm{Softmax}(\mathbf{s}_{\mathrm{cen}})$", fontsize=5.4, color='#B45309')
    ax.text(col3_x, leg_y + 0.010, r"$\hat{y} = \arg\max_c (\mathbf{p}_c)$ : Final prediction", fontsize=5.6, color='#047857')

    # Xuất file PDF và PNG chất lượng cao
    out_dir = "enhanced/figures"
    os.makedirs(out_dir, exist_ok=True)
    pdf_path = os.path.join(out_dir, "fig1_framework_architecture.pdf")
    png_path = os.path.join(out_dir, "fig1_framework_architecture.png")
    
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight', pad_inches=0.05)
    plt.savefig(png_path, format='png', dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.close()
    print(f"Exported successfully to:\n- {pdf_path}\n- {png_path}")

if __name__ == "__main__":
    render_optimal_gemma_clip_architecture()
