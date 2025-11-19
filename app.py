import gradio as gr
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont

# Load YOLOv8 detection model
model_detect = YOLO('yolov8n.pt')


def detect_motorcycle(image):
    """Detect motorcycle and return bounding box coordinates"""
    results = model_detect.predict(image, classes=[3], conf=0.3)
    
    for result in results:
        for box in result.boxes:
            if int(box.cls[0]) == 3:
                coords = box.xyxy[0].int().tolist()
                return coords
    return None


def draw_bounding_box(image, coords, padding_x=0, padding_y=0):
    """Draw bounding box on original image"""
    img_pil = Image.fromarray(image.copy())
    draw = ImageDraw.Draw(img_pil)
    
    x1, y1, x2, y2 = coords
    h, w = image.shape[:2]
    
    x1_padded = max(0, x1 - padding_x)
    y1_padded = max(0, y1 - padding_y)
    x2_padded = min(w, x2 + padding_x)
    y2_padded = min(h, y2 + padding_y)
    
    draw.rectangle([x1_padded, y1_padded, x2_padded, y2_padded], outline=(0, 255, 0), width=5)
    
    try:
        font = ImageFont.truetype("arial.ttf", 30)
    except:
        font = ImageFont.load_default()
    
    text = "Motorcycle"
    text_bbox = draw.textbbox((x1_padded, y1_padded - 35), text, font=font)
    draw.rectangle([text_bbox[0]-5, text_bbox[1]-5, text_bbox[2]+5, text_bbox[3]+5], fill=(0, 255, 0))
    draw.text((x1_padded, y1_padded - 35), text, fill=(0, 0, 0), font=font)
    
    return np.array(img_pil)


def generate_background_seamless(reference_sample, target_height, target_width):
    """Generate seamless background with perfect blending"""
    ref_h, ref_w = reference_sample.shape[:2]
    
    if target_height <= 0 or target_width <= 0 or ref_h < 5 or ref_w < 5:
        avg_color = np.mean(reference_sample, axis=(0, 1)).astype(np.uint8)
        return np.full((max(1, target_height), target_width, 3), avg_color, dtype=np.uint8)
    
    reference_resized = cv2.resize(reference_sample, (target_width, ref_h), interpolation=cv2.INTER_LANCZOS4)
    
    blend_zone_height = min(ref_h // 2, 50)
    blend_reference = reference_resized[:blend_zone_height, :].copy()
    
    generated = np.zeros((target_height, target_width, 3), dtype=np.float32)
    
    for i in range(target_height):
        progress = i / max(1, target_height)
        sample_y = int(progress * (blend_zone_height - 1))
        sampled_row = blend_reference[sample_y:sample_y+1, :]
        variation = np.random.normal(0, 1, (1, target_width, 3))
        generated[i:i+1, :] = sampled_row + variation
    
    generated = np.clip(generated, 0, 255).astype(np.uint8)
    
    kernel_size = max(21, target_height // 5)
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel_size = min(kernel_size, 99)
    
    if kernel_size >= 3:
        generated = cv2.GaussianBlur(generated, (kernel_size, kernel_size), 0)
    
    transition_height = min(40, target_height // 3, ref_h // 3)
    
    if transition_height > 0:
        connection_row = reference_resized[0:1, :].copy()
        
        for i in range(transition_height):
            row_idx = target_height - transition_height + i
            if row_idx >= 0 and row_idx < target_height:
                t = i / transition_height
                alpha = 1 / (1 + np.exp(-10 * (t - 0.5)))
                
                generated[row_idx, :] = (
                    generated[row_idx, :].astype(np.float32) * (1 - alpha) +
                    connection_row.astype(np.float32) * alpha
                ).astype(np.uint8)
        
        generated[-1:, :] = connection_row
    
    for c in range(3):
        ref_channel = blend_reference[:, :, c].flatten()
        gen_channel = generated[:, :, c].flatten()
        
        gen_mean = gen_channel.mean()
        gen_std = max(gen_channel.std(), 0.1)
        ref_mean = ref_channel.mean()
        ref_std = ref_channel.std()
        
        normalized = (gen_channel - gen_mean) * (ref_std / (gen_std + 1e-6)) + ref_mean
        generated[:, :, c] = np.clip(normalized, 0, 255).reshape(target_height, target_width).astype(np.uint8)
    
    generated = cv2.GaussianBlur(generated, (15, 15), 0)
    
    return generated


def crop_type_1_basic(image):
    """Type 1: Basic crop with fixed padding"""
    img_array = np.array(image)
    coords = detect_motorcycle(img_array)
    
    if coords is None:
        return image, image, "No motorcycle detected"
    
    x1, y1, x2, y2 = coords
    h, w = img_array.shape[:2]
    
    padding = 20
    x1_p = max(0, x1 - padding)
    y1_p = max(0, y1 - padding)
    x2_p = min(w, x2 + padding)
    y2_p = min(h, y2 + padding)
    
    bbox_img = draw_bounding_box(img_array, coords, padding, padding)
    cropped = img_array[y1_p:y2_p, x1_p:x2_p]
    ground_removed = h - y2_p
    
    return Image.fromarray(cropped), Image.fromarray(bbox_img), f"✓ Cropped! Size: {x2_p-x1_p}x{y2_p-y1_p}px | Ground removed: {ground_removed}px"


def crop_type_2_size_aware(image, padding_percent, target_width, apply_resize):
    """Type 2: Custom crop with adjustable parameters"""
    img_array = np.array(image)
    coords = detect_motorcycle(img_array)
    
    if coords is None:
        return image, image, "No motorcycle detected"
    
    x1, y1, x2, y2 = coords
    h, w = img_array.shape[:2]
    
    bike_width = x2 - x1
    bike_height = y2 - y1
    
    padding_x = int(bike_width * (padding_percent / 100.0))
    padding_y = int(bike_height * (padding_percent / 100.0))
    
    x1_p = max(0, x1 - padding_x)
    y1_p = max(0, y1 - padding_y)
    x2_p = min(w, x2 + padding_x)
    y2_p = min(h, y2 + padding_y)
    
    bbox_img = draw_bounding_box(img_array, coords, padding_x, padding_y)
    cropped = img_array[y1_p:y2_p, x1_p:x2_p]
    
    resize_info = ""
    if apply_resize and target_width > 0:
        current_width = x2_p - x1_p
        scale = target_width / current_width
        new_width = int(current_width * scale)
        new_height = int((y2_p - y1_p) * scale)
        cropped = cv2.resize(cropped, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
        resize_info = f" | Resized: {new_width}x{new_height}px"
    
    ground_removed = h - y2_p
    
    return Image.fromarray(cropped), Image.fromarray(bbox_img), f"✓ Custom crop! Padding: {padding_percent}% ({padding_x}×{padding_y}px){resize_info}"


def crop_type_3_ground_removal_with_generation(image):
    """
    Type 3: Remove ground to tyres + Generate background on top
    1. Crops ground to just below tyres (small margin)
    2. Adds same area on top with AI-generated background
    """
    img_array = np.array(image)
    coords = detect_motorcycle(img_array)
    
    if coords is None:
        return image, image, "No motorcycle detected"
    
    x1, y1, x2, y2 = coords
    h, w = img_array.shape[:2]
    
    bike_width = x2 - x1
    bike_height = y2 - y1
    
    # Horizontal padding
    padding_x = int(bike_width * 0.1)
    # Vertical padding - tight to tyres
    padding_top = int(bike_height * 0.05)
    padding_bottom = int(bike_height * 0.02)  # Small margin at tyres
    
    x1_p = max(0, x1 - padding_x)
    y1_p = max(0, y1 - padding_top)
    x2_p = min(w, x2 + padding_x)
    y2_p = min(h, y2 + padding_bottom)  # Crop to tyres
    
    # Create bounding box
    bbox_img = draw_bounding_box(img_array, coords, padding_x, padding_top)
    
    # Calculate floor height to remove
    floor_height = h - y2_p
    
    if floor_height <= 0:
        return image, image, "No ground to remove"
    
    # Crop WITHOUT floor
    cropped_no_floor = img_array[y1_p:y2_p, x1_p:x2_p].copy()
    crop_h, crop_w = cropped_no_floor.shape[:2]
    
    if crop_h <= 0 or crop_w <= 0:
        return image, image, "Invalid crop"
    
    # Sample reference from top 50%
    reference_height = max(20, int(crop_h * 0.5))
    reference_sample = cropped_no_floor[:reference_height, :].copy()
    
    # Generate background for top
    try:
        extended_top = generate_background_seamless(reference_sample, floor_height, crop_w)
    except Exception as e:
        avg_color = np.mean(reference_sample, axis=(0, 1)).astype(np.uint8)
        extended_top = np.full((floor_height, crop_w, 3), avg_color, dtype=np.uint8)
        print(f"Background error: {e}")
    
    # Stack: generated top + original (no floor)
    result = np.vstack([extended_top, cropped_no_floor])
    
    # Apply junction blur for seamless blending
    junction_line = floor_height
    blur_zone = 30
    
    if blur_zone > 0 and junction_line - blur_zone // 2 >= 0 and junction_line + blur_zone // 2 < result.shape[0]:
        y_start = max(0, junction_line - blur_zone // 2)
        y_end = min(result.shape[0], junction_line + blur_zone // 2)
        
        junction_region = result[y_start:y_end, :].copy()
        junction_blurred = cv2.GaussianBlur(junction_region, (31, 31), 0)
        
        for i, y in enumerate(range(y_start, y_end)):
            distance = abs(y - junction_line)
            weight = np.exp(-distance / (blur_zone / 4))
            
            result[y, :] = (
                result[y, :].astype(np.float32) * (1 - weight) +
                junction_blurred[i, :].astype(np.float32) * weight
            ).astype(np.uint8)
    
    final_h, final_w = result.shape[:2]
    
    return Image.fromarray(result), Image.fromarray(bbox_img), f"✓ Ground removed & top generated! Floor removed: {floor_height}px | Generated top: {floor_height}px | Size: {final_w}x{final_h}px"


# Bilingual text
text = {
    "title_en": "🏍️ Motorcycle Cropping System",
    "title_ja": "🏍️ バイククロッピングシステム",
    "subtitle_en": "Upload a motorcycle image and choose a cropping method. Powered by YOLOv8.",
    "subtitle_ja": "バイク画像をアップロードして、クロッピング方法を選択してください。YOLOv8搭載。",
    "tab1_en": "Type 1: Basic Crop",
    "tab1_ja": "タイプ1：基本クロップ",
    "tab1_desc_en": "Fixed padding (20px) with bounding box",
    "tab1_desc_ja": "固定パディング（20px）とバウンディングボックス",
    "tab2_en": "Type 2: Custom Crop",
    "tab2_ja": "タイプ2：カスタムクロップ",
    "tab2_desc_en": "Adjustable padding and size",
    "tab2_desc_ja": "調整可能なパディングとサイズ",
    "tab3_en": "Type 3: AI Background",
    "tab3_ja": "タイプ3：AI背景生成",
    "tab3_desc_en": "Removes ground to tyres, generates background on top",
    "tab3_desc_ja": "タイヤまで地面削除、上部に背景を生成",
    "upload_en": "Upload Motorcycle Image",
    "upload_ja": "バイク画像をアップロード",
    "btn1_en": "Crop Basic",
    "btn1_ja": "基本クロップ",
    "btn2_en": "Crop Custom",
    "btn2_ja": "カスタムクロップ",
    "btn3_en": "Generate Background",
    "btn3_ja": "背景を生成",
    "result_en": "Cropped Result",
    "result_ja": "クロップ結果",
    "bbox_en": "Bounding Box",
    "bbox_ja": "バウンディングボックス",
    "info_en": "Info",
    "info_ja": "情報"
}

# Custom CSS
custom_css = """
#lang_dropdown {
    min-width: 120px !important;
    max-width: 120px !important;
}
.gradio-container {
    max-width: 1600px !important;
}
"""

# Create interface
with gr.Blocks(title="Motorcycle Cropping System", css=custom_css) as demo:
    
    current_lang = gr.State("en")
    
    with gr.Row():
        with gr.Column(scale=10):
            title_md = gr.Markdown("# 🏍️ Motorcycle Cropping System")
        with gr.Column(scale=1, min_width=120):
            language = gr.Dropdown(
                choices=["English", "日本語"],
                value="English",
                label="Language",
                elem_id="lang_dropdown",
                container=False,
                show_label=False
            )
    
    subtitle_md = gr.Markdown("Upload a motorcycle image and choose a cropping method. Powered by YOLOv8.")
    
    # Type 1
    with gr.Tab("Type 1: Basic Crop") as tab1:
        tab1_desc = gr.Markdown("### Fixed padding (20px) with bounding box")
        with gr.Row():
            with gr.Column():
                input_img_1 = gr.Image(type="pil", label="Upload Motorcycle Image")
                btn_1 = gr.Button("Crop Basic", variant="primary")
            with gr.Column():
                output_img_1 = gr.Image(type="pil", label="Cropped Result")
                bbox_img_1 = gr.Image(type="pil", label="Bounding Box")
                output_text_1 = gr.Textbox(label="Info")
        
        btn_1.click(crop_type_1_basic, inputs=input_img_1, outputs=[output_img_1, bbox_img_1, output_text_1])
    
    # Type 2
    with gr.Tab("Type 2: Custom Crop") as tab2:
        tab2_desc = gr.Markdown("### Adjustable padding and size")
        with gr.Row():
            with gr.Column():
                input_img_2 = gr.Image(type="pil", label="Upload Motorcycle Image")
                padding_slider = gr.Slider(0, 50, value=10, step=1, label="Padding %")
                target_width_slider = gr.Slider(0, 2000, value=800, step=50, label="Target Width")
                resize_checkbox = gr.Checkbox(label="Enable Resize", value=True)
                btn_2 = gr.Button("Crop Custom", variant="primary")
            with gr.Column():
                output_img_2 = gr.Image(type="pil", label="Cropped Result")
                bbox_img_2 = gr.Image(type="pil", label="Bounding Box")
                output_text_2 = gr.Textbox(label="Info")
        
        btn_2.click(
            crop_type_2_size_aware,
            inputs=[input_img_2, padding_slider, target_width_slider, resize_checkbox],
            outputs=[output_img_2, bbox_img_2, output_text_2]
        )
    
    # Type 3
    with gr.Tab("Type 3: AI Background") as tab3:
        tab3_desc = gr.Markdown("### Removes ground to tyres, generates background on top")
        with gr.Row():
            with gr.Column():
                input_img_3 = gr.Image(type="pil", label="Upload Motorcycle Image")
                btn_3 = gr.Button("Generate Background", variant="primary")
            with gr.Column():
                output_img_3 = gr.Image(type="pil", label="Cropped Result")
                bbox_img_3 = gr.Image(type="pil", label="Bounding Box")
                output_text_3 = gr.Textbox(label="Info")
        
        btn_3.click(crop_type_3_ground_removal_with_generation, inputs=input_img_3, outputs=[output_img_3, bbox_img_3, output_text_3])
    
    tips_md = gr.Markdown("---\n💡 **Tips:** Type 1: Quick | Type 2: Customizable | Type 3: Ground removal + AI background generation")
    
    # Language switching
    def switch_language(choice, curr_lang):
        new_lang = "ja" if choice == "日本語" else "en"
        return [
            f"# {text[f'title_{new_lang}']}",
            text[f"subtitle_{new_lang}"],
            gr.update(label=text[f"tab1_{new_lang}"]),
            f"### {text[f'tab1_desc_{new_lang}']}",
            gr.update(label=text[f"tab2_{new_lang}"]),
            f"### {text[f'tab2_desc_{new_lang}']}",
            gr.update(label=text[f"tab3_{new_lang}"]),
            f"### {text[f'tab3_desc_{new_lang}']}",
            gr.update(label=text[f"upload_{new_lang}"]),
            gr.update(value=text[f"btn1_{new_lang}"]),
            gr.update(label=text[f"result_{new_lang}"]),
            gr.update(label=text[f"bbox_{new_lang}"]),
            gr.update(label=text[f"info_{new_lang}"]),
            gr.update(label=text[f"upload_{new_lang}"]),
            gr.update(value=text[f"btn2_{new_lang}"]),
            gr.update(label=text[f"result_{new_lang}"]),
            gr.update(label=text[f"bbox_{new_lang}"]),
            gr.update(label=text[f"info_{new_lang}"]),
            gr.update(label=text[f"upload_{new_lang}"]),
            gr.update(value=text[f"btn3_{new_lang}"]),
            gr.update(label=text[f"result_{new_lang}"]),
            gr.update(label=text[f"bbox_{new_lang}"]),
            gr.update(label=text[f"info_{new_lang}"]),
            new_lang
        ]
    
    language.change(
        switch_language,
        inputs=[language, current_lang],
        outputs=[
            title_md, subtitle_md,
            tab1, tab1_desc, tab2, tab2_desc, tab3, tab3_desc,
            input_img_1, btn_1, output_img_1, bbox_img_1, output_text_1,
            input_img_2, btn_2, output_img_2, bbox_img_2, output_text_2,
            input_img_3, btn_3, output_img_3, bbox_img_3, output_text_3,
            current_lang
        ]
    )


if __name__ == "__main__":
    demo.launch()
