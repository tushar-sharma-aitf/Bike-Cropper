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


def crop_type_1_basic(image):
    """Type 1: Extreme tight crop - 0px horizontal, 3px vertical"""
    img_array = np.array(image)
    coords = detect_motorcycle(img_array)
    
    if coords is None:
        return image, image, "No motorcycle detected"
    
    x1, y1, x2, y2 = coords
    h, w = img_array.shape[:2]
    
    padding_x = 0
    padding_y = 3
    
    x1_p = max(0, x1 - padding_x)
    y1_p = max(0, y1 - padding_y)
    x2_p = min(w, x2 + padding_x)
    y2_p = min(h, y2 + padding_y)
    
    bbox_img = draw_bounding_box(img_array, coords, padding_x, padding_y)
    cropped = img_array[y1_p:y2_p, x1_p:x2_p]
    
    return Image.fromarray(cropped), Image.fromarray(bbox_img), f"✓ Extreme tight! Margin: H={padding_x}px V={padding_y}px | Size: {x2_p-x1_p}x{y2_p-y1_p}px"


def crop_type_2_size_aware(image, padding_percent, target_width, apply_resize):
    """Type 2: Minimal padding - extreme reduction"""
    img_array = np.array(image)
    coords = detect_motorcycle(img_array)
    
    if coords is None:
        return image, image, "No motorcycle detected"
    
    x1, y1, x2, y2 = coords
    h, w = img_array.shape[:2]
    
    bike_width = x2 - x1
    bike_height = y2 - y1
    
    base_padding_x = int(bike_width * (padding_percent / 100.0))
    base_padding_y = int(bike_height * (padding_percent / 100.0))
    
    padding_x = max(0, base_padding_x // 10)
    padding_y = max(3, base_padding_y // 5)
    
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
    
    return Image.fromarray(cropped), Image.fromarray(bbox_img), f"✓ Extreme tight! H={padding_x}px V={padding_y}px{resize_info}"


def crop_type_3_ground_removal_with_solid_color(image):
    """
    Type 3: Ground removal with SIMPLE SOLID COLOR top
    No AI generation, no blur - just solid color matching background
    """
    img_array = np.array(image)
    coords = detect_motorcycle(img_array)
    
    if coords is None:
        return image, image, "No motorcycle detected"
    
    x1, y1, x2, y2 = coords
    h, w = img_array.shape[:2]
    
    # Extreme tight
    padding_x = 0
    padding_top = 3
    padding_bottom = 3
    
    x1_p = max(0, x1 - padding_x)
    y1_p = max(0, y1 - padding_top)
    x2_p = min(w, x2 + padding_x)
    y2_p = min(h, y2 + padding_bottom)
    
    bbox_img = draw_bounding_box(img_array, coords, padding_x, padding_top)
    
    floor_height = h - y2_p
    
    if floor_height <= 0:
        return image, image, "No ground to remove"
    
    # Crop WITHOUT floor
    cropped_no_floor = img_array[y1_p:y2_p, x1_p:x2_p].copy()
    crop_h, crop_w = cropped_no_floor.shape[:2]
    
    if crop_h <= 0 or crop_w <= 0:
        return image, image, "Invalid crop"
    
    # Sample background color from TOP 10% of image (average color)
    sample_height = max(10, int(crop_h * 0.1))
    top_sample = cropped_no_floor[:sample_height, :]
    
    # Get average background color (median for robustness)
    avg_color = np.median(top_sample, axis=(0, 1)).astype(np.uint8)
    
    # Create SOLID COLOR top section (same size as removed floor)
    solid_color_top = np.full((floor_height, crop_w, 3), avg_color, dtype=np.uint8)
    
    # Stack: solid color top + original (no floor)
    result = np.vstack([solid_color_top, cropped_no_floor])
    
    final_h, final_w = result.shape[:2]
    
    return Image.fromarray(result), Image.fromarray(bbox_img), f"✓ Simple solid color! H={padding_x}px V={padding_top}px | Floor removed: {floor_height}px | Color top: {floor_height}px | RGB: {avg_color}"


# Bilingual text
text = {
    "title_en": "🏍️ Motorcycle Cropping System",
    "title_ja": "🏍️ バイククロッピングシステム",
    "subtitle_en": "Extreme tight cropping with simple solid color backgrounds. Powered by YOLOv8.",
    "subtitle_ja": "シンプルな単色背景で極限タイトクロッピング。YOLOv8搭載。",
    "tab1_en": "Type 1: Extreme Tight",
    "tab1_ja": "タイプ1：極限タイト",
    "tab1_desc_en": "0px horizontal, 3px vertical margins",
    "tab1_desc_ja": "水平0px、垂直3pxマージン",
    "tab2_en": "Type 2: Extreme Custom",
    "tab2_ja": "タイプ2：極限カスタム",
    "tab2_desc_en": "Minimal padding (90% reduction)",
    "tab2_desc_ja": "最小パディング（90%削減）",
    "tab3_en": "Type 3: Solid Color",
    "tab3_ja": "タイプ3：単色背景",
    "tab3_desc_en": "0px horizontal, removes ground, adds solid color top",
    "tab3_desc_ja": "水平0px、地面削除、単色上部追加",
    "upload_en": "Upload Motorcycle Image",
    "upload_ja": "バイク画像をアップロード",
    "btn1_en": "Extreme Tight",
    "btn1_ja": "極限タイト",
    "btn2_en": "Extreme Custom",
    "btn2_ja": "極限カスタム",
    "btn3_en": "Solid Color Top",
    "btn3_ja": "単色上部",
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
    
    subtitle_md = gr.Markdown("Extreme tight cropping with simple solid color backgrounds. Powered by YOLOv8.")
    
    # Type 1
    with gr.Tab("Type 1: Extreme Tight") as tab1:
        tab1_desc = gr.Markdown("### 0px horizontal, 3px vertical margins")
        with gr.Row():
            with gr.Column():
                input_img_1 = gr.Image(type="pil", label="Upload Motorcycle Image")
                btn_1 = gr.Button("Extreme Tight", variant="primary")
            with gr.Column():
                output_img_1 = gr.Image(type="pil", label="Cropped Result")
                bbox_img_1 = gr.Image(type="pil", label="Bounding Box")
                output_text_1 = gr.Textbox(label="Info")
        
        btn_1.click(crop_type_1_basic, inputs=input_img_1, outputs=[output_img_1, bbox_img_1, output_text_1])
    
    # Type 2
    with gr.Tab("Type 2: Extreme Custom") as tab2:
        tab2_desc = gr.Markdown("### Minimal padding (90% reduction)")
        with gr.Row():
            with gr.Column():
                input_img_2 = gr.Image(type="pil", label="Upload Motorcycle Image")
                padding_slider = gr.Slider(0, 50, value=10, step=1, label="Padding % (applies 10%)")
                target_width_slider = gr.Slider(0, 2000, value=800, step=50, label="Target Width")
                resize_checkbox = gr.Checkbox(label="Enable Resize", value=True)
                btn_2 = gr.Button("Extreme Custom", variant="primary")
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
    with gr.Tab("Type 3: Solid Color") as tab3:
        tab3_desc = gr.Markdown("### 0px horizontal, removes ground, adds solid color top")
        with gr.Row():
            with gr.Column():
                input_img_3 = gr.Image(type="pil", label="Upload Motorcycle Image")
                btn_3 = gr.Button("Solid Color Top", variant="primary")
            with gr.Column():
                output_img_3 = gr.Image(type="pil", label="Cropped Result")
                bbox_img_3 = gr.Image(type="pil", label="Bounding Box")
                output_text_3 = gr.Textbox(label="Info")
        
        btn_3.click(crop_type_3_ground_removal_with_solid_color, inputs=input_img_3, outputs=[output_img_3, bbox_img_3, output_text_3])
    
    tips_md = gr.Markdown("---\n💡 **Type 3:** Removes floor, adds simple solid color at top matching background - clean and fast!")
    
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
