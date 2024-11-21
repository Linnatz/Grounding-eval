import os
import json
import re
import ast
from PIL import Image, ImageDraw, ImageFont
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import io
import numpy as np
import random

def parse_boxes_and_refs(value):
    matches = re.findall(r"<ref>(.*?)</ref>\s*<box>(.*?)</box>", value)
    negative_matches = re.findall(r"<ref>(.*?)</ref>None", value)
    refs = [match[0] for match in matches]
    boxes = [ast.literal_eval(match[1]) for match in matches]
    if negative_matches:
        neg_refs = [match[0] for neg_match in negative_matches]
        return refs, boxes, neg_refs
    else:
        return refs, boxes, []

def normalize_to_pixel(box, width, height):
    pixel_boxes = []
    for coords in box:
        x1 = coords[0] * width / 1000
        y1 = coords[1] * height / 1000
        x2 = coords[2] * width / 1000
        y2 = coords[3] * height / 1000
        pixel_boxes.append([x1, y1, x2, y2])
    return pixel_boxes

def draw_boxes_with_labels(image_path, refs, boxes, neg_refs, width, height):
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    font_path = "/mnt/afs/user/luojiapeng/.fonts/msyh.ttc" 
    font = ImageFont.truetype(font_path, 16)

    for ref, box_list in zip(refs, boxes):
        pixel_boxes = normalize_to_pixel(box_list, width, height)
        for pixel_box in pixel_boxes:
            x1, y1, x2, y2 = pixel_box
            
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            
            draw.text((x1, y1 -20), ref, fill="red", font=font)

    if neg_refs:
        text = "Negative refs:\n"
        for neg_ref in neg_refs:
            text += neg_ref + " "

        text_image = Image.new("RGB", (width, 30 + 40))
        text_draw = ImageDraw.Draw(text_image)
        text_draw.multiline_text((20, 20), text, "#FFFFFF", font, spacing=5)
        answer_image = Image.fromarray(
            np.concatenate([np.asarray(image), np.asarray(text_image)])
        )
        return answer_image
    return image


def save_images_to_pdf(jsonl_file, output_pdf, n=30):

    pdf = canvas.Canvas(output_pdf)
    
    with open(jsonl_file, "r") as file:
        data = file.readlines()
        sample_data = random.sample(data, n)
        for line in sample_data:
            data = json.loads(line.strip())
            image_path = data["image"]
            conversations = data["conversations"]
            width = data["width"]
            height = data["height"]

            refs, boxes, neg_refs = [], [], []
            for conversation in conversations:
                if conversation["from"] == "gpt":
                    ref_list, box_list, neg_ref_list = parse_boxes_and_refs(conversation["value"])
                    refs.extend(ref_list)
                    boxes.extend(box_list)
                    neg_refs.extend(neg_ref_list)

            annotated_image = draw_boxes_with_labels(image_path, refs, boxes, neg_refs, width, height)

            image_io = io.BytesIO()
            annotated_image.save(image_io, format="PNG")
            image_io.seek(0)
            image_reader = ImageReader(image_io)

            pdf.setPageSize((width, height))
            pdf.drawImage(image_reader, 0, 0)
            pdf.showPage()

    pdf.save()

jsonl_file = "/mnt/afs2/dailinjun/workdir/model_train/train_data/vit0.3b_qwen2.5_0.5b/coco_grounding_1115_r115362.jsonl"  
output_pdf = "output.pdf"  
n = 30
save_images_to_pdf(jsonl_file, output_pdf, n)
