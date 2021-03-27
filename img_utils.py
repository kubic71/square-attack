from PIL import Image, ImageDraw
import numpy as np

def write_text_to_img(img, text, max_lines=20):
    """Writes text to the top of the image"""

    font_size = 10
    text = "\n".join(text.split("\n")[:max_lines])
    n_lines = len(text.split("\n"))
    margin = int(n_lines * font_size * 1.5)

    width, height = img.size
    new_img = Image.new("RGB", size=(width + 200, max(height, margin)))
    new_img.paste(img, (0, 0))

    draw = ImageDraw.Draw(new_img)
    draw.text((width + 10, 0), text, (255,255,255))

    return new_img


def convert_to_pillow(img):
    # convert from channels-first to channels-last
    img = img.transpose(1, 2, 0)
    img = (img * 255).astype(np.uint8)
    return Image.fromarray(img)