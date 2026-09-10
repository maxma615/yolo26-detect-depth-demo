"""Generate the invite QR. Requires qrcode[pil], svglib and reportlab.

Official symbol: https://discord.com/branding (web/discord-symbol.svg).
The UI's white card supplies additional clear space around the image.
"""
from io import BytesIO
from pathlib import Path
import qrcode
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPM
from PIL import Image, ImageDraw

INVITE = "https://discord.gg/rzqg8TgUgb"

if __name__ == "__main__":
    qr = qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_H,
                       box_size=12, border=2)
    qr.add_data(INVITE)
    qr.make(fit=True)
    destination = Path(__file__).resolve().parents[1] / "web" / "discord.png"
    image = qr.make_image(fill_color="black", back_color="white").convert("RGB")
    # A small centered badge leaves finder/alignment patterns untouched.
    edge = image.width
    badge = 84
    start = (edge - badge) // 2
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((start - 8, start - 8, start + badge + 7,
                            start + badge + 7), radius=16, fill="white")
    draw.rounded_rectangle((start, start, start + badge - 1,
                            start + badge - 1), radius=12, fill="#5865F2")
    symbol = svg2rlg(str(destination.with_name("discord-symbol.svg")))
    logo = Image.open(BytesIO(renderPM.drawToString(
        symbol, fmt="PNG", bg=0x5865F2, dpi=216))).convert("RGBA")
    logo = logo.resize((56, 42), Image.Resampling.LANCZOS)
    image.paste(logo, ((edge - 56) // 2, (edge - 42) // 2), logo)
    image.save(destination)
    print(f"Saved {destination}: {INVITE}")
