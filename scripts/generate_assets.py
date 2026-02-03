#!/usr/bin/env python3
"""Generate favicon and Open Graph images for METTLE."""

import os
from pathlib import Path

# Try to import PIL, fallback to cairosvg if available
try:
    from PIL import Image, ImageDraw, ImageFont

    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("Note: PIL not installed. Install with: pip install Pillow")

try:
    import cairosvg

    HAS_CAIRO = True
except ImportError:
    HAS_CAIRO = False


STATIC_DIR = Path(__file__).parent.parent / "static"

# SVG source for favicon
FAVICON_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
  <rect width="100" height="100" rx="20" fill="#0f172a"/>
  <path d="M20 75V25l20 25 20-25 20 25 20-25v50"
        stroke="#6366f1" stroke-width="8" fill="none" stroke-linecap="round" stroke-linejoin="round"/>
  <circle cx="75" cy="30" r="8" fill="#22c55e"/>
</svg>"""


def generate_favicon_pngs():
    """Generate PNG favicons from SVG using cairosvg."""
    if not HAS_CAIRO:
        print("cairosvg not installed. Install with: pip install cairosvg")
        print("Skipping PNG favicon generation.")
        return

    sizes = {
        "favicon-16.png": 16,
        "favicon-32.png": 32,
        "favicon-48.png": 48,
        "favicon-192.png": 192,
        "favicon-512.png": 512,
        "apple-touch-icon.png": 180,
    }

    for filename, size in sizes.items():
        output_path = STATIC_DIR / filename
        cairosvg.svg2png(
            bytestring=FAVICON_SVG.encode(),
            write_to=str(output_path),
            output_width=size,
            output_height=size,
        )
        print(f"Created {filename} ({size}x{size})")


def generate_favicon_ico():
    """Generate multi-resolution ICO from PNGs."""
    if not HAS_PIL:
        print("Pillow not installed. Skipping ICO generation.")
        return

    ico_sizes = [16, 32, 48]
    images = []

    for size in ico_sizes:
        png_path = STATIC_DIR / f"favicon-{size}.png"
        if png_path.exists():
            img = Image.open(png_path)
            images.append(img)

    if images:
        ico_path = STATIC_DIR / "favicon.ico"
        images[0].save(
            ico_path,
            format="ICO",
            sizes=[(img.width, img.height) for img in images],
            append_images=images[1:] if len(images) > 1 else [],
        )
        print("Created favicon.ico")
    else:
        print("No PNG files found for ICO generation. Run generate_favicon_pngs first.")


def generate_og_image():
    """Generate Open Graph image (1200x630)."""
    if not HAS_PIL:
        print("Pillow not installed. Skipping OG image generation.")
        return

    width, height = 1200, 630
    bg_color = (15, 23, 42)  # #0f172a
    purple = (99, 102, 241)  # #6366f1
    green = (34, 197, 94)  # #22c55e
    white = (255, 255, 255)
    gray = (148, 163, 184)  # #94a3b8

    img = Image.new("RGB", (width, height), bg_color)
    draw = ImageDraw.Draw(img)

    # Try to load fonts
    try:
        # macOS system fonts
        title_font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 100)
        subtitle_font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 36)
        small_font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 28)
    except OSError:
        try:
            # Linux system fonts
            title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 100)
            subtitle_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 36)
            small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 28)
        except OSError:
            # Fallback to default
            title_font = ImageFont.load_default()
            subtitle_font = ImageFont.load_default()
            small_font = ImageFont.load_default()
            print("Using default font - install system fonts for better results")

    # Draw the stylized M logo on the right
    logo_x, logo_y = 900, 180
    logo_size = 250
    draw.rounded_rectangle(
        [logo_x, logo_y, logo_x + logo_size, logo_y + logo_size],
        radius=40,
        fill=bg_color,
        outline=purple,
        width=4,
    )

    # Draw verification checkmark circle
    check_x = logo_x + logo_size - 50
    check_y = logo_y + 30
    draw.ellipse([check_x - 25, check_y - 25, check_x + 25, check_y + 25], fill=green)

    # Draw M shape
    m_points = [
        (logo_x + 40, logo_y + logo_size - 50),
        (logo_x + 40, logo_y + 50),
        (logo_x + 85, logo_y + 100),
        (logo_x + 125, logo_y + 50),
        (logo_x + 165, logo_y + 100),
        (logo_x + logo_size - 40, logo_y + 50),
        (logo_x + logo_size - 40, logo_y + logo_size - 50),
    ]
    for i in range(len(m_points) - 1):
        draw.line([m_points[i], m_points[i + 1]], fill=purple, width=12)

    # Draw text
    draw.text((80, 200), "METTLE", fill=white, font=title_font)
    draw.text((80, 320), "Prove Your Metal", fill=purple, font=subtitle_font)
    draw.text((80, 380), "A reverse-CAPTCHA for AI agents", fill=gray, font=small_font)
    draw.text((80, 520), "mettle.sh", fill=gray, font=small_font)

    # Save
    og_path = STATIC_DIR / "og-image.png"
    img.save(og_path, "PNG")
    print("Created og-image.png (1200x630)")


def main():
    """Generate all assets."""
    print("METTLE Asset Generator")
    print("=" * 40)

    os.makedirs(STATIC_DIR, exist_ok=True)

    print("\n1. Generating favicon PNGs...")
    generate_favicon_pngs()

    print("\n2. Generating favicon.ico...")
    generate_favicon_ico()

    print("\n3. Generating Open Graph image...")
    generate_og_image()

    print("\n" + "=" * 40)
    print("Done! Generated assets in", STATIC_DIR)


if __name__ == "__main__":
    main()
