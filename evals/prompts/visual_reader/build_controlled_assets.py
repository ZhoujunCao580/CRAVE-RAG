"""Build deterministic raster assets for Visual Reader boundary cases."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont


WIDTH = 900
HEIGHT = 600
OUT = Path(__file__).parent / "assets" / "controlled"


def font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    names = ["arialbd.ttf" if bold else "arial.ttf", "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"]
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


TITLE = font(38, bold=True)
LABEL = font(28)
SMALL = font(23)
BIG = font(64, bold=True)


def canvas(title: str) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (WIDTH, HEIGHT), "#f8fafc")
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((30, 30, WIDTH - 30, HEIGHT - 30), 24, fill="white", outline="#94a3b8", width=3)
    draw.text((65, 58), title, font=TITLE, fill="#172033")
    return image, draw


def save(image: Image.Image, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    image.save(OUT / name, format="PNG", optimize=True)


def cb01() -> None:
    image, draw = canvas("Quarterly revenue")
    values = [("Q1", 18), ("Q2", 31), ("Q3", 47), ("Q4", 26)]
    draw.line((120, 470, 790, 470), fill="#475569", width=4)
    for index, (quarter, value) in enumerate(values):
        x = 150 + index * 160
        height = value * 7
        draw.rounded_rectangle((x, 470 - height, x + 90, 470), 8, fill="#4f7cff")
        draw.text((x + 22, 485), quarter, font=LABEL, fill="#172033")
        draw.text((x + 23, 430 - height), str(value), font=LABEL, fill="#172033")
    save(image, "cb01_many_facts_one_target.png")


def cb02() -> None:
    image, draw = canvas("Model A accuracy: 92%")
    draw.rounded_rectangle((300, 190, 600, 455), 16, fill="#4f7cff")
    draw.text((388, 290), "92%", font=BIG, fill="white")
    draw.text((225, 500), "Legend: Model A — 92% accuracy", font=LABEL, fill="#172033")
    save(image, "cb02_duplicate_presentation.png")


def cb03() -> None:
    image, draw = canvas("Right-side ports")
    rows = [("1", "USB-C"), ("2", "HDMI"), ("3", "3.5 mm")]
    y = 165
    for number, port in rows:
        draw.rounded_rectangle((170, y, 730, y + 90), 15, fill="#eaf0ff", outline="#4f7cff", width=3)
        draw.text((220, y + 24), number, font=LABEL, fill="#2762d9")
        draw.text((340, y + 24), port, font=LABEL, fill="#172033")
        y += 110
    save(image, "cb03_exhaustive_list.png")


def cb04() -> None:
    image, draw = canvas("Annual total")
    draw.text((210, 205), "FY2024 audited result", font=LABEL, fill="#475569")
    patch = Image.new("RGB", (520, 150), "white")
    patch_draw = ImageDraw.Draw(patch)
    patch_draw.text((20, 35), "$18.7 million", font=BIG, fill="#172033")
    patch = patch.filter(ImageFilter.GaussianBlur(radius=8))
    image.paste(patch, (185, 300))
    save(image, "cb04_unreadable_target.png")


def cb05() -> None:
    image, draw = canvas("Top contributors")
    draw.text((180, 175), "1. North — 41", font=LABEL, fill="#172033")
    draw.text((180, 265), "2. West — 35", font=LABEL, fill="#172033")
    patch = Image.new("RGB", (550, 75), "white")
    ImageDraw.Draw(patch).text((0, 12), "3. Central — 2?", font=LABEL, fill="#172033")
    patch = patch.filter(ImageFilter.GaussianBlur(radius=5))
    image.paste(patch, (180, 355))
    save(image, "cb05_partially_readable_list.png")


def cb06() -> None:
    image, draw = canvas("Device specifications")
    draw.text((160, 185), "Weight: 1.24 kg", font=LABEL, fill="#172033")
    draw.text((160, 275), "Battery: 18 hours", font=LABEL, fill="#172033")
    draw.rounded_rectangle((760, 380, 1030, 495), 14, fill="#eaf0ff", outline="#4f7cff", width=3)
    draw.text((825, 415), "Price: $799", font=LABEL, fill="#172033")
    save(image, "cb06_cropped_target.png")


def simple_panel(title: str, lines: list[str], name: str) -> None:
    image, draw = canvas(title)
    y = 200
    for line in lines:
        draw.text((155, y), line, font=LABEL, fill="#172033")
        y += 95
    save(image, name)


def cb07() -> None:
    simple_panel("Sales report", ["2023 sales: $12 million", "Region: Global"], "cb07_i1_sales.png")
    simple_panel("Weather summary", ["Rainfall: 84 mm", "Average temperature: 18 C"], "cb07_i2_weather.png")


def cb08() -> None:
    simple_panel("Revenue 2022", ["Revenue: $8 million"], "cb08_i1_2022.png")
    simple_panel("Revenue 2023", ["Revenue: $11 million"], "cb08_i2_2023.png")


def cb09() -> None:
    simple_panel("Evaluation report A", ["Model Z score: 72"], "cb09_i1_score_72.png")
    simple_panel("Evaluation report B", ["Model Z score: 79"], "cb09_i2_score_79.png")


def cb10() -> None:
    image, draw = canvas("Model summary")
    headers = ["Model", "Accuracy", "Latency"]
    rows = [["A", "91%", "23 ms"], ["B", "89%", "18 ms"]]
    x0, y0, col, row = 105, 180, 230, 90
    for c, value in enumerate(headers):
        draw.rectangle((x0 + c * col, y0, x0 + (c + 1) * col, y0 + row), fill="#eaf0ff", outline="#64748b", width=2)
        draw.text((x0 + c * col + 35, y0 + 25), value, font=SMALL, fill="#172033")
    for r, values in enumerate(rows, start=1):
        for c, value in enumerate(values):
            draw.rectangle((x0 + c * col, y0 + r * row, x0 + (c + 1) * col, y0 + (r + 1) * row), fill="white", outline="#64748b", width=2)
            draw.text((x0 + c * col + 55, y0 + r * row + 25), value, font=SMALL, fill="#172033")
    save(image, "cb10_requested_fact_absent.png")


def main() -> None:
    cb01()
    cb02()
    cb03()
    cb04()
    cb05()
    cb06()
    cb07()
    cb08()
    cb09()
    cb10()


if __name__ == "__main__":
    main()
