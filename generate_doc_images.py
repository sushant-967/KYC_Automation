"""Generate synthetic KYC document images for all personas."""
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

OUT = Path(__file__).parent / "server" / "uploads"
OUT.mkdir(parents=True, exist_ok=True)


def make_font(size):
    for name in ["/System/Library/Fonts/Helvetica.ttc",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                 "/usr/share/fonts/dejavu/DejaVuSans.ttf"]:
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            pass
    return ImageFont.load_default()


def new_doc(w=800, h=500, bg="#FFFFFF"):
    img = Image.new("RGB", (w, h), bg)
    return img, ImageDraw.Draw(img)


def header(d, text, color, y=20, font_size=28):
    d.text((30, y), text, fill=color, font=make_font(font_size))


def field(d, label, value, y, label_color="#555555", value_color="#111111"):
    d.text((30, y), label + ":", fill=label_color, font=make_font(16))
    d.text((240, y), value, fill=value_color, font=make_font(16))


def border(d, w, h, color):
    d.rectangle([(5, 5), (w - 5, h - 5)], outline=color, width=3)


# ── PRIYA AADHAAR ────────────────────────────────────────────────────────────
img, d = new_doc(800, 450, "#FFF8F0")
border(d, 800, 450, "#E87722")
header(d, "AADHAAR CARD", "#E87722", y=20, font_size=26)
header(d, "Government of India — Unique Identification Authority", "#888888", y=54, font_size=14)
d.line([(30, 78), (770, 78)], fill="#E87722", width=2)
field(d, "Name",           "PRIYA SHARMA",                              y=100)
field(d, "Date of Birth",  "14/07/1992",                                y=130)
field(d, "Gender",         "FEMALE",                                    y=160)
field(d, "Address",        "4th Block, Koramangala,",                   y=190)
field(d, "",               "Bengaluru, Karnataka 560034",               y=215)
field(d, "Aadhaar No.",    "8765 4321 2345",                            y=260)
d.text((30, 340), "Digitally issued. Verify at uidai.gov.in", fill="#AAAAAA", font=make_font(12))
img.save(OUT / "priya-aadhaar.png")

# ── PRIYA PAN ────────────────────────────────────────────────────────────────
img, d = new_doc(800, 450, "#FFFBF0")
border(d, 800, 450, "#1A3A6B")
header(d, "INCOME TAX DEPARTMENT — GOVT. OF INDIA", "#1A3A6B", y=20, font_size=20)
header(d, "PERMANENT ACCOUNT NUMBER CARD", "#333333", y=50, font_size=16)
d.line([(30, 74), (770, 74)], fill="#1A3A6B", width=2)
field(d, "PAN",            "ABYPS1234K",                                y=100)
field(d, "Name",           "PRIYA SHARMA",                              y=130)
field(d, "Father's Name",  "MAHESH SHARMA",                             y=160)
field(d, "Date of Birth",  "14/07/1992",                                y=190)
field(d, "Status",         "Individual",                                y=220)
d.text((30, 340), "This card is the property of Income Tax Department, Govt. of India", fill="#AAAAAA", font=make_font(12))
img.save(OUT / "priya-pan.png")

# ── PRIYA UTILITY BILL ───────────────────────────────────────────────────────
img, d = new_doc(800, 500, "#F0F8FF")
border(d, 800, 500, "#2563EB")
header(d, "BESCOM — ELECTRICITY BILL", "#2563EB", y=20, font_size=24)
header(d, "Bangalore Electricity Supply Company Limited", "#888888", y=54, font_size=14)
d.line([(30, 78), (770, 78)], fill="#2563EB", width=2)
field(d, "Consumer Name",  "PRIYA SHARMA",                              y=100)
field(d, "Address",        "4th Block, Koramangala,",                   y=130)
field(d, "",               "Bengaluru, Karnataka 560034",               y=155)
field(d, "Account No.",    "BES-KRM-00421987",                          y=185)
field(d, "Bill Month",     "May 2026",                                  y=215)
field(d, "Units Consumed", "210 kWh",                                   y=245)
field(d, "Amount Due",     "INR 1,890",                                 y=275)
field(d, "Due Date",       "15/06/2026",                                y=305)
img.save(OUT / "priya-utility-bill.png")

# ── RAJESH AADHAAR ───────────────────────────────────────────────────────────
img, d = new_doc(800, 450, "#FFF8F0")
border(d, 800, 450, "#E87722")
header(d, "AADHAAR CARD", "#E87722", y=20, font_size=26)
header(d, "Government of India — Unique Identification Authority", "#888888", y=54, font_size=14)
d.line([(30, 78), (770, 78)], fill="#E87722", width=2)
field(d, "Name",           "RAJESH KUMAR SINGH",                        y=100)
field(d, "Date of Birth",  "22/03/1971",                                y=130)
field(d, "Gender",         "MALE",                                      y=160)
field(d, "Address",        "Civil Lines,",                              y=190)
field(d, "",               "Lucknow, Uttar Pradesh 226001",             y=215)
field(d, "Aadhaar No.",    "1234 5678 9012",                            y=260)
img.save(OUT / "rajesh-aadhaar.png")

# ── RAJESH PAN ───────────────────────────────────────────────────────────────
img, d = new_doc(800, 450, "#FFFBF0")
border(d, 800, 450, "#1A3A6B")
header(d, "INCOME TAX DEPARTMENT — GOVT. OF INDIA", "#1A3A6B", y=20, font_size=20)
header(d, "PERMANENT ACCOUNT NUMBER CARD", "#333333", y=50, font_size=16)
d.line([(30, 74), (770, 74)], fill="#1A3A6B", width=2)
field(d, "PAN",            "BXRKS5678J",                                y=100)
field(d, "Name",           "RAJESH KUMAR SINGH",                        y=130)
field(d, "Father's Name",  "SURENDRA SINGH",                            y=160)
field(d, "Date of Birth",  "22/03/1971",                                y=190)
field(d, "Status",         "Individual",                                y=220)
img.save(OUT / "rajesh-pan.png")

# ── VIKTOR PASSPORT ──────────────────────────────────────────────────────────
img, d = new_doc(800, 500, "#F5F5F5")
border(d, 800, 500, "#1B3A6B")
header(d, "REPUBLIC OF CYPRUS — PASSPORT", "#1B3A6B", y=20, font_size=24)
header(d, "ΚΥΠΡΙΑΚΗ ΔΗΜΟΚΡΑΤΙΑ", "#666666", y=54, font_size=14)
d.line([(30, 78), (770, 78)], fill="#1B3A6B", width=2)
field(d, "Surname",        "NAZAROV",                                   y=100)
field(d, "Given Names",    "VIKTOR",                                    y=130)
field(d, "Nationality",    "CYPRIOT",                                   y=160)
field(d, "Date of Birth",  "02/04/1979",                                y=190)
field(d, "Place of Birth", "LIMASSOL",                                  y=220)
field(d, "Passport No.",   "K12837654",                                 y=250)
field(d, "Issue Date",     "10/01/2022",                                y=280)
field(d, "Expiry Date",    "09/01/2032",                                y=310)
d.text((30, 380), "P<CYPNAZAROV<<VIKTOR<<<<<<<<<<<<<<<<<<<<<<<<<", fill="#111111", font=make_font(15))
d.text((30, 405), "K128376549CYP7904024M3201091<<<<<<<<<<<<<<08", fill="#111111", font=make_font(15))
img.save(OUT / "viktor-passport.png")

print("Generated documents:")
for f in sorted(OUT.iterdir()):
    print(f"  {f.name}  ({f.stat().st_size // 1024} KB)")
