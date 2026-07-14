#!/usr/bin/env python3
"""Minimal Markdown -> PDF converter for the benchmark summary.md files.

Handles the subset of Markdown used by analyze-gpu-fryer.py / analyze-nccl-1node.py:
  # / ## headers, bullet lists ("- "), pipe tables, **bold**, and plain paragraphs.
Not a general Markdown renderer -- built for these two report formats.

Usage:
    ./md-to-pdf.py input.md [output.pdf]   # default output: input.pdf next to input.md
"""
import os
import re
import sys

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
CODE_RE = re.compile(r"`([^`]+)`")


def inline(text):
    text = BOLD_RE.sub(r"<b>\1</b>", text)
    text = CODE_RE.sub(r"<font face='Courier'>\1</font>", text)
    return text


def parse_table(lines, i):
    """Consume a pipe-table starting at lines[i]; return (rows, next_index)."""
    rows = []
    while i < len(lines) and lines[i].strip().startswith("|"):
        row = lines[i]
        if re.match(r"^\|[\s:|-]+\|$", row.strip()):
            i += 1
            continue
        cells = [c.strip() for c in row.strip().strip("|").split("|")]
        rows.append(cells)
        i += 1
    return rows, i


def build_story(md_path):
    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("H1", parent=styles["Heading1"], spaceAfter=10)
    h2 = ParagraphStyle("H2", parent=styles["Heading2"], spaceBefore=14, spaceAfter=8)
    body = ParagraphStyle("Body", parent=styles["BodyText"], spaceAfter=6, leading=14)
    bullet = ParagraphStyle("Bullet", parent=body, leftIndent=14, bulletIndent=0)

    story = []
    with open(md_path) as fh:
        lines = fh.read().splitlines()

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if not stripped:
            i += 1
            continue

        if stripped.startswith("### "):
            story.append(Paragraph(inline(stripped[4:]), h2))
            i += 1
        elif stripped.startswith("## "):
            story.append(Paragraph(inline(stripped[3:]), h2))
            i += 1
        elif stripped.startswith("# "):
            story.append(Paragraph(inline(stripped[2:]), h1))
            i += 1
        elif stripped.startswith("|"):
            rows, i = parse_table(lines, i)
            if rows:
                table_data = [[Paragraph(inline(c), body) for c in r] for r in rows]
                col_count = len(rows[0])
                avail_width = 6.5 * inch
                t = Table(table_data, colWidths=[avail_width / col_count] * col_count)
                t.setStyle(
                    TableStyle(
                        [
                            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
                            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f2f2")]),
                            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                            ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                            ("TOPPADDING", (0, 0), (-1, -1), 4),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                        ]
                    )
                )
                story.append(t)
                story.append(Spacer(1, 10))
        elif stripped.startswith("- "):
            story.append(Paragraph("&#8226; " + inline(stripped[2:]), bullet))
            i += 1
        else:
            story.append(Paragraph(inline(stripped), body))
            i += 1

    return story


def main():
    if len(sys.argv) < 2:
        sys.exit("Usage: ./md-to-pdf.py input.md [output.pdf]")
    md_path = sys.argv[1]
    pdf_path = sys.argv[2] if len(sys.argv) > 2 else os.path.splitext(md_path)[0] + ".pdf"

    story = build_story(md_path)
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
    )
    doc.build(story)
    print(f"Written to {pdf_path}")


if __name__ == "__main__":
    main()
