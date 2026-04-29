from dotenv import load_dotenv
from groq import Groq
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.colors import HexColor, white, black
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.units import cm
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import os

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def generate_coaching_report(player_stats, total_frames, video_name):
    """Generate AI coaching report using Groq LLaMA3"""

    # Build stats summary for the AI
    top_players = player_stats[:5]
    stats_text = ""
    for p in top_players:
        stats_text += f"""
        - Player {p['player_id']}: tracked for {p['frames_tracked']} frames, 
          field coverage {p['coverage_percent']}%, 
          average position x={p['avg_x_position']}, y={p['avg_y_position']}
        """

    prompt = f"""
    You are an expert cricket coach and sports analyst. Based on the following 
    player tracking data from a cricket match, write a detailed professional 
    coaching report.

    Match Data:
    - Video: {video_name}
    - Total frames analyzed: {total_frames}
    - Players tracked: {len(player_stats)}
    
    Player Statistics:
    {stats_text}

    Write a coaching report with these exact sections:
    1. MATCH OVERVIEW - Brief summary of the analysis
    2. PLAYER PERFORMANCE ANALYSIS - Analyze each player's movement and coverage
    3. KEY OBSERVATIONS - 3 important tactical observations
    4. FATIGUE INDICATORS - Which players showed signs of fatigue based on coverage
    5. COACHING RECOMMENDATIONS - 3 specific actionable recommendations
    6. CONCLUSION - Brief closing summary

    Write professionally, specifically, and with cricket expertise. 
    Be specific about player IDs and their stats.
    Keep total length around 400 words.
    """

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1000
    )

    return response.choices[0].message.content


def create_pdf_report(player_stats, total_frames, video_name, output_path):
    """Generate a professional PDF coaching report"""

    # Get AI report text
    print("🤖 Generating AI coaching report...")
    report_text = generate_coaching_report(player_stats, total_frames, video_name)

    # --- PDF SETUP ---
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        rightMargin=2*cm,
        leftMargin=2*cm,
        topMargin=2*cm,
        bottomMargin=2*cm
    )

    # --- COLORS ---
    dark_blue  = HexColor("#0F172A")
    mid_blue   = HexColor("#1E3A5F")
    accent     = HexColor("#3B82F6")
    light_gray = HexColor("#F1F5F9")
    text_dark  = HexColor("#1E293B")
    text_gray  = HexColor("#64748B")

    # --- STYLES ---
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "Title",
        fontSize=24,
        fontName="Helvetica-Bold",
        textColor=white,
        alignment=TA_CENTER,
        spaceAfter=6
    )
    subtitle_style = ParagraphStyle(
        "Subtitle",
        fontSize=11,
        fontName="Helvetica",
        textColor=HexColor("#93C5FD"),
        alignment=TA_CENTER,
        spaceAfter=4
    )
    section_style = ParagraphStyle(
        "Section",
        fontSize=13,
        fontName="Helvetica-Bold",
        textColor=accent,
        spaceBefore=14,
        spaceAfter=6
    )
    body_style = ParagraphStyle(
        "Body",
        fontSize=10,
        fontName="Helvetica",
        textColor=text_dark,
        leading=16,
        spaceAfter=6
    )
    meta_style = ParagraphStyle(
        "Meta",
        fontSize=9,
        fontName="Helvetica",
        textColor=text_gray,
        alignment=TA_CENTER
    )

    # --- BUILD PDF ELEMENTS ---
    elements = []

    # Header banner (dark blue background using table trick)
    header_data = [[
        Paragraph("🏏 NeuroCoach", title_style),
    ]]
    header_table = Table(header_data, colWidths=[17*cm])
    header_table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), dark_blue),
        ("ROUNDEDCORNERS", [8, 8, 8, 8]),
        ("TOPPADDING", (0,0), (-1,-1), 20),
        ("BOTTOMPADDING", (0,0), (-1,-1), 10),
        ("LEFTPADDING", (0,0), (-1,-1), 20),
    ]))
    elements.append(header_table)
    elements.append(Spacer(1, 0.3*cm))

    subtitle_data = [[Paragraph("AI Cricket Performance Analysis Report", subtitle_style)]]
    subtitle_table = Table(subtitle_data, colWidths=[17*cm])
    subtitle_table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), mid_blue),
        ("TOPPADDING", (0,0), (-1,-1), 8),
        ("BOTTOMPADDING", (0,0), (-1,-1), 8),
    ]))
    elements.append(subtitle_table)
    elements.append(Spacer(1, 0.5*cm))

    # Match meta info
    elements.append(Paragraph(f"Video: {video_name}   |   Frames Analyzed: {total_frames}   |   Players Tracked: {len(player_stats)}", meta_style))
    elements.append(Spacer(1, 0.4*cm))
    elements.append(HRFlowable(width="100%", thickness=1, color=accent))
    elements.append(Spacer(1, 0.4*cm))

    # Player stats table
    elements.append(Paragraph("Player Tracking Summary", section_style))

    table_data = [["Player ID", "Frames Tracked", "Field Coverage %", "Avg X", "Avg Y"]]
    for p in player_stats[:8]:
        table_data.append([
            f"Player {p['player_id']}",
            str(p['frames_tracked']),
            f"{p['coverage_percent']}%",
            str(p['avg_x_position']),
            str(p['avg_y_position']),
        ])

    stats_table = Table(table_data, colWidths=[3.5*cm, 3.5*cm, 3.5*cm, 3*cm, 3*cm])
    stats_table.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,0),  dark_blue),
        ("TEXTCOLOR",     (0,0), (-1,0),  white),
        ("FONTNAME",      (0,0), (-1,0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0,0), (-1,-1), 9),
        ("ALIGN",         (0,0), (-1,-1), "CENTER"),
        ("ROWBACKGROUNDS",(0,1), (-1,-1), [light_gray, white]),
        ("GRID",          (0,0), (-1,-1), 0.3, HexColor("#CBD5E1")),
        ("TOPPADDING",    (0,0), (-1,-1), 7),
        ("BOTTOMPADDING", (0,0), (-1,-1), 7),
    ]))
    elements.append(stats_table)
    elements.append(Spacer(1, 0.5*cm))
    elements.append(HRFlowable(width="100%", thickness=1, color=HexColor("#E2E8F0")))

    # AI Report sections
    elements.append(Paragraph("AI Coaching Analysis", section_style))

    # Split report into paragraphs and add
    for line in report_text.split("\n"):
        line = line.strip()
        if not line:
            elements.append(Spacer(1, 0.2*cm))
        elif any(line.startswith(f"{i}.") for i in range(1, 7)):
            elements.append(Paragraph(line, section_style))
        elif line.startswith("-"):
            elements.append(Paragraph(f"• {line[1:].strip()}", body_style))
        else:
            try:
                elements.append(Paragraph(line, body_style))
            except Exception:
                pass  # skip any problematic characters

    elements.append(Spacer(1, 0.5*cm))
    elements.append(HRFlowable(width="100%", thickness=1, color=accent))
    elements.append(Spacer(1, 0.3*cm))
    elements.append(Paragraph("Generated by NeuroCoach AI", meta_style))

    # Build PDF
    try:
        doc.build(elements)
        print(f"✅ PDF report saved: {output_path}")
    except Exception as e:
        print(f"❌ PDF build error: {e}")
    
    return output_path