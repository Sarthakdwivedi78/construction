from io import BytesIO
from xml.sax.saxutils import escape

def pdf(report):
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, KeepTogether

    buffer = BytesIO()
    styles = getSampleStyleSheet()
    styles["Title"].textColor = colors.HexColor("#14344A")
    story = []
    def add(text, style="BodyText"):
        story.append(Paragraph(escape(str(text)), styles[style]))
    add("Construction Site Safety Inspection", "Title")
    add("Inspection " + report["inspection_id"])
    add(report["site_type"] + " | " + report["overall_risk_level"])
    add(f'{report["frames_processed"]} frame(s) reviewed. Human review required.')
    estimate = report.get("visual_risk_estimate_percent")
    add("Highest visual risk-cue score: " + (f"{estimate:.1f}%" if estimate is not None else "not available"))
    add(report.get("risk_probability_note", "Detector scores are not calibrated incident probabilities."))
    for warning in report["warnings"]:
        add(warning)
    add("Hazard Findings", "Heading1")
    if not report["identified_hazards"]:
        add("No hazard candidates were recorded in the reviewed frames. Coverage is undetermined.")
    for i, hazard in enumerate(report["identified_hazards"], 1):
        heading = f'{i}. {hazard["hazard"]}'
        start = len(story)
        add(heading, "Heading2")
        add(f'{hazard["timestamp_seconds"]:.2f} s | Risk type: {hazard.get("risk_type", "Site safety")} | {hazard["severity"]} priority | {hazard.get("assessment_display_status", hazard["violation_status"])}')
        if hazard.get("potential_harm"):
            add("Potential harm: " + hazard["potential_harm"])
        add("Observation: " + hazard["visual_evidence"])
        add("Review: " + hazard["reasoning"])
        if hazard["source_url"]:
            add(hazard["indian_regulation"] + " | " + hazard["provision_reference"])
            add(hazard["retrieved_evidence"])
            add(hazard["source_url"])
        add("Action: " + hazard["corrective_action"])
        for measure in hazard.get("preventive_measures", []):
            add("Preventive measure: " + measure)
        story[start:] = [KeepTogether(story[start:])]
    def footer(canvas, doc):
        canvas.setFillColor(colors.HexColor("#526476"))
        canvas.setFont("Helvetica", 9)
        canvas.drawString(46, 28, "Indian construction safety screening")
        canvas.drawRightString(A4[0] - 46, 28, str(doc.page))
    SimpleDocTemplate(buffer, pagesize=A4, leftMargin=46, rightMargin=46,
                      topMargin=46, bottomMargin=46).build(story, onFirstPage=footer, onLaterPages=footer)
    return buffer.getvalue()
