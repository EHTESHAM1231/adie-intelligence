"""
ADIE — Report Generator
Produces both a plain-text report (legacy) and a professional PDF report
using reportlab.  Falls back to text-only if reportlab is not installed.
"""

import datetime
import os

# ── PDF support (optional dependency) ────────────────────────────────────────
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        HRFlowable, KeepTogether
    )
    from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# TEXT REPORT (legacy — kept for backward compatibility)
# ─────────────────────────────────────────────────────────────────────────────

def generate_text_report(diagnostics, expert_report, orig_results, cleaned_results, task_type, selected_algo):
    """
    Generates a structured text report summarizing the dataset diagnostics, expert analysis, and ML results.
    This function compiles all findings into a professional document.
    """
    report = []
    # --- BLOCK 1: REPORT HEADER ---
    # We add the title, project name, and current timestamp to the report.
    report.append("="*70)
    report.append(" PROFESSIONAL DATASET ASSESSMENT & AutoML REPORT")
    report.append(" FYP: Automated Dataset Diagnostics and Repair Framework")
    report.append(" Generated on: " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    report.append("="*70 + "\n")

    # --- BLOCK 2: EXECUTIVE SUMMARY ---
    # We show the high-level details: is the data suitable? what is the quality score?
    report.append("1. EXECUTIVE SUMMARY & SUITABILITY")
    report.append("-" * 40)
    summary = expert_report['summary']
    report.append(f"Version:             {summary['version']}")
    report.append(f"Suitability Status:  {summary['suitability']}")
    report.append(f"Overall Quality Score: {summary['quality_score']}/100")
    report.append(f"Mapped Domain:       {summary['domain']}")
    report.append(f"Primary Industry:    {summary['industry']}")
    report.append(f"Dataset Dimensions:  {summary['rows']} rows x {summary['cols']} columns")
    report.append("\n")

    # --- BLOCK 3: ISSUE IDENTIFICATION (Severity) ---
    # We list all identified issues along with their severity (High/Medium).
    report.append("2. ISSUE IDENTIFICATION & CLASSIFICATION")
    report.append("-" * 40)
    if diagnostics.get('identified_issues'):
        for issue in diagnostics['identified_issues']:
            report.append(f"  [!] {issue['type']:<20} | Severity: {issue['severity']:<10} | Impact Score: {issue['score']:.4f}")
    else:
        report.append("  [+] No critical data quality issues identified.")
    report.append("\n")

    # --- BLOCK 4: INTERVENTION MODULE ---
    # We list the repair strategies we chose and why we chose them.
    report.append("3. INTERVENTION SELECTION MODULE")
    report.append("-" * 40)
    if expert_report.get('interventions'):
        for item in expert_report['interventions']:
            report.append(f"  Issue:     {item['issue']}")
            report.append(f"  Strategy:  {item['strategy']}")
            report.append(f"  Rationale: {item['rationale']}")
            report.append("-" * 20)
    else:
        report.append("  No interventions required.")
    report.append("\n")

    # --- BLOCK 5: SWOT ANALYSIS ---
    # We provide a simple list of the dataset's Strengths and Weaknesses.
    report.append("4. SWOT ANALYSIS (Strengths & Weaknesses)")
    report.append("-" * 40)
    report.append("STRENGTHS:")
    for s in expert_report['swot']['strengths']: report.append(f"  [+] {s}")
    report.append("\nWEAKNESSES:")
    for w in expert_report['swot']['weaknesses']: report.append(f"  [-] {w}")
    report.append("\n")

    # --- BLOCK 6: DETAILED DIAGNOSTICS ---
    # We provide the raw numbers found during analysis (duplicates, noise, missing values).
    report.append("5. DETAILED DATA DIAGNOSTICS")
    report.append("-" * 40)
    report.append(f"Total Duplicates: {diagnostics['duplicates']}")
    report.append(f"Total Missing Values: {diagnostics['missing_values']['total']}")
    report.append(f"Total Outliers: {diagnostics['outliers']['total']}")
    report.append(f"Label Noise (KNN): {diagnostics.get('label_noise', 0)} samples")
    report.append(f"Target Column: {diagnostics['class_imbalance']['target_column']}")
    
    # Add mixed fields information
    if diagnostics.get('mixed_fields'):
        report.append(f"\nMixed Field Inconsistencies: {len(diagnostics['mixed_fields'])} columns")
        for col, info in diagnostics['mixed_fields'].items():
            report.append(f"  - {col}: {info['type']} ({info.get('numeric_count', info.get('date_count', 0))} valid, {info.get('text_count', info.get('non_date_count', 0))} invalid)")
    
    if diagnostics.get('correlations'):
        report.append("\nTop Feature Correlations with Target:")
        for feat, val in diagnostics['correlations'].items():
            report.append(f"  - {feat}: {val}")
    
    if diagnostics.get('leakage_risk'):
        report.append(f"\nPotential Data Leakage Risk: {', '.join(diagnostics['leakage_risk'])}")

    report.append("\nClass Distribution:")
    for cls, count in diagnostics['class_imbalance']['distribution'].items():
        report.append(f"  - Class {cls}: {count}")
    report.append("\n")

    # --- BLOCK 7: PERFORMANCE COMPARISON ---
    # We show the "Before" vs "After" metrics for each model to prove 
    # that our repairs actually improved the results.
    report.append("6. MACHINE LEARNING PERFORMANCE COMPARISON")
    report.append("-" * 40)
    report.append(f"Task Type: {task_type.capitalize()}")
    report.append(f"Selected Algorithm: {selected_algo}")
    
    for model_name, metrics in cleaned_results.items():
        if 'error' in metrics:
            report.append(f"\nModel: {model_name}")
            report.append(f"  ERROR: {metrics['error']}")
        else:
            report.append(f"\nModel: {model_name}")
            orig_metrics = orig_results.get(model_name, {})
            
            if task_type == 'classification':
                report.append(f"  Metric       | Original | Repaired")
                report.append(f"  Accuracy     | {orig_metrics.get('Accuracy', 'N/A'):<8} | {metrics['Accuracy']}")
                report.append(f"  Precision    | {orig_metrics.get('Precision', 'N/A'):<8} | {metrics['Precision']}")
                report.append(f"  Recall       | {orig_metrics.get('Recall', 'N/A'):<8} | {metrics['Recall']}")
                report.append(f"  F1-Score     | {orig_metrics.get('F1-Score', 'N/A'):<8} | {metrics['F1-Score']}")
            else:
                report.append(f"  Metric       | Original | Repaired")
                report.append(f"  R2 Score     | {orig_metrics.get('R2 Score', 'N/A'):<8} | {metrics['R2 Score']}")
                report.append(f"  MAE          | {orig_metrics.get('MAE', 'N/A'):<8} | {metrics.get('MAE', 'N/A')}")
            
            if 'feature_importance' in metrics:
                report.append("  Top Features (Repaired):")
                sorted_features = sorted(metrics['feature_importance'].items(), key=lambda x: x[1], reverse=True)[:5]
                for feat, imp in sorted_features:
                    report.append(f"    - {feat}: {imp}")

    # --- BLOCK 8: FINAL DETERMINATION ---
    # We end with a clear approval or rejection of the dataset for production use.
    report.append("\n" + "="*70)
    report.append(" FINAL ELIGIBILITY DETERMINATION")
    if "HIGHLY SUITABLE" in summary['suitability']:
        report.append(" Dataset is APPROVED for high-stakes production use cases.")
    elif "CONDITIONALLY SUITABLE" in summary['suitability']:
        report.append(" Dataset is ELIGIBLE for experimental use; follow recommendations.")
    else:
        report.append(" Dataset is NOT SUITABLE for current project objectives.")
    report.append("="*70)

    # Join all lines into a single string for the final .txt file
    return "\n".join(report)


# ─────────────────────────────────────────────────────────────────────────────
# PDF REPORT
# ─────────────────────────────────────────────────────────────────────────────

def generate_pdf_report(
    diagnostics, expert_report, orig_results, cleaned_results,
    task_type, selected_algo, output_path,
    eval_results=None
):
    """
    Generate a professional PDF report.
    Falls back to a text file if reportlab is not installed.

    Parameters
    ----------
    output_path : str  — full path where the PDF should be saved
    eval_results: dict | None — system evaluation results (optional)

    Returns
    -------
    str — actual path of the generated file (may be .txt if PDF unavailable)
    """
    if not PDF_AVAILABLE:
        # Graceful fallback
        txt_path = output_path.replace(".pdf", ".txt")
        text = generate_text_report(
            diagnostics, expert_report, orig_results or {},
            cleaned_results or {}, task_type or "classification", selected_algo or "All"
        )
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)
        return txt_path

    # ── Colour palette ────────────────────────────────────────────────────
    C_PRIMARY   = colors.HexColor("#065f46")
    C_ACCENT    = colors.HexColor("#10b981")
    C_DANGER    = colors.HexColor("#991b1b")
    C_WARNING   = colors.HexColor("#92400e")
    C_MUTED     = colors.HexColor("#475569")
    C_LIGHT     = colors.HexColor("#f8fafc")
    C_BORDER    = colors.HexColor("#e2e8f0")
    C_WHITE     = colors.white
    C_BLACK     = colors.HexColor("#1e293b")

    # ── Styles ────────────────────────────────────────────────────────────
    base_styles = getSampleStyleSheet()

    def _style(name, parent="Normal", **kwargs):
        s = ParagraphStyle(name, parent=base_styles[parent], **kwargs)
        return s

    style_title = _style("Title", "Title",
        fontSize=22, textColor=C_WHITE, alignment=TA_LEFT,
        fontName="Helvetica-Bold", spaceAfter=4)
    style_subtitle = _style("Subtitle", "Normal",
        fontSize=10, textColor=colors.HexColor("#cbd5e1"),
        fontName="Helvetica", spaceAfter=2)
    style_h2 = _style("H2", "Heading2",
        fontSize=13, textColor=C_PRIMARY, fontName="Helvetica-Bold",
        spaceBefore=14, spaceAfter=6)
    style_h3 = _style("H3", "Heading3",
        fontSize=10, textColor=C_BLACK, fontName="Helvetica-Bold",
        spaceBefore=8, spaceAfter=4)
    style_body = _style("Body", "Normal",
        fontSize=9, textColor=C_BLACK, fontName="Helvetica",
        leading=14, spaceAfter=4)
    style_muted = _style("Muted", "Normal",
        fontSize=8, textColor=C_MUTED, fontName="Helvetica",
        leading=12, spaceAfter=3)
    style_label = _style("Label", "Normal",
        fontSize=7, textColor=C_MUTED, fontName="Helvetica-Bold",
        spaceAfter=1)
    style_value = _style("Value", "Normal",
        fontSize=14, textColor=C_PRIMARY, fontName="Helvetica-Bold",
        spaceAfter=4)
    style_center = _style("Center", "Normal",
        fontSize=9, textColor=C_BLACK, fontName="Helvetica",
        alignment=TA_CENTER)
    style_badge_green = _style("BadgeGreen", "Normal",
        fontSize=8, textColor=C_WHITE, fontName="Helvetica-Bold",
        backColor=C_ACCENT, alignment=TA_CENTER)
    style_badge_red = _style("BadgeRed", "Normal",
        fontSize=8, textColor=C_WHITE, fontName="Helvetica-Bold",
        backColor=C_DANGER, alignment=TA_CENTER)

    # ── Document ──────────────────────────────────────────────────────────
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm,
        title="ADIE Analysis Report",
        author="ADIE Intelligence Engine"
    )

    story = []
    W = A4[0] - 4*cm   # usable width

    # ── Helper: section divider ───────────────────────────────────────────
    def divider():
        story.append(Spacer(1, 6))
        story.append(HRFlowable(width="100%", thickness=1, color=C_BORDER))
        story.append(Spacer(1, 6))

    def kv_table(rows, col_widths=None):
        """Two-column key-value table."""
        if col_widths is None:
            col_widths = [W * 0.40, W * 0.60]
        data = [[Paragraph(f"<b>{k}</b>", style_muted),
                 Paragraph(str(v), style_body)] for k, v in rows]
        t = Table(data, colWidths=col_widths)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (0, -1), C_LIGHT),
            ("GRID", (0, 0), (-1, -1), 0.5, C_BORDER),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        return t

    # ════════════════════════════════════════════════════════════════════════
    # COVER BLOCK
    # ════════════════════════════════════════════════════════════════════════
    cover_data = [[
        Paragraph("ADIE INTELLIGENCE ENGINE", style_title),
        Paragraph("Professional Dataset Assessment & AutoML Report", style_subtitle),
        Paragraph(
            f"Generated: {datetime.datetime.now().strftime('%d %B %Y  %H:%M')}",
            style_subtitle
        )
    ]]
    cover_table = Table([[cover_data[0][0]], [cover_data[0][1]], [cover_data[0][2]]],
                        colWidths=[W])
    cover_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), C_PRIMARY),
        ("LEFTPADDING", (0, 0), (-1, -1), 16),
        ("RIGHTPADDING", (0, 0), (-1, -1), 16),
        ("TOPPADDING", (0, 0), (0, 0), 16),
        ("BOTTOMPADDING", (0, -1), (-1, -1), 16),
    ]))
    story.append(cover_table)
    story.append(Spacer(1, 20))

    # ════════════════════════════════════════════════════════════════════════
    # 1. EXECUTIVE SUMMARY
    # ════════════════════════════════════════════════════════════════════════
    story.append(Paragraph("1. EXECUTIVE SUMMARY", style_h2))
    divider()

    summary = expert_report.get("summary", {})
    summary_rows = [
        ("Version",          summary.get("version", "N/A")),
        ("Suitability",      summary.get("suitability", "N/A")),
        ("Quality Score",    f"{summary.get('quality_score', 0)}/100"),
        ("Domain",           summary.get("domain", "N/A")),
        ("Industry",         summary.get("industry", "N/A")),
        ("Dataset Size",     f"{summary.get('rows', 0):,} rows × {summary.get('cols', 0)} columns"),
    ]
    story.append(kv_table(summary_rows))
    story.append(Spacer(1, 12))

    # SWOT
    swot = expert_report.get("swot", {})
    strengths = swot.get("strengths", [])
    weaknesses = swot.get("weaknesses", [])

    if strengths or weaknesses:
        story.append(Paragraph("SWOT Analysis", style_h3))
        swot_data = [
            [Paragraph("<b>STRENGTHS</b>", style_muted),
             Paragraph("<b>WEAKNESSES</b>", style_muted)]
        ]
        s_text = "<br/>".join(f"✓ {s}" for s in strengths) if strengths else "None identified"
        w_text = "<br/>".join(f"✗ {w}" for w in weaknesses) if weaknesses else "None identified"
        swot_data.append([
            Paragraph(s_text, style_body),
            Paragraph(w_text, style_body)
        ])
        swot_table = Table(swot_data, colWidths=[W/2, W/2])
        swot_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (0, 0), colors.HexColor("#dcfce7")),
            ("BACKGROUND", (1, 0), (1, 0), colors.HexColor("#fee2e2")),
            ("BACKGROUND", (0, 1), (0, 1), colors.HexColor("#f0fdf4")),
            ("BACKGROUND", (1, 1), (1, 1), colors.HexColor("#fff1f2")),
            ("GRID", (0, 0), (-1, -1), 0.5, C_BORDER),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ]))
        story.append(swot_table)
        story.append(Spacer(1, 12))

    # ════════════════════════════════════════════════════════════════════════
    # 2. ISSUE IDENTIFICATION
    # ════════════════════════════════════════════════════════════════════════
    story.append(Paragraph("2. ISSUE IDENTIFICATION & CLASSIFICATION", style_h2))
    divider()

    issues = diagnostics.get("identified_issues", [])
    if issues:
        issue_data = [[
            Paragraph("<b>Issue Type</b>", style_muted),
            Paragraph("<b>Severity</b>", style_muted),
            Paragraph("<b>Score</b>", style_muted)
        ]]
        for issue in issues:
            sev = issue.get("severity", "Medium")
            sev_style = style_badge_red if sev == "High" else _style(
                "BadgeWarn", "Normal", fontSize=8, textColor=C_WHITE,
                fontName="Helvetica-Bold",
                backColor=colors.HexColor("#d97706"), alignment=TA_CENTER
            )
            issue_data.append([
                Paragraph(issue.get("type", ""), style_body),
                Paragraph(sev, sev_style),
                Paragraph(f"{issue.get('score', 0):.4f}", style_center)
            ])
        issue_table = Table(issue_data, colWidths=[W*0.55, W*0.25, W*0.20])
        issue_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), C_LIGHT),
            ("GRID", (0, 0), (-1, -1), 0.5, C_BORDER),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(issue_table)
    else:
        story.append(Paragraph("✓ No critical data quality issues identified.", style_body))
    story.append(Spacer(1, 12))

    # ════════════════════════════════════════════════════════════════════════
    # 3. INTERVENTION MODULE
    # ════════════════════════════════════════════════════════════════════════
    story.append(Paragraph("3. INTERVENTION SELECTION MODULE", style_h2))
    divider()

    interventions = expert_report.get("interventions", [])
    if interventions:
        int_data = [[
            Paragraph("<b>Issue</b>", style_muted),
            Paragraph("<b>Strategy</b>", style_muted),
            Paragraph("<b>Rationale</b>", style_muted)
        ]]
        for item in interventions:
            int_data.append([
                Paragraph(item.get("issue", ""), style_body),
                Paragraph(item.get("strategy", ""), style_body),
                Paragraph(item.get("rationale", ""), style_muted)
            ])
        int_table = Table(int_data, colWidths=[W*0.22, W*0.28, W*0.50])
        int_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), C_LIGHT),
            ("GRID", (0, 0), (-1, -1), 0.5, C_BORDER),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(int_table)
    else:
        story.append(Paragraph("No interventions required.", style_body))
    story.append(Spacer(1, 12))

    # ════════════════════════════════════════════════════════════════════════
    # 4. DETAILED DIAGNOSTICS
    # ════════════════════════════════════════════════════════════════════════
    story.append(Paragraph("4. DETAILED DATA DIAGNOSTICS", style_h2))
    divider()

    diag_rows = [
        ("Total Missing Values",  diagnostics.get("missing_values", {}).get("total", 0)),
        ("Duplicate Rows",        diagnostics.get("duplicates", 0)),
        ("Total Outliers",        diagnostics.get("outliers", {}).get("total", 0)),
        ("Label Noise (KNN)",     diagnostics.get("label_noise", 0)),
        ("Target Column",         diagnostics.get("class_imbalance", {}).get("target_column", "N/A")),
    ]
    story.append(kv_table(diag_rows))

    # Correlations
    corrs = diagnostics.get("correlations", {})
    if corrs:
        story.append(Spacer(1, 8))
        story.append(Paragraph("Top Feature Correlations with Target", style_h3))
        corr_data = [[Paragraph("<b>Feature</b>", style_muted),
                      Paragraph("<b>Correlation</b>", style_muted)]]
        for feat, val in corrs.items():
            corr_data.append([
                Paragraph(feat, style_body),
                Paragraph(f"{val:.4f}", style_center)
            ])
        corr_table = Table(corr_data, colWidths=[W*0.70, W*0.30])
        corr_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), C_LIGHT),
            ("GRID", (0, 0), (-1, -1), 0.5, C_BORDER),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(corr_table)
    story.append(Spacer(1, 12))

    # ════════════════════════════════════════════════════════════════════════
    # 5. ML PERFORMANCE COMPARISON
    # ════════════════════════════════════════════════════════════════════════
    if cleaned_results:
        story.append(Paragraph("5. MACHINE LEARNING PERFORMANCE COMPARISON", style_h2))
        divider()
        story.append(Paragraph(
            f"Task Type: <b>{(task_type or 'N/A').capitalize()}</b>  |  "
            f"Algorithm: <b>{selected_algo or 'All'}</b>",
            style_body
        ))
        story.append(Spacer(1, 8))

        for model_name, metrics in (cleaned_results or {}).items():
            if "error" in metrics:
                story.append(Paragraph(f"<b>{model_name}</b>: ERROR — {metrics['error']}", style_muted))
                continue

            orig_m = (orig_results or {}).get(model_name, {})
            story.append(Paragraph(f"Model: {model_name}", style_h3))

            if task_type == "classification":
                perf_data = [
                    [Paragraph("<b>Metric</b>", style_muted),
                     Paragraph("<b>Original</b>", style_muted),
                     Paragraph("<b>Repaired</b>", style_muted),
                     Paragraph("<b>Δ Change</b>", style_muted)],
                ]
                for metric in ["Accuracy", "Precision", "Recall", "F1-Score"]:
                    orig_v = orig_m.get(metric, "N/A")
                    rep_v = metrics.get(metric, "N/A")
                    if isinstance(orig_v, float) and isinstance(rep_v, float):
                        delta = rep_v - orig_v
                        delta_str = f"{'▲' if delta >= 0 else '▼'} {abs(delta):.4f}"
                        delta_color = "#166534" if delta >= 0 else "#991b1b"
                    else:
                        delta_str = "N/A"
                        delta_color = "#475569"
                    perf_data.append([
                        Paragraph(metric, style_body),
                        Paragraph(str(orig_v), style_center),
                        Paragraph(str(rep_v), style_center),
                        Paragraph(f'<font color="{delta_color}">{delta_str}</font>', style_center)
                    ])
            else:
                perf_data = [
                    [Paragraph("<b>Metric</b>", style_muted),
                     Paragraph("<b>Original</b>", style_muted),
                     Paragraph("<b>Repaired</b>", style_muted),
                     Paragraph("<b>Δ Change</b>", style_muted)],
                ]
                for metric in ["R2 Score", "MAE", "MSE"]:
                    orig_v = orig_m.get(metric, "N/A")
                    rep_v = metrics.get(metric, "N/A")
                    if isinstance(orig_v, float) and isinstance(rep_v, float):
                        delta = rep_v - orig_v
                        delta_str = f"{'▲' if delta >= 0 else '▼'} {abs(delta):.4f}"
                        delta_color = "#166534" if delta >= 0 else "#991b1b"
                    else:
                        delta_str = "N/A"
                        delta_color = "#475569"
                    perf_data.append([
                        Paragraph(metric, style_body),
                        Paragraph(str(orig_v), style_center),
                        Paragraph(str(rep_v), style_center),
                        Paragraph(f'<font color="{delta_color}">{delta_str}</font>', style_center)
                    ])

            perf_table = Table(perf_data, colWidths=[W*0.30, W*0.23, W*0.23, W*0.24])
            perf_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), C_LIGHT),
                ("GRID", (0, 0), (-1, -1), 0.5, C_BORDER),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]))
            story.append(perf_table)
            story.append(Spacer(1, 8))

    # ════════════════════════════════════════════════════════════════════════
    # 6. SYSTEM EVALUATION (optional)
    # ════════════════════════════════════════════════════════════════════════
    if eval_results:
        story.append(Paragraph("6. SYSTEM SELF-EVALUATION", style_h2))
        divider()

        agg = eval_results.get("aggregate", {})
        eval_rows = [
            ("Total Datasets Evaluated", agg.get("total_datasets", 0)),
            ("Successful Runs",          agg.get("successful", 0)),
            ("Failed Runs",              agg.get("failed", 0)),
            ("Success Rate",             f"{agg.get('success_rate', 0):.0%}"),
            ("Improvement Rate",         f"{agg.get('improvement_rate', 0):.0%}"),
            ("Avg Accuracy Improvement", agg.get("avg_accuracy_improvement", "N/A")),
            ("Avg F1 Improvement",       agg.get("avg_f1_improvement", "N/A")),
            ("Avg Issues Resolved",      agg.get("avg_issues_resolved", 0)),
        ]
        story.append(kv_table(eval_rows))
        story.append(Spacer(1, 12))

    # ════════════════════════════════════════════════════════════════════════
    # FOOTER
    # ════════════════════════════════════════════════════════════════════════
    story.append(Spacer(1, 20))
    story.append(HRFlowable(width="100%", thickness=2, color=C_PRIMARY))
    story.append(Spacer(1, 6))

    suitability = summary.get("suitability", "")
    if "HIGHLY SUITABLE" in suitability:
        verdict = "Dataset is APPROVED for high-stakes production use cases."
        verdict_color = "#065f46"
    elif "CONDITIONALLY" in suitability:
        verdict = "Dataset is ELIGIBLE for experimental use; follow recommendations."
        verdict_color = "#92400e"
    else:
        verdict = "Dataset is NOT SUITABLE for current project objectives."
        verdict_color = "#991b1b"

    story.append(Paragraph(
        f'<font color="{verdict_color}"><b>FINAL DETERMINATION: {verdict}</b></font>',
        style_body
    ))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "Generated by ADIE Intelligence Engine — Automated Dataset Diagnostics & Repair Framework",
        style_muted
    ))

    doc.build(story)
    return output_path
