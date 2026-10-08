import os
import pandas as pd
import numpy as np
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, HRFlowable, PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

def create_pdf():
    pdf_filename = "Polynomial_Regression_Report.pdf"
    doc = SimpleDocTemplate(
        pdf_filename,
        pagesize=letter,
        leftMargin=28,
        rightMargin=28,
        topMargin=24,
        bottomMargin=24
    )

    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=16,
        leading=19,
        textColor=colors.HexColor('#1A365D'),
        spaceAfter=2,
        alignment=1
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9.5,
        leading=12,
        textColor=colors.HexColor('#2B6CB0'),
        spaceAfter=6,
        alignment=1
    )
    
    h1_style = ParagraphStyle(
        'Heading1_Custom',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=10.5,
        leading=13,
        textColor=colors.HexColor('#1A365D'),
        spaceBefore=4,
        spaceAfter=2,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        'Body_Custom',
        parent=styles['BodyText'],
        fontName='Helvetica',
        fontSize=8,
        leading=10.8,
        textColor=colors.HexColor('#2D3748'),
        spaceAfter=3
    )
    
    table_text = ParagraphStyle(
        'TableText',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=6.8,
        leading=8,
        textColor=colors.HexColor('#1A202C')
    )
    
    table_header = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=7.2,
        leading=8.5,
        textColor=colors.white,
        alignment=1
    )

    story = []

    # ================= PAGE 1 =================
    # Title & Header
    story.append(Paragraph("Machine Learning Assignment: Polynomial Regression Analysis", title_style))
    story.append(Paragraph("Lead Energy & Survey Engineering Technical Report | Power Plant Optimization & Reservoir Mapping", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.0, color=colors.HexColor('#1A365D'), spaceAfter=4))

    # Executive Summary & Methodology
    story.append(Paragraph("1. Executive Summary & Mathematical Methodology", h1_style))
    exec_summary_text = (
        "This technical report documents the empirical modeling, polynomial degree selection, cross-validation, and test inference for two continuous target estimation problems:<br/>"
        "• <b>Phase 1 (var1) — Steam Turbine Optimization:</b> Modeling Net Power Score (<i>y</i>) from six operational parameters (<i>x<sub>1</sub>, ..., x<sub>6</sub></i>).<br/>"
        "• <b>Phase 2 (var2) — Subterranean Reservoir Mapping:</b> Modeling Thermal Anomaly Score (<i>y</i>) across a 3D spatial coordinate grid (<i>x<sub>1</sub>, x<sub>2</sub>, x<sub>3</sub></i>).<br/>"
        "Polynomial feature expansion maps input vector <b>x</b> ∈ ℝ<sup><i>D</i></sup> into monomial feature space <b>Φ</b>(<b>x</b>) with <i>N<sub>feats</sub></i> = C(<i>D+d</i>, <i>d</i>). "
        "All features were normalized using <b>StandardScaler</b> and evaluated under <b>10-Fold Cross-Validation</b> comparing OLS and L2 Ridge Regression."
    )
    story.append(Paragraph(exec_summary_text, body_style))

    # Phase 1 Section
    story.append(Paragraph("2. Phase 1: Steam Turbine Optimization (var1) Results", h1_style))
    
    p1_df = pd.read_csv('p1_metrics.csv')
    
    t1_data = [[
        Paragraph("Degree", table_header),
        Paragraph("Terms", table_header),
        Paragraph("OLS Val MSE", table_header),
        Paragraph("OLS Val R²", table_header),
        Paragraph("Ridge Val MSE", table_header),
        Paragraph("Ridge Val R²", table_header)
    ]]
    
    for _, row in p1_df.iterrows():
        deg = int(row['degree'])
        n_t = int(row['features'])
        ols_mse = f"{row['ols_val_mse']:.4f}" if not np.isnan(row['ols_val_mse']) else "N/A"
        ols_r2 = f"{row['ols_val_r2']:.4f}" if not np.isnan(row['ols_val_r2']) else "N/A"
        ridge_mse = f"{row['ridge_val_mse']:.4f}"
        ridge_r2 = f"{row['ridge_val_r2']:.4f}"
        
        t1_data.append([
            Paragraph(f"<b>Degree {deg}</b>" if deg == 5 else f"Degree {deg}", table_text),
            Paragraph(str(n_t), table_text),
            Paragraph(ols_mse, table_text),
            Paragraph(ols_r2, table_text),
            Paragraph(f"<b>{ridge_mse}</b>" if deg == 5 else ridge_mse, table_text),
            Paragraph(f"<b>{ridge_r2}</b>" if deg == 5 else ridge_r2, table_text)
        ])
        
    t1 = Table(t1_data, colWidths=[65, 50, 100, 100, 110, 100])
    t1.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1A365D')),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 1),
        ('BOTTOMPADDING', (0,0), (-1,-1), 1),
        ('GRID', (0,0), (-1,-1), 0.35, colors.HexColor('#CBD5E0')),
        ('BACKGROUND', (0,5), (-1,5), colors.HexColor('#EBF8FF')),
    ]))
    
    story.append(t1)

    p1_discussion = (
        "<b>Degree Selection Rationale (Phase 1):</b> Unregularized OLS peaks at Degree 4 (Val MSE 0.8290, R² 0.9278) but suffers severe coefficient explosion at Degree 6 (924 terms). "
        "Applying <b>Ridge Regression (α = 12.0)</b> regularizes high-order interaction weights, allowing the model to capture degree 5 interaction terms cleanly. "
        "<b>Optimal Choice: Degree 5 Ridge</b> achieves minimum 10-Fold CV MSE of <b>0.4734</b> and maximum R² of <b>0.9583</b>."
    )
    story.append(Paragraph(p1_discussion, body_style))

    # Phase 1 Images
    img1 = Image('p1_degree_vs_error.png', width=3.55*inch, height=1.9*inch)
    img2 = Image('p1_pred_vs_actual.png', width=3.35*inch, height=1.9*inch)
    img_table1 = Table([[img1, img2]], colWidths=[3.6*inch, 3.4*inch])
    img_table1.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('LEFTPADDING', (0,0), (-1,-1), 0),
        ('RIGHTPADDING', (0,0), (-1,-1), 0),
        ('TOPPADDING', (0,0), (-1,-1), 0),
        ('BOTTOMPADDING', (0,0), (-1,-1), 0),
    ]))
    story.append(img_table1)

    # ================= PAGE 2 =================
    story.append(PageBreak())

    # Phase 2 Section
    story.append(Paragraph("3. Phase 2: Subterranean Reservoir Mapping (var2) Results", h1_style))
    
    p2_df = pd.read_csv('p2_metrics.csv')
    
    t2_data = [[
        Paragraph("Degree", table_header),
        Paragraph("Terms", table_header),
        Paragraph("OLS Val MSE", table_header),
        Paragraph("OLS Val R²", table_header),
        Paragraph("Ridge Val MSE", table_header),
        Paragraph("Ridge Val R²", table_header)
    ]]
    
    for _, row in p2_df.iterrows():
        deg = int(row['degree'])
        n_t = int(row['features'])
        ols_mse = f"{row['ols_val_mse']:.4f}" if not np.isnan(row['ols_val_mse']) and row['ols_val_mse'] < 100 else ("N/A" if np.isnan(row['ols_val_mse']) else f"{row['ols_val_mse']:.1f}")
        ols_r2 = f"{row['ols_val_r2']:.4f}" if not np.isnan(row['ols_val_r2']) and row['ols_val_r2'] > -10 else "N/A"
        ridge_mse = f"{row['ridge_val_mse']:.4f}"
        ridge_r2 = f"{row['ridge_val_r2']:.4f}"
        
        t2_data.append([
            Paragraph(f"<b>Degree {deg}</b>" if deg == 11 else f"Degree {deg}", table_text),
            Paragraph(str(n_t), table_text),
            Paragraph(ols_mse, table_text),
            Paragraph(ols_r2, table_text),
            Paragraph(f"<b>{ridge_mse}</b>" if deg == 11 else ridge_mse, table_text),
            Paragraph(f"<b>{ridge_r2}</b>" if deg == 11 else ridge_r2, table_text)
        ])
        
    t2 = Table(t2_data, colWidths=[65, 50, 100, 100, 110, 100])
    t2.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1A365D')),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 0.8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 0.8),
        ('GRID', (0,0), (-1,-1), 0.35, colors.HexColor('#CBD5E0')),
        ('BACKGROUND', (0,11), (-1,11), colors.HexColor('#EBF8FF')),
    ]))
    
    story.append(t2)

    p2_discussion = (
        "<b>Degree Selection Rationale (Phase 2):</b> For 3D spatial coordinate mapping (3 variables), unregularized OLS reaches its error minimum at Degree 8 (Val MSE 0.2454, R² 0.9947) before ill-conditioning degrades performance at higher degrees. "
        "With <b>Ridge Regression (α = 1.0)</b>, model capacity expands smoothly to <b>Degree 11 (364 feature terms)</b>, achieving a outstanding 10-Fold CV MSE of <b>0.2227</b> and R² of <b>0.9951</b>."
    )
    story.append(Paragraph(p2_discussion, body_style))

    # Phase 2 Images
    img3 = Image('p2_degree_vs_error.png', width=3.55*inch, height=1.9*inch)
    img4 = Image('p2_3d_spatial_map.png', width=3.45*inch, height=1.9*inch)
    img_table2 = Table([[img3, img4]], colWidths=[3.6*inch, 3.4*inch])
    img_table2.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('LEFTPADDING', (0,0), (-1,-1), 0),
        ('RIGHTPADDING', (0,0), (-1,-1), 0),
        ('TOPPADDING', (0,0), (-1,-1), 0),
        ('BOTTOMPADDING', (0,0), (-1,-1), 0),
    ]))
    story.append(img_table2)

    # Deliverables Summary
    story.append(Paragraph("4. Summary of Deliverables & Output Files", h1_style))
    summary_text = (
        "• <b>pred_var1.csv:</b> 1000 predicted continuous Net Power Score values (<i>y</i>) using Degree 5 Ridge (α = 12.0).<br/>"
        "• <b>pred_var2.csv:</b> 1000 predicted continuous Thermal Anomaly Score values (<i>y</i>) using Degree 11 Ridge (α = 1.0).<br/>"
        "• <b>train_and_predict.py:</b> Fully reproducible Python pipeline for model training, validation, and inference."
    )
    story.append(Paragraph(summary_text, body_style))

    # Build PDF
    doc.build(story)
    print(f"PDF report updated: {pdf_filename}")

if __name__ == '__main__':
    create_pdf()
