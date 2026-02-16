from configuration import *


class NumberedCanvas(canvas.Canvas):
    """Custom canvas for page numbering and footer"""
    def __init__(self, *args, **kwargs):
        canvas.Canvas.__init__(self, *args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_number(num_pages)
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)

    def draw_page_number(self, page_count):
        self.setFont("Helvetica", 9)
        self.setFillColor(colors.grey)
        page_num = "Page %d of %d" % (self._pageNumber, page_count)
        self.drawRightString(letter[0] - 0.75*inch, 0.5*inch, page_num)
        
        # Add footer line
        self.setStrokeColor(colors.lightgrey)
        self.setLineWidth(0.5)
        self.line(0.75*inch, 0.65*inch, letter[0] - 0.75*inch, 0.65*inch)


class MLReportAgent:
    def __init__(self, llm):
        self.llm = llm
        self.report_date = datetime.datetime.now().strftime("%B %d, %Y at %H:%M")
        self.styles = self._create_styles()

    def _create_styles(self):
        """Create custom paragraph styles for the report"""
        styles = getSampleStyleSheet()
        
        # Title style
        styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#003366'),
            spaceAfter=30,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        ))
        
        # Section Header
        styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=styles['Heading1'],
            fontSize=16,
            textColor=colors.HexColor('#003366'),
            spaceAfter=12,
            spaceBefore=12,
            fontName='Helvetica-Bold',
            borderWidth=0,
            borderColor=colors.HexColor('#003366'),
            borderPadding=5,
            backColor=colors.HexColor('#E6F2FF')
        ))
        
        # Subsection Header
        styles.add(ParagraphStyle(
            name='SubsectionHeader',
            parent=styles['Heading2'],
            fontSize=13,
            textColor=colors.HexColor('#0066CC'),
            spaceAfter=8,
            spaceBefore=10,
            fontName='Helvetica-Bold'
        ))
        
        # Body text
        styles.add(ParagraphStyle(
            name='CustomBody',
            parent=styles['Normal'],
            fontSize=10,
            leading=14,
            alignment=TA_JUSTIFY,
            spaceAfter=6
        ))
        
        # Code/Data style
        styles.add(ParagraphStyle(
            name='CodeStyle',
            parent=styles['Normal'],
            fontSize=9,
            fontName='Courier',
            backColor=colors.HexColor('#F5F5F5'),
            borderWidth=1,
            borderColor=colors.lightgrey,
            borderPadding=5,
            spaceAfter=10
        ))
        
        # Bullet points
        styles.add(ParagraphStyle(
            name='CustomBullet',
            parent=styles['Normal'],
            fontSize=10,
            leftIndent=20,
            spaceAfter=4
        ))
        
        return styles

    def clean_text(self, text):
        """Replaces common Unicode characters that crash standard PDF fonts."""
        if not isinstance(text, str):
            return str(text)
        
        replacements = {
            "\u2014": "-",  # em-dash
            "\u2013": "-",  # en-dash
            "\u201c": '"',  # left double quote
            "\u201d": '"',  # right double quote
            "\u2018": "'",  # left single quote
            "\u2019": "'",  # right single quote
            "\u2022": "*",  # bullet point
            "\u2122": "(TM)",
            "\u00ae": "(R)",
            "\u00a9": "(C)",
        }
        for unicode_char, ascii_char in replacements.items():
            text = text.replace(unicode_char, ascii_char)
        
        return text.encode('latin-1', 'ignore').decode('latin-1')

    def _create_cover_page(self):
        """Create professional cover page"""
        story = []
        story.append(Spacer(1, 1.5*inch))
        title = Paragraph("AUTOMATED MACHINE LEARNING", self.styles['CustomTitle'])
        story.append(title)
        subtitle = Paragraph("ANALYSIS REPORT", self.styles['CustomTitle'])
        story.append(subtitle)
        story.append(Spacer(1, 0.5*inch))
        story.append(HRFlowable(width="80%", thickness=2, color=colors.HexColor('#003366')))
        story.append(Spacer(1, 0.5*inch))
        meta_style = ParagraphStyle('MetaStyle', parent=self.styles['Normal'], fontSize=12, alignment=TA_CENTER, textColor=colors.HexColor('#666666'))
        story.append(Paragraph(f"<b>Generated:</b> {self.report_date}", meta_style))
        story.append(Spacer(1, 0.1*inch))
        story.append(Paragraph("<b>Report Type:</b> Full ML Pipeline Analysis", meta_style))
        story.append(PageBreak())
        return story

    def _create_table_of_contents(self):
        """Create table of contents"""
        story = []
        story.append(Paragraph("TABLE OF CONTENTS", self.styles['SectionHeader']))
        story.append(Spacer(1, 0.2*inch))
        toc_items = [
            ["1.", "Executive Summary", "3"],
            ["2.", "Data Exploratory Analysis", "4"],
            ["3.", "Data Preprocessing & Feature Engineering", "5"],
            ["4.", "Model Selection & Training", "6"],
            ["5.", "Model Performance Evaluation", "7"],
            ["6.", "Visual Analysis", "8"],
            ["7.", "Error Analysis & Insights", "9"],
            ["8.", "Recommendations & Next Steps", "10"],
        ]
        toc_table = Table(toc_items, colWidths=[0.5*inch, 4.5*inch, 1*inch])
        toc_table.setStyle(TableStyle([
            ('FONT', (0, 0), (-1, -1), 'Helvetica', 10),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
            ('ALIGN', (-1, 0), (-1, -1), 'RIGHT'),
            ('LINEBELOW', (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ]))
        story.append(toc_table)
        story.append(PageBreak())
        return story

    def _create_executive_summary(self, ml_results):
        """Create executive summary section"""
        story = []
        story.append(Paragraph("1. EXECUTIVE SUMMARY", self.styles['SectionHeader']))
        story.append(Spacer(1, 0.15*inch))
        summary_text = f"""
        This report presents a comprehensive analysis of an automated machine learning pipeline 
        executed on {self.report_date}. The analysis encompassed data exploration, preprocessing, 
        feature engineering, model selection, and performance evaluation.
        """
        story.append(Paragraph(self.clean_text(summary_text), self.styles['CustomBody']))
        story.append(Spacer(1, 0.1*inch))
        story.append(Paragraph("Key Findings", self.styles['SubsectionHeader']))
        findings_data = [
            ["Metric", "Value"],
            ["Best Model", self.clean_text(ml_results['best_model_name'])],
            ["Model Score", f"{ml_results['best_score']:.4f}"],
            ["Models Evaluated", str(len(ml_results['leaderboard']))],
        ]
        findings_table = Table(findings_data, colWidths=[2.5*inch, 3*inch])
        findings_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#003366')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('TOPPADDING', (0, 1), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 8),
        ]))
        story.append(findings_table)
        story.append(PageBreak())
        return story

    def _create_data_analysis_section(self, data_summary):
        """Create data exploratory analysis section with embedded plots"""
        story = []
        story.append(Paragraph("2. DATA EXPLORATORY ANALYSIS", self.styles['SectionHeader']))
        story.append(Spacer(1, 0.15*inch))
        
        # Handle dict or string input
        if isinstance(data_summary, dict):
            text_content = data_summary.get("text_summary", "No textual summary available.")
            plot_paths = data_summary.get("plot_paths", [])
        else:
            text_content = str(data_summary)
            plot_paths = []

        story.append(Paragraph("2.1 Dataset Overview", self.styles['SubsectionHeader']))
        clean_summary = self.clean_text(text_content)
        story.append(Paragraph(clean_summary, self.styles['CustomBody']))
        story.append(Spacer(1, 0.15*inch))
        
        story.append(Paragraph("2.2 Data Quality Assessment", self.styles['SubsectionHeader']))
        quality_text = """
        The dataset underwent comprehensive quality checks including missing value detection, 
        outlier identification using IQR method (1.5x threshold), and distribution analysis. 
        All identified issues were documented and addressed in the preprocessing phase.
        """
        story.append(Paragraph(quality_text, self.styles['CustomBody']))
        story.append(Spacer(1, 0.15*inch))

        if plot_paths:
            story.append(Paragraph("2.3 Key Data Visualizations", self.styles['SubsectionHeader']))
            intro_plot_text = """
            The following visualizations illustrate key distributions and relationships found 
            within the dataset during the exploratory phase.
            """
            story.append(Paragraph(intro_plot_text, self.styles['CustomBody']))
            story.append(Spacer(1, 0.1*inch))

            for path in plot_paths:
                if os.path.exists(path):
                    try:
                        img = Image(path, width=6*inch, height=3*inch) 
                        story.append(img)
                        story.append(Spacer(1, 0.2*inch))
                    except Exception as e:
                        print(f"Warning: Could not add image {path}. Error: {e}")

        story.append(PageBreak())
        return story

    def _create_preprocessing_section(self, preproc_info):
        """Create preprocessing section"""
        story = []
        story.append(Paragraph("3. DATA PREPROCESSING & FEATURE ENGINEERING", self.styles['SectionHeader']))
        story.append(Spacer(1, 0.15*inch))
        
        story.append(Paragraph("3.1 Feature Categorization", self.styles['SubsectionHeader']))
        num_cols = preproc_info.get('num_cols', [])
        cat_cols = preproc_info.get('cat_cols', [])
        feature_text = f"""
        <b>Numerical Features ({len(num_cols)}):</b><br/>
        {', '.join(str(col) for col in num_cols) if num_cols else 'None identified'}<br/><br/>
        <b>Categorical Features ({len(cat_cols)}):</b><br/>
        {', '.join(str(col) for col in cat_cols) if cat_cols else 'None identified'}
        """
        story.append(Paragraph(self.clean_text(feature_text), self.styles['CustomBody']))
        story.append(Spacer(1, 0.15*inch))
        
        story.append(Paragraph("3.2 Preprocessing Pipeline", self.styles['SubsectionHeader']))
        pipeline_data = [
            ["Step", "Method", "Purpose"],
            ["1. Missing Values", "Median/Mode Imputation", "Handle null values"],
            ["2. Outlier Detection", "IQR Method (1.5x)", "Identify anomalous data points"],
            ["3. Scaling", "StandardScaler", "Normalize numerical features"],
            ["4. Encoding", "One-Hot Encoding", "Convert categorical to numerical"],
            ["5. Feature Selection", "Correlation Analysis", "Remove redundant features"],
        ]
        pipeline_table = Table(pipeline_data, colWidths=[1*inch, 2*inch, 2.5*inch])
        pipeline_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#003366')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('TOPPADDING', (0, 1), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        story.append(pipeline_table)
        story.append(PageBreak())
        return story

    def _create_model_selection_section(self, ml_results):
        """Create model selection and training section"""
        story = []
        story.append(Paragraph("4. MODEL SELECTION & TRAINING", self.styles['SectionHeader']))
        story.append(Spacer(1, 0.15*inch))
        story.append(Paragraph("4.1 Model Architecture", self.styles['SubsectionHeader']))
        model_text = f"""
        <b>Selected Model:</b> {self.clean_text(ml_results['best_model_name'])}<br/><br/>
        <b>Hyperparameters:</b><br/>
        """
        story.append(Paragraph(model_text, self.styles['CustomBody']))
        params = ml_results.get('best_params', {})
        if params:
            params_text = self.clean_text(str(params))
            story.append(Paragraph(params_text, self.styles['CodeStyle']))
        story.append(Spacer(1, 0.15*inch))
        story.append(Paragraph("4.2 Training Configuration", self.styles['SubsectionHeader']))
        config_text = """
        The model was trained using cross-validation with stratified splitting to ensure 
        balanced representation across all classes. Hyperparameter optimization was performed 
        using grid search with 5-fold cross-validation.
        """
        story.append(Paragraph(config_text, self.styles['CustomBody']))
        story.append(PageBreak())
        return story

    def _create_performance_section(self, ml_results):
        """Create model performance section"""
        story = []
        story.append(Paragraph("5. MODEL PERFORMANCE EVALUATION", self.styles['SectionHeader']))
        story.append(Spacer(1, 0.15*inch))
        story.append(Paragraph("5.1 Model Leaderboard", self.styles['SubsectionHeader']))
        leaderboard_text = """
        Multiple machine learning algorithms were evaluated on the dataset. 
        The following table presents the comparative performance:
        """
        story.append(Paragraph(leaderboard_text, self.styles['CustomBody']))
        story.append(Spacer(1, 0.1*inch))
        
        lb_df = ml_results['leaderboard']
        table_data = [['Rank', 'Model', 'Test Score']]
        for idx, row in lb_df.iterrows():
            rank = str(idx + 1)
            model = self.clean_text(str(row['Model']))
            score = f"{row['Score']:.4f}"
            table_data.append([rank, model, score])
        
        lb_table = Table(table_data, colWidths=[0.75*inch, 3.5*inch, 1.25*inch])
        lb_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#003366')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
            ('BACKGROUND', (0, 1), (0, 1), colors.HexColor('#FFD700')),
            ('BACKGROUND', (0, 2), (-1, -1), colors.white),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('TOPPADDING', (0, 1), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 8),
        ]))
        story.append(lb_table)
        story.append(Spacer(1, 0.2*inch))
        story.append(Paragraph("5.2 Performance Insights", self.styles['SubsectionHeader']))
        best_score = ml_results['best_score']
        insights_text = f"""
        The winning model achieved a test score of {best_score:.4f}, demonstrating 
        {'excellent' if best_score > 0.9 else 'strong' if best_score > 0.8 else 'moderate'} 
        performance on the held-out test set. This score indicates the model's ability to 
        generalize to unseen data.
        """
        story.append(Paragraph(insights_text, self.styles['CustomBody']))
        story.append(PageBreak())
        return story

    def _create_visual_analysis_section(self, ml_results):
        """Create visual analysis section with plots (Model Evaluation Plots)"""
        story = []
        story.append(Paragraph("6. VISUAL ANALYSIS", self.styles['SectionHeader']))
        story.append(Spacer(1, 0.15*inch))
        
        # Extract plot paths
        plots = ml_results.get("ml_plots", {})
        
        # 1. Performance Plot
        perf_path = plots.get('performance_plot')
        if perf_path and os.path.exists(perf_path):
            if "confusion" in perf_path:
                title = "6.1 Confusion Matrix"
                desc = "The confusion matrix visualizes the model's prediction accuracy across different classes."
            else:
                title = "6.1 Residual Analysis"
                desc = "The residual plot shows the difference between observed and predicted values."
            
            story.append(Paragraph(title, self.styles['SubsectionHeader']))
            story.append(Paragraph(desc, self.styles['CustomBody']))
            story.append(Spacer(1, 0.1*inch))
            story.append(Image(perf_path, width=6*inch, height=4*inch))
            story.append(Spacer(1, 0.2*inch))

        # 2. Model Comparison Plot
        comp_path = plots.get('comparison_plot')
        if comp_path and os.path.exists(comp_path):
            story.append(Paragraph("6.2 Model Comparison", self.styles['SubsectionHeader']))
            desc = "The following chart compares the performance scores of all candidate models."
            story.append(Paragraph(desc, self.styles['CustomBody']))
            story.append(Spacer(1, 0.1*inch))
            story.append(Image(comp_path, width=6*inch, height=3*inch))
            story.append(Spacer(1, 0.2*inch))
            
        story.append(PageBreak())
        return story

    def _create_error_analysis_section(self, error_analysis):
        """Create detailed error analysis section"""
        story = []
        story.append(Paragraph("7. ERROR ANALYSIS & INSIGHTS", self.styles['SectionHeader']))
        story.append(Spacer(1, 0.15*inch))
        story.append(Paragraph("7.1 Detailed Analysis", self.styles['SubsectionHeader']))
        clean_analysis = self.clean_text(error_analysis)
        paragraphs = clean_analysis.split('\n\n')
        for para in paragraphs:
            if para.strip():
                story.append(Paragraph(para.strip(), self.styles['CustomBody']))
                story.append(Spacer(1, 0.08*inch))
        story.append(PageBreak())
        return story

    def _create_recommendations_section(self):
        """Create recommendations and next steps section (ADDED MISSING METHOD)"""
        story = []
        story.append(Paragraph("8. RECOMMENDATIONS & NEXT STEPS", self.styles['SectionHeader']))
        story.append(Spacer(1, 0.15*inch))
        story.append(Paragraph("8.1 Model Deployment Recommendations", self.styles['SubsectionHeader']))
        
        deploy_text = """
        Based on the analysis results, the following recommendations are provided for 
        model deployment and future improvements:
        """
        story.append(Paragraph(deploy_text, self.styles['CustomBody']))
        story.append(Spacer(1, 0.1*inch))
        
        recommendations = [
            "Monitor model performance in production with regular retraining schedules",
            "Implement A/B testing to validate model improvements",
            "Collect additional data to address identified error patterns",
            "Consider ensemble methods to further improve prediction accuracy",
            "Establish performance thresholds and alerting mechanisms",
            "Document model versioning and maintain audit trails"
        ]
        
        for rec in recommendations:
            bullet = Paragraph(f"• {rec}", self.styles['CustomBullet'])
            story.append(bullet)
        
        story.append(Spacer(1, 0.2*inch))
        story.append(Paragraph("8.2 Future Work", self.styles['SubsectionHeader']))
        
        future_text = """
        Potential areas for future investigation include feature engineering optimization, 
        advanced hyperparameter tuning techniques, and exploration of deep learning approaches 
        if additional computational resources become available.
        """
        story.append(Paragraph(future_text, self.styles['CustomBody']))
        return story

    def generate_pdf_report(self, filename, data_summary, preproc_info, ml_results, error_analysis):
        """Generate comprehensive professional PDF report"""
        doc = SimpleDocTemplate(filename, pagesize=letter, rightMargin=0.75*inch, leftMargin=0.75*inch, topMargin=1*inch, bottomMargin=1*inch)
        story = []
        
        story.extend(self._create_cover_page())
        story.extend(self._create_table_of_contents())
        story.extend(self._create_executive_summary(ml_results))
        story.extend(self._create_data_analysis_section(data_summary))
        story.extend(self._create_preprocessing_section(preproc_info))
        story.extend(self._create_model_selection_section(ml_results))
        story.extend(self._create_performance_section(ml_results))
        story.extend(self._create_visual_analysis_section(ml_results))
        story.extend(self._create_error_analysis_section(error_analysis))
        story.extend(self._create_recommendations_section()) # This will now work
        
        doc.build(story, canvasmaker=NumberedCanvas)
        
        print(f"\n{'='*60}")
        print(f"Professional ML Report Generated Successfully!")
        print(f"Filename: {filename}")
        print(f"{'='*60}\n")
        return filename