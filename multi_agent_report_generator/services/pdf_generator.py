import re
import logging
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def clean_text(content: str) -> str:
    """Clean markdown formatting and special characters from text.
    
    Removes markdown formatting markers (bold, italic), standalone asterisks,
    and normalizes whitespace to prepare text for PDF generation.
    
    Args:
        content: Raw text content that may contain markdown formatting.
        
    Returns:
        Cleaned text without markdown markers and normalized whitespace.
        
    Example:
        >>> clean_text("**Bold** and *italic* text")
        'Bold and italic text'
        >>> clean_text("Text with  *  asterisks")
        'Text with asterisks'
    """
    if not content:
        return ""
    
    # Remove markdown bold/italic markers (**text** or *text*)
    content = re.sub(r'\*\*([^*]+)\*\*', r'\1', content)  # Remove **bold**
    content = re.sub(r'\*([^*]+)\*', r'\1', content)  # Remove *italic*
    content = re.sub(r'\*\s+', '', content)  # Remove standalone asterisks with spaces
    
    # Replace multiple asterisks with empty string
    content = re.sub(r'\*+', '', content)
    
    # Clean up multiple spaces
    content = re.sub(r' +', ' ', content)
    
    # Clean up multiple newlines
    content = re.sub(r'\n{3,}', '\n\n', content)
    
    return content.strip()


def save_report_to_pdf(text: str, file_path: str) -> str:
    """Save text content to a PDF file with proper formatting.
    
    Creates a professionally formatted PDF document from text content,
    cleaning markdown formatting and applying appropriate styles for
    headings and body text.
    
    Args:
        text: Text content to convert to PDF (may contain markdown).
        file_path: Path where the PDF file should be saved.
        
    Returns:
        Path to the saved PDF file.
        
    Raises:
        ImportError: If reportlab package is not installed.
        OSError: If the output directory cannot be created or file cannot be written.
        
    Example:
        >>> text = "# Title\\n\\nThis is a paragraph."
        >>> pdf_path = save_report_to_pdf(text, "output/report.pdf")
        >>> os.path.exists(pdf_path)
        True
    """
    try:
        from reportlab.lib.pagesizes import LETTER  # type: ignore[reportMissingImports]
        from reportlab.lib.units import inch  # type: ignore[reportMissingImports]
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer  # type: ignore[reportMissingImports]
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle  # type: ignore[reportMissingImports]
        from reportlab.lib.enums import TA_LEFT, TA_JUSTIFY  # type: ignore[reportMissingImports]
    except ImportError as e:
        error_msg = "reportlab is required to export PDF. Install with: pip install reportlab"
        logger.error(f"{error_msg}: {e}")
        raise ImportError(error_msg) from e

    try:
        output_dir = os.path.dirname(file_path) or "."
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Creating PDF at: {file_path}")
        
        # Clean the text content
        cleaned_text = clean_text(text)

        # Create the PDF document
        doc = SimpleDocTemplate(
            file_path,
            pagesize=LETTER,
            rightMargin=0.75 * inch,
            leftMargin=0.75 * inch,
            topMargin=0.75 * inch,
            bottomMargin=0.75 * inch
        )

        # Define styles
        styles = getSampleStyleSheet()
        
        # Custom styles for better formatting
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=16,
            textColor='black',
            spaceAfter=12,
            alignment=TA_LEFT,
            fontName='Helvetica-Bold'
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=14,
            textColor='black',
            spaceAfter=10,
            spaceBefore=10,
            alignment=TA_LEFT,
            fontName='Helvetica-Bold'
        )
        
        body_style = ParagraphStyle(
            'CustomBody',
            parent=styles['Normal'],
            fontSize=11,
            textColor='black',
            spaceAfter=8,
            alignment=TA_JUSTIFY,
            fontName='Times-Roman',
            leading=14
        )

        # Build the story (content)
        story = []
        
        # Split text into paragraphs and process
        paragraphs = cleaned_text.split('\n\n')
        
        for para in paragraphs:
            if not para.strip():
                story.append(Spacer(1, 0.2 * inch))
                continue
            
            para = para.strip()
            
            # Check if it's a heading (all caps or starts with #)
            if para.upper() == para and len(para) < 100 and para.isupper():
                story.append(Paragraph(para, heading_style))
            elif para.startswith('#'):
                # Markdown heading
                level = len(para) - len(para.lstrip('#'))
                para_text = para.lstrip('#').strip()
                if level == 1:
                    story.append(Paragraph(para_text, title_style))
                else:
                    story.append(Paragraph(para_text, heading_style))
            else:
                # Regular paragraph - escape special characters for ReportLab
                para = para.replace('&', '&amp;')
                para = para.replace('<', '&lt;')
                para = para.replace('>', '&gt;')
                story.append(Paragraph(para, body_style))
            
            story.append(Spacer(1, 0.15 * inch))

        # Build the PDF
        doc.build(story)
        logger.info(f"PDF successfully created at: {file_path}")
        return file_path
        
    except OSError as e:
        error_msg = f"Failed to create output directory or write PDF file: {e}"
        logger.error(error_msg)
        raise OSError(error_msg) from e
    except Exception as e:
        error_msg = f"Unexpected error while generating PDF: {e}"
        logger.error(error_msg, exc_info=True)
        raise RuntimeError(error_msg) from e

