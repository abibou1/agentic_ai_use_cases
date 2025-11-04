"""Multi-agent report generator using CrewAI.

This module implements a multi-agent system for generating research reports
on various topics using web research, trend analysis, report writing, and proofreading agents.
"""

from crewai import Crew, Task, Agent, Process, LLM  # pyright: ignore[reportMissingImports]
from langchain_community.tools import BraveSearch

import logging
import os
import re
from pathlib import Path
from typing import Dict, Any, Optional

from email_sender import send_email_with_attachment

from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()
receiver_email: Optional[str] = os.getenv('RECEIVER_EMAIL')

def ensure_openai_api_key() -> str:
    """Ensure OpenAI API key is set in environment variables.
    
    Loads environment variables from .env file and checks for OPENAI_API_KEY.
    
    Returns:
        str: The OpenAI API key if found.
        
    Raises:
        ValueError: If OPENAI_API_KEY is not set in environment variables.
        
    Example:
        >>> api_key = ensure_openai_api_key()
        >>> isinstance(api_key, str)
        True
    """
    load_dotenv()
    openai_api_key: Optional[str] = os.getenv('OPENAI_API_KEY')
    
    if not openai_api_key:
        error_msg = "OPENAI_API_KEY is not set in environment variables"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    logger.info("OpenAI API key found in environment variables")
    return openai_api_key


def create_llm() -> LLM:  # type: ignore[name-defined]
    """Create and configure the LLM instance.
    
    Creates an LLM instance using OpenAI's GPT-4o model with the API key
    from environment variables.
    
    Returns:
        LLM: Configured LLM instance for use by agents.
        
    Raises:
        ValueError: If OpenAI API key is not available.
        
    Example:
        >>> llm = create_llm()
        >>> llm is not None
        True
    """
    try:
        api_key = ensure_openai_api_key()
        llm = LLM(
            model='openai/gpt-4o',
            api_key=api_key
        )
        logger.info("LLM instance created successfully with model: openai/gpt-4o")
        return llm
    except ValueError as e:
        logger.error(f"Failed to create LLM instance: {e}")
        raise


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


def brave_search_wrapper(query: str) -> str:
    """Wrapper function for BraveSearch tool.
    
    Executes a web search using BraveSearch API and returns relevant results.
    
    Args:
        query: Search query string (must be non-empty).
        
    Returns:
        Search results as a string.
        
    Raises:
        ValueError: If query is not a non-empty string.
        RuntimeError: If BraveSearch API call fails.
        
    Example:
        >>> result = brave_search_wrapper("Python programming")
        >>> isinstance(result, str)
        True
    """
    if not isinstance(query, str) or not query.strip():
        error_msg = "Query must be a non-empty string"
        logger.error(error_msg)
        raise ValueError(error_msg)

    try:
        brave_api_key: str = os.getenv("BRAVE_API_KEY", "BRAVE-API-KEY")
        
        if brave_api_key == "BRAVE-API-KEY":
            logger.warning("Using default BRAVE_API_KEY value. Set BRAVE_API_KEY in environment for actual searches.")
        
        brave_search = BraveSearch.from_api_key(
            api_key=brave_api_key,
            search_kwargs={"count": 3}
        )
        
        logger.info(f"Executing BraveSearch query: {query}")
        result = brave_search.run(query)
        logger.info(f"BraveSearch completed for query: {query}")
        return result
        
    except Exception as e:
        error_msg = f"BraveSearch API call failed: {e}"
        logger.error(error_msg, exc_info=True)
        raise RuntimeError(error_msg) from e


def create_brave_search_tool() -> Dict[str, Any]:
    """Create a tool dictionary for BraveSearch integration.
    
    Returns a dictionary containing tool metadata and function reference
    for use with CrewAI agents.
    
    Returns:
        Dictionary with tool name, description, and function reference.
        
    Example:
        >>> tool = create_brave_search_tool()
        >>> tool['name']
        'brave_search_tool'
        >>> callable(tool['function'])
        True
    """
    return {
        "name": "brave_search_tool",
        "description": (
            "Searches the web using BraveSearch and returns relevant information for a given query. "
            "Useful for finding up-to-date and accurate information on a wide range of topics."
        ),
        "function": brave_search_wrapper,
    }


def main() -> None:
    """Main execution function for the report generation pipeline.
    
    Initializes agents, creates tasks, executes the crew workflow,
    and saves the final report to a PDF file.
    
    Raises:
        ValueError: If required API keys are not configured.
        RuntimeError: If report generation or PDF creation fails.
    """
    try:
        # Initialize LLM
        llm = create_llm()
        
        # Create the BraveSearch tool
        search_tool = create_brave_search_tool()
        
        # Define agents
        web_researcher_agent = Agent(
            name="Web Researcher",
            role="Web Research Specialist",
            goal=(
                "To find the most recent, impactful, and relevant information about {topic}. "
                "This includes identifying key use cases, challenges, and statistics to provide "
                "a foundation for deeper analysis."
            ),
            backstory=(
                "You are a former investigative journalist known for your ability to uncover "
                "technology breakthroughs and market insights. With years of experience, you "
                "excel at identifying actionable data and trends."
            ),
            tools=[],
            llm=llm,
            verbose=True
        )

        trend_analyst_agent = Agent(
            name="Trend Analyst",
            role="Insight Synthesizer",
            goal=(
                "To analyze research findings, extract significant trends, and rank them by "
                "industry impact, growth potential, and uniqueness. Provide actionable insights "
                "for decision-makers."
            ),
            backstory=(
                "You are a seasoned strategy consultant who transitioned into {topic} analysis. "
                "With an eye for patterns, you specialize in translating raw data into clear, "
                "actionable insights."
            ),
            tools=[],
            llm=llm,
            verbose=True
        )

        report_writer_agent = Agent(
            name="Report Writer",
            role="Narrative Architect",
            goal=(
                "To craft a detailed, professional report that communicates research findings "
                "and analysis effectively. Focus on clarity, logical flow, and engagement."
            ),
            backstory=(
                "Once a technical writer for a renowned journal, you are now dedicated to "
                "creating industry-leading reports. You blend storytelling with data to ensure "
                "your work is both informative and captivating."
            ),
            tools=[],
            llm=llm,
            verbose=True
        )

        proofreader_agent = Agent(
            name="Proofreader",
            role="Polisher of Excellence",
            goal=(
                "To refine the report for grammatical accuracy, readability, and formatting, "
                "ensuring it meets professional publication standards."
            ),
            backstory=(
                "An award-winning editor turned proofreader, you specialize in perfecting "
                "written content. Your sharp eye for detail ensures every document is flawless."
            ),
            tools=[],
            llm=llm,
            verbose=True
        )

        manager_agent = Agent(
            name="Manager",
            role="Workflow Maestro",
            goal=(
                "To coordinate agents, manage task dependencies, and ensure all outputs meet "
                "quality standards. Your focus is on delivering a cohesive final product through "
                "efficient task management."
            ),
            backstory=(
                "A former project manager with a passion for efficient teamwork, you ensure "
                "every process runs smoothly, overseeing tasks and verifying results."
            ),
            tools=[],
            llm=llm,
            verbose=True,
            allow_delegation=True
        )

        # Define tasks
        web_research_task = Task(
            description=(
                "Conduct web-based research to identify 5-7 key insights about {topic}. "
                "Focus on key use cases, recent developments, and significant trends."
            ),
            expected_output="A structured list of 5-7 key insights about {topic}."
        )

        trend_analysis_task = Task(
            description="Analyze the research findings to rank trends by importance and impact.",
            expected_output=(
                "A table ranking trends by impact, with concise descriptions of each trend."
            )
        )

        report_writing_task = Task(
            description=(
                "Draft report summarizing the findings and analysis of {topic}. Include sections "
                "for Introduction, Trends Overview, Analysis, and Recommendations."
            ),
            expected_output=(
                "A structured, professional draft with a clear flow of information. Ensure "
                "logical organization and consistent tone."
            )
        )

        proofreading_task = Task(
            description=(
                "Refine the draft for grammatical accuracy, coherence, and formatting. Ensure "
                "the final document is polished and ready for publication."
            ),
            expected_output=(
                "A professional, polished report free of grammatical errors and inconsistencies. "
                "Format the document for easy readability."
            )
        )

        # Create crew
        crew = Crew(
            agents=[
                web_researcher_agent,
                trend_analyst_agent,
                report_writer_agent,
                proofreader_agent
            ],
            tasks=[
                web_research_task,
                trend_analysis_task,
                report_writing_task,
                proofreading_task
            ],
            process=Process.hierarchical,
            manager_agent=manager_agent,
            verbose=True
        )

        # Execute crew workflow
        logger.info("Starting crew workflow...")
        topic = "NYC Real Estate Market"
        logger.info(f"Processing topic: {topic}")
        
        crew_output = crew.kickoff(inputs={"topic": topic})

        # Extract final output
        logger.info("Extracting final report...")
        report_text: Optional[str] = None
        
        try:
            report_text = crew_output.raw[0].final_output
        except (AttributeError, IndexError, KeyError) as e:
            logger.warning(f"Could not extract from raw output: {e}, trying string conversion")
            try:
                report_text = str(crew_output)
            except Exception as e:
                logger.error(f"Failed to convert crew output to string: {e}")
                report_text = ""

        if not report_text:
            raise RuntimeError("Failed to extract report text from crew output")

        logger.info("Report text extracted successfully")
        print("\n\n***** Final Output *****\n\n")
        print(report_text)

        # Save to PDF
        current_dir = Path(__file__).parent
        outputs_dir = current_dir / "outputs"
        pdf_path = outputs_dir / "final_report.pdf"
        
        logger.info(f"Saving PDF to: {pdf_path}")
        saved_path = save_report_to_pdf(report_text, str(pdf_path))
        print(f"\nSaved PDF to: {saved_path}")

        # === Email the final PDF ===
        try:
            to_email = receiver_email
            subject = "Automated Research Report"
            body = "Hello,\n\nPlease find attached the latest research report generated by the multi-agent workflow.\n\nBest regards,\nYour AI Assistant"
            send_email_with_attachment(to_email, subject, body, str(saved_path))
            logger.info(f"Email sent successfully to {to_email}")
        except Exception as e:
            logger.error(f"Failed to send email: {e}")

        
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        raise
    except RuntimeError as e:
        logger.error(f"Runtime error during report generation: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in main execution: {e}", exc_info=True)
        raise RuntimeError(f"Failed to generate report: {e}") from e


if __name__ == "__main__":
    main()