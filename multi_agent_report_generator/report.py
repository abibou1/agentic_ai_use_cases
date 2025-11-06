"""Multi-agent report generator using CrewAI.

This module implements a multi-agent system for generating research reports
on various topics using web research, trend analysis, report writing, and proofreading agents.
"""

from crewai import Crew, Task, Agent, Process, LLM  # pyright: ignore[reportMissingImports]
from langchain_community.tools import BraveSearch

import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional

from services.email_sender import send_email_with_attachment
from services.pdf_generator import save_report_to_pdf


from dotenv import load_dotenv
 

from datetime import date
as_of = date.today().isoformat()

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

# create llm with gpt 3.5 turbo
def create_llm_gpt_3_5_turbo() -> LLM:
    """Create and configure the LLM instance.
    
    Creates an LLM instance using OpenAI's GPT-3.5 Turbo model with the API key
    from environment variables.
    
    Returns:
        LLM: Configured LLM instance for use by agents.
    """
    try:
        api_key = ensure_openai_api_key()
        llm = LLM(
            model='openai/gpt-3.5-turbo',
            api_key=api_key
        )
        logger.info("LLM instance created successfully with model: openai/gpt-3.5-turbo")
        return llm

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
    """Create a simple dict-based tool CrewAI Agent can accept.
    
    Returns:
        Dict[str, Any]: name, description, and callable function.
    """
    return {
        "name": "brave_search",
        "description": (
            "Search the web using BraveSearch and return relevant, recent information for a given query."
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
        # llm = create_llm()
        llm = create_llm_gpt_3_5_turbo()
        
        # (Temporarily disabled) Create the BraveSearch tool
        # search_tool = create_brave_search_tool()
        
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
                "Conduct web-based research to identify 5-7 key insights about {topic} as of {as_of}. "
                "Use only recent and credible sources (prefer last 6–12 months). Include the source URL "
                "for every insight. Prefer primary sources, government/official stats, and reputable media."
            ),
            expected_output="A structured list of 5-7 insights with a short summary and a URL for each."
        )

        trend_analysis_task = Task(
            description="Analyze the findings (with citations) and rank trends by importance and impact; flag any stale sources.",
            expected_output="A table ranking trends by impact, with concise descriptions and source URLs."
        )

        report_writing_task = Task(
            description=(
                "Draft a professional report on {topic} as of {as_of}. Include: Introduction, Trends Overview, "
                "Analysis, Recommendations. Retain footnote-style citations for all referenced facts/figures."
            ),
            expected_output="A structured draft with clear flow and in-text or footnote citations."
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
        
        crew_output = crew.kickoff(inputs={"topic": topic, "as_of": as_of})

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

        header = f"# {topic} — Report\n\nLast updated: {as_of}\n\n"
        print("Header: ", header)

        report_text = header + report_text
        
        logger.info("Report text extracted successfully")

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