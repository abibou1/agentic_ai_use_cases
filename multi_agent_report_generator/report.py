from crewai import Crew, Task, Agent, Process, LLM  # pyright: ignore[reportMissingImports]
from pydantic import BaseModel
from langchain_community.tools import BraveSearch

import os
from dotenv import load_dotenv


# ensure openai api key is set
def ensure_openai_api_key():
    load_dotenv()
    openai_api_key = os.getenv('OPENAI_API_KEY')
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY is not set")
    return openai_api_key

ensure_openai_api_key()


def create_llm():
    llm = LLM(
        model='openai/gpt-4o',
        api_key=ensure_openai_api_key()
    )
    return llm

llm = create_llm()

# Save text content to a PDF file
def save_report_to_pdf(text: str, file_path: str) -> str:
    try:
        from reportlab.lib.pagesizes import LETTER  # type: ignore[reportMissingImports]
        from reportlab.lib.units import inch  # type: ignore[reportMissingImports]
        from reportlab.pdfgen import canvas  # type: ignore[reportMissingImports]
        from reportlab.pdfbase.pdfmetrics import stringWidth  # type: ignore[reportMissingImports]
    except ImportError as e:
        raise ImportError("reportlab is required to export PDF. Install with: pip install reportlab") from e

    output_dir = os.path.dirname(file_path) or "."
    os.makedirs(output_dir, exist_ok=True)

    c = canvas.Canvas(file_path, pagesize=LETTER)
    width, height = LETTER
    margin = 0.75 * inch
    usable_width = width - 2 * margin
    line_height = 14
    y = height - margin

    c.setFont("Times-Roman", 11)

    def new_page_if_needed():
        nonlocal y
        if y < margin:
            c.showPage()
            c.setFont("Times-Roman", 11)
            y = height - margin

    for paragraph in (text or "").split("\n\n"):
        for raw_line in paragraph.split("\n"):
            words = raw_line.split(" ") if raw_line else [""]
            current = ""
            for word in words:
                candidate = (current + " " + word).strip()
                if stringWidth(candidate, "Times-Roman", 11) <= usable_width:
                    current = candidate
                else:
                    c.drawString(margin, y, current)
                    y -= line_height
                    new_page_if_needed()
                    current = word
            if current:
                c.drawString(margin, y, current)
                y -= line_height
                new_page_if_needed()
        y -= line_height  # paragraph spacing
        new_page_if_needed()

    c.save()
    return file_path

def brave_search_wrapper(query: str) -> str:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query must be a non-empty string.")

    brave_api_key = os.getenv("BRAVE_API_KEY", "BRAVE-API-KEY")
    brave_search = BraveSearch.from_api_key(
        api_key=brave_api_key,
        search_kwargs={"count": 3}
    )

    return brave_search.run(query)

def create_brave_search_tool():
    return {
        "name": "brave_search_tool",
        "description": (
            "Searches the web using BraveSearch and returns relevant information for a given query. "
            "Useful for finding up-to-date and accurate information on a wide range of topics."
        ),
        "function": brave_search_wrapper,
    }

# Create the BraveSearch tool
SearchTool = create_brave_search_tool()


# Define agents

web_researcher_agent = Agent(
    name="Web Researcher",
    role="Web Research Specialist",
    goal=(
        "To find the most recent, impactful, and relevant about {topic}. This includes identifying "
        "key use cases, challenges, and statistics to provide a foundation for deeper analysis."
    ),
    backstory=(
        "You are a former investigative journalist known for your ability to uncover technology breakthroughs "
        "and market insights. With years of experience, you excel at identifying actionable data and trends."
    ),
    tools=[], 
    llm=llm,
    verbose=True
)

trend_analyst_agent = Agent(
    name="Trend Analyst",
    role="Insight Synthesizer",
    goal=(
        "To analyze research findings, extract significant trends, and rank them by industry impact, growth potential, "
        "and uniqueness. Provide actionable insights for decision-makers."
    ),
    backstory=(
        "You are a seasoned strategy consultant who transitioned into {topic} analysis. With an eye for patterns, "
        "you specialize in translating raw data into clear, actionable insights."
    ),
    tools=[],
    llm=llm,
    verbose=True
)

report_writer_agent = Agent(
    name="Report Writer",
    role="Narrative Architect",
    goal=(
        "To craft a detailed, professional report that communicates research findings and analysis effectively. "
        "Focus on clarity, logical flow, and engagement."
    ),
    backstory=(
        "Once a technical writer for a renowned journal, you are now dedicated to creating industry-leading reports. "
        "You blend storytelling with data to ensure your work is both informative and captivating."
    ),
    tools=[],  
    llm=llm,  
    verbose=True
)

proofreader_agent = Agent(
    name="Proofreader",
    role="Polisher of Excellence",
    goal=(
        "To refine the report for grammatical accuracy, readability, and formatting, ensuring it meets professional "
        "publication standards."
    ),
    backstory=(
        "An award-winning editor turned proofreader, you specialize in perfecting written content. Your sharp eye for "
        "detail ensures every document is flawless."
    ),
    tools=[],  
    llm=llm,  
    verbose=True
)

manager_agent = Agent(
    name="Manager",
    role="Workflow Maestro",
    goal=(
        "To coordinate agents, manage task dependencies, and ensure all outputs meet quality standards. Your focus "
        "is on delivering a cohesive final product through efficient task management."
    ),
    backstory=(
        "A former project manager with a passion for efficient teamwork, you ensure every process runs smoothly, "
        "overseeing tasks and verifying results."
    ),
    tools=[],  
    llm=llm, 
    verbose=True,
    allow_delegation=True
)


# Define tasks
web_research_task = Task(
    description=(
        "Conduct web-based research to identify 5-7 of the {topic}. Focus on key use cases. "
    ),
    expected_output=(
        "A structured list of 5-7 {topic}."
    )
)

print("\n\n***** Web Research Task *****\n\n")
print(web_research_task.description)

trend_analysis_task = Task(
    description=(
        "Analyze the research findings to rank {topic}. "
    ),
    expected_output=(
        "A table ranking trends by impact, with concise descriptions of each trend."
    )
)

report_writing_task = Task(
    description=(
        "Draft report summarizing the findings and analysis of {topic}. Include sections for "
        "Introduction, Trends Overview, Analysis, and Recommendations."
    ),
    expected_output=(
        "A structured, professional draft with a clear flow of information. Ensure logical organization and consistent tone."
    )
)

proofreading_task = Task(
    description=(
        "Refine the draft for grammatical accuracy, coherence, and formatting. Ensure the final document is polished "
        "and ready for publication."
    ),
    expected_output=(
        "A professional, polished report free of grammatical errors and inconsistencies. Format the document for "
        "easy readability."
    )
)

crew = Crew(
    agents=[web_researcher_agent, trend_analyst_agent, report_writer_agent, proofreader_agent],
    tasks=[web_research_task, trend_analysis_task, report_writing_task, proofreading_task],
    process=Process.hierarchical,
    manager_agent=manager_agent,
    verbose=True
)

print("\n\n***** Crew *****\n\n")
# crew_output = crew.kickoff(inputs={"topic": "AI Trends"})
crew_output = crew.kickoff(inputs={"topic": "NYC Real Estate Market"})

# get final output
print("\n\n***** Final Output *****\n\n")
report_text = None
try:
    report_text = crew_output.raw[0].final_output
except Exception:
    try:
        report_text = str(crew_output)
    except Exception:
        report_text = ""
print(report_text)

# save to PDF under this project folder's outputs directory
current_dir = os.path.dirname(__file__)
outputs_dir = os.path.join(current_dir, "outputs")
pdf_path = os.path.join(outputs_dir, "final_report.pdf")
saved = save_report_to_pdf(report_text, pdf_path)
print(f"\nSaved PDF to: {saved}")