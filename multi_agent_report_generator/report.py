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

print("\n\n***** BraveSearch Tool *****\n\n")
print(SearchTool["description"])


# Define agents

web_researcher_agent = Agent(
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

print("\n\n***** Web Researcher Agent *****\n\n")
print(web_researcher_agent.role)
print(web_researcher_agent.goal)

trend_analyst_agent = Agent(
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
    verbose=True
)

print("\n\n***** Manager Agent *****\n\n")
print(manager_agent.role)
print(manager_agent.goal)

print("\n\n***** Proofreader Agent *****\n\n")
print(proofreader_agent.role)
print(proofreader_agent.goal)

print("\n\n***** Report Writer Agent *****\n\n")
print(report_writer_agent.role)
print(report_writer_agent.goal)

print("\n\n***** Trend Analyst Agent *****\n\n")
print(trend_analyst_agent.role)
print(trend_analyst_agent.goal)