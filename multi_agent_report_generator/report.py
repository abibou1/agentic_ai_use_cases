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
print(web_researcher_agent.tools)