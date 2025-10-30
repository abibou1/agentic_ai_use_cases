from crewai import Crew, Task, Agent, Process, LLM
from pydantic import BaseModel
from crewai.tools.structured_tool import CrewStructuredTool
from langchain_community.tools import BraveSearch