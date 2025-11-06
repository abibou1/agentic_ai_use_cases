# multi_agent_report_generator

Multi-agent workflow that researches a topic, analyzes trends, writes a report, proofreads it, and saves the final report as a PDF.

## Features
- Web research agent (extensible to use Brave search)
- Trend analysis, report writing, and proofreading agents
- Hierarchical process with a manager agent
- Exports final output to PDF at `multi_agent_report_generator/outputs/final_report.pdf`

## Technologies Used
- Python 3.10+
- CrewAI (agents, tasks, hierarchical process)
- OpenAI API (via CrewAI `LLM`)
- LangChain Community Tools (`BraveSearch` wrapper)
- ReportLab (PDF generation)
- python-dotenv (environment management)
- Pydantic (schemas/validation)

## Prerequisites

- An OpenAI API key
- (Optional) A Brave Search API key if you enable the search tool
- Windows PowerShell (examples use PowerShell syntax)

## Setup
1) Create and activate a virtual environment (Windows PowerShell):
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

2) Install dependencies (pip):
```powershell
pip install crewai langchain-community python-dotenv reportlab pydantic
```

3) Environment variables:
- `OPENAI_API_KEY` (required)
- `BRAVE_API_KEY` (optional, only if enabling Brave search)

3) Create a `.env` file with the following variables:
```
OPENAI_API_KEY=your-openai-key
BRAVE_API_KEY=your-brave-key   # optional

# Email sending (optional)
SENDER_EMAIL=you@example.com
SENDER_PASSWORD=your-app-password-or-password
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
RECEIVER_EMAIL=recipient@example.com
```

## Run
```powershell
python report.py
```

The final PDF is saved to:
```
multi_agent_report_generator/outputs/final_report.pdf
```

## Configuration

- Topic: Update the default topic inside `report.py` by changing the `topic` variable.

## Troubleshooting

- Missing OpenAI key: Ensure `OPENAI_API_KEY` is set in your environment or `.env`.
- PDF generation errors: Install ReportLab (`pip install reportlab`). The script creates the `outputs` directory if missing.
- Email issues: Confirm `SENDER_EMAIL`, `SENDER_PASSWORD`, and SMTP settings. Some providers require an app password.

## Notes

- Code style: Ruff is recommended for linting/formatting.
- Python 3.10+ is required; 3.11+ also works.