# multi_agent_report_generator

Multi-agent workflow that researches a topic, analyzes trends, writes a report, proofreads it, and saves the final report as a PDF.

## Features
- Web research agent (extensible to use Brave search)
- Trend analysis, report writing, and proofreading agents
- Hierarchical process with a manager agent
- Exports final output to PDF at `multi_agent_report_generator/outputs/final_report.pdf`

## Setup
1) Create and activate a virtual environment:
```sh
python -m venv myenv
myenv\Scripts\activate
```

2) Install dependencies:
```sh
pip install -r requirements.txt
```

3) Environment variables:
- `OPENAI_API_KEY` (required)
- `BRAVE_API_KEY` (optional, only if enabling Brave search)

You can place them in a `.env` file in this folder:
```
OPENAI_API_KEY=...
BRAVE_API_KEY=...
```

## Run
```sh
python report.py
```

The final PDF is saved to:
```
multi_agent_report_generator/outputs/final_report.pdf
```