# Recursive Document Analysis Agent

This project implements a recursive document analysis system using LangGraph and OpenAI's GPT models. The system processes documents through multiple iterations to improve the quality of analysis based on predefined criteria.

## Features

- PDF text extraction and processing
- Recursive improvement of document analysis
- Quality assessment based on multiple criteria
- Parallel processing of multiple prompt sets
- Configurable prompt sets via JSON
- History tracking of improvements
- Quality metrics tracking
- State management for iterative improvements

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
pip install pdfplumber  # For PDF text extraction
```
3. Create a `.env` file with your OpenAI API key:
```

```

## Usage

### PDF Text Extraction

```python
from recursive_agent import extract_text_from_pdf

# Extract text from a PDF file
pdf_path = "path/to/your/document.pdf"
document_text = extract_text_from_pdf(pdf_path)
```

### Basic Usage

```python
from recursive_agent import RecursiveAgent

# Initialize the agent
agent = RecursiveAgent()

# Process a PDF document
pdf_path = "path/to/your/document.pdf"
document_text = extract_text_from_pdf(pdf_path)

# Process with a specific prompt set
results = agent.process_prompt_set(document_text, "metadata")

# Process with all prompt sets
all_results = agent.process_all_prompt_sets(document_text)
```

### State Management

The system uses a state-based approach to manage the analysis process:

```python
class AgentState:
    document_text: str          # Input text from PDF
    current_prompt: str         # Current prompt being processed
    previous_outputs: List[Dict] # History of outputs
    iteration_count: int        # Current iteration (0-3)
    max_iterations: int = 3     # Maximum improvement attempts
    improvement_threshold: float = 0.8  # Quality threshold
    current_quality_score: float # Current quality (0-1)
    final_output: str           # Current/final output
    system_prompt: str          # Model instructions
```

### Quality Assessment

The system assesses quality based on weighted criteria:
- Completeness (30%)
- Accuracy (30%)
- Clarity (20%)
- Technical depth (20%)

Quality thresholds:
- Improvement threshold: 0.8 (80%)
- Maximum iterations: 3
- Stops when either threshold is reached

### Customizing Prompt Sets

Prompt sets are defined in `prompt_sets.json`. Each prompt set has:
- A system prompt
- A set of prompts to process
- Optional sub-prompts

Example prompt set structure:
```json
{
    "metadata": {
        "system_prompt": "You are a metadata extractor for technical documents. Be concise.",
        "prompts": {
            "document_title": "What is the title of the document?",
            "document_type": "What type of document is this?"
        }
    }
}
```

### Configuration

You can configure:
- Maximum iterations (default: 3)
- Quality threshold (default: 0.8)
- Quality metrics weights
- Model parameters
- PDF extraction settings

## Architecture

The system uses a recursive workflow:
1. PDF text extraction
2. Initial analysis
3. Quality assessment
4. Improvement generation
5. Repeat until quality threshold or max iterations

### State Flow
1. Initialize state with document text
2. Process initial prompt
3. Assess quality
4. If quality < threshold and iterations < max:
   - Generate improvements
   - Update state
   - Repeat from step 3
5. Return final state with results

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License 