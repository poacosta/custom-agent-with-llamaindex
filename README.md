# Custom Agent with LlamaIndex

## What is an AI Agent?

> By themselves, language models can't take actions — they just output text.
> Agents are systems that use LLMs as reasoning engines to determine which actions to take and the inputs to pass
> them. After executing actions, the results can be fed back into the LLM to determine whether more actions are needed,
> or
> whether it is okay to finish.

## What does this Agent do?

## Getting Started

### Prerequisites

You need to have the following installed on your machine:

- [Python 3.12](https://www.python.org/downloads/release/python-3124/) or later
- [Pip](https://pypi.org/project/pip/)
- [Virtualenv](https://pypi.org/project/virtualenv/)

```bash
# LlamaIndex Llms Integration: Openai
pip install llama-index-llms-openai

# LlamaIndex Embeddings Integration: Openai
pip install llama-index-embeddings-openai

# LlamaIndex Program Integration: Openai Program
pip install llama-index-program-openai

# Wikipedia is a Python library that makes it easy to access and parse data from Wikipedia.
pip install wikipedia

# The Wikipedia Reader reads Wikipedia pages and retrieves their content. 
# It allows you to specify a list of pages to read, and it retrieves the text content of each page.
pip install llama-index-readers-wikipedia
```

## How to use the Agent

```bash
# Clone the repository
git clone

# Change directory
cd custom-agent-with-llamaindex

# Create a virtual environment
virtualenv venv

# Activate the virtual environment
source venv/bin/activate

# Install the dependencies
pip install -r requirements.txt

# Run the agent
python main.py
```

## Related Docs

- [LlamaIndex Documentation](https://docs.llamaindex.ai/en/stable/)
- [Building a Custom Agent](https://docs.llamaindex.ai/en/stable/examples/agent/custom_agent/)