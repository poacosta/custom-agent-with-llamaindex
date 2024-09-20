# Custom Agent with LlamaIndex

## What is an AI Agent?

> By themselves, language models can't take actions — they just output text.
> Agents are systems that use LLMs as reasoning engines to determine which actions to take and the inputs to pass
> them.
> After executing actions, the results can be fed back into the LLM to determine whether more actions are necessary,
> or whether it is okay to finish.

## What does this Agent do?

Using predefined data of some cities (in the constants section of the code), the agent could be queried about these cities and will primarily provide information based on the data it has access to, extract contextual data from Wikipedia, and use the LLM to generate a response and evaluate it.

## Getting Started

### Prerequisites

You need to have the following installed on your machine:

- [Python 3.12](https://www.python.org/downloads/release/python-3124/) or later
- [Pip](https://pypi.org/project/pip/)
- [Virtualenv](https://pypi.org/project/virtualenv/)

```bash
# LlamaIndex LLMs Integration: Openai
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

### Running the Agent

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

Make sure to set your OpenAI key before running: `export OPENAI_API_KEY-="sk-..."`

### Querying the Agent

Once the agent is running, you can add a query to the agent by typing it in the terminal.

If the terminal is ready will show the following message:

```bash
Initializing City Query System...
City Query System is ready. You can start asking questions.
Type 'exit' or press Ctrl+C to end the session.

Enter your question: 
```

Add your query and press `Enter`.

For example, you can ask the agent:

```bash
Which countries are each city from?
```

Have in mind that the agent will primarily provide information based on the data it has access to,
will extract context data from Wikipedia, and will use the LLM to generate a response and evaluate it.

See the Constants section in code to know the loaded info.

## Related Docs

- [LlamaIndex Documentation](https://docs.llamaindex.ai/en/stable/)
- [Building a Custom Agent](https://docs.llamaindex.ai/en/stable/examples/agent/custom_agent/)
