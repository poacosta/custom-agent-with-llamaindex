# Custom Agent with LlamaIndex

A sophisticated AI agent system built with LlamaIndex that combines structured data querying, Wikipedia knowledge
extraction, and intelligent response evaluation. Perfect for developers looking to explore the intersection of SQL
databases, vector stores, and LLM-powered agents.

> 📚 **Academic Project Notice**: This is an educational project designed for learning and experimentation with
> LLM-powered agents. While it demonstrates interesting capabilities, it's not intended for production use. Think of it
> as
> a sophisticated playground for understanding agent architectures and LlamaIndex implementations.

## 🎯 What's This All About?

Using predefined data of some cities (in the constants section of the code), the agent could be queried about these
cities and will primarily provide information based on the data it has access to, extract contextual data from
Wikipedia, and use the LLM to generate a response and evaluate it.

It's like having a smart assistant that can:

- Navigate both structured city data and Wikipedia knowledge
- Self-evaluate its responses for accuracy
- Automatically retry queries with improved prompting when needed
- Handle natural language questions about cities with grace

## 🧠 Understanding the Agent's Behavior

The agent employs a process that combines several AI reasoning techniques and can be best characterized as **"Iterative
Self-Refinement with Hybrid Data Integration."**

Here’s a breakdown of the components and how they align with existing concepts:

### Key Components:

1. **Iterative Self-Evaluation/Refinement**  
   The agent critiques its own responses, identifies gaps (e.g., outdated data, lack of context), and reformulates
   questions to improve accuracy. This mirrors **self-correction** and **feedback loops**, where each iteration refines
   the output based on internal evaluation.

2. **Hybrid Data Integration**  
   It combines:
    - **Structured data** (predefined constants/SQL queries for city statistics).
    - **Unstructured semantic data** (Wikipedia/contextual information via LLMs).  
      This resembles **multi-engine query systems** that switch between structured and semantic approaches depending on
      the question.

3. **Contextual Adaptation**  
   The agent dynamically adjusts its focus (e.g., clarifying "metropolitan area" vs. city proper) to resolve
   ambiguities, akin to **active questioning** or **problem decomposition**.

### Alignment with Existing Concepts:

- **Chain of Thought (CoT)**: Extended here with iterative self-critique, making it closer to **"Self-Reflective Chain
  of Thought"** or **"Chain of Verification"** (though the latter typically verifies facts rather than refining
  questions).
- **Tree of Thoughts (ToT)**: Less about branching exploration and more about linear refinement, so not a perfect fit.
- **Recursive Reasoning**: The repeated re-evaluation aligns with recursive problem-solving but adds structured data
  integration.
- **Meta-Reasoning**: The agent reasons about its own reasoning process (e.g., selecting query engines, evaluating
  responses).

Here's a full example of responses and self-evaluation in action:

```bash
Initializing City Query System...
City Query System is ready. You can start asking questions.
Type 'exit' or press Ctrl+C to end the session.

Enter your question: Tell me about the population of Monterrey

Selecting query engine 0: The question is about the population of Monterrey, which requires translating a natural language query into a SQL query over a table containing city statistics, including population..
> Question: Tell me about the population of Monterrey
> Response: Monterrey has a population of approximately 5,339,425 people.
> Response eval: {'has_error': True, 'new_question': 'What is the current population of Monterrey, Mexico, and how has it changed over recent years?', 'explanation': 'The response provides a specific population number without context or a source, which may not be accurate or up-to-date. A modified question asking for the current population and recent changes would prompt a more comprehensive and accurate response.'}

Selecting query engine 2: Choice (3) is relevant because it is useful for answering semantic questions about Monterrey, which is directly related to the question about the current population and its changes over recent years..
> Question: What is the current population of Monterrey, Mexico, and how has it changed over recent years?
> Response: The current population of Monterrey, Mexico is 1,142,194 as of 2020.
> Response eval: {'has_error': True, 'new_question': 'What is the current population of Monterrey, Mexico, including its metropolitan area, and how has it changed over recent years?', 'explanation': 'The response provided the population of Monterrey city proper, but the initial question likely intended to include the metropolitan area, which is significantly larger. Additionally, the response did not address the change in population over recent years. The modified question clarifies the need for information about the metropolitan area and the population trend.'}

Selecting query engine 2: Choice 3 is relevant because it is useful for answering semantic questions about Monterrey, which is the city in question..
> Question: What is the current population of Monterrey, Mexico, including its metropolitan area, and how has it changed over recent years?
> Response: The estimated population of Monterrey, Mexico, including its metropolitan area, is 5,341,171 people as of 2020. The population of Monterrey itself is 1,142,194 according to the 2020 census.
> Response eval: {'has_error': True, 'new_question': 'What is the current population of Monterrey, Mexico, including its metropolitan area, and how has it changed over the past decade?', 'explanation': 'The response provided an outdated population figure for Monterrey and did not address the change over recent years. By specifying the need for historical context over the past decade, the question can guide the response to include both current and historical population data.'}

Selecting query engine 2: Choice (3) is most relevant because it is specifically useful for answering semantic questions about Monterrey, which is the city in question..
> Question: What is the current population of Monterrey, Mexico, including its metropolitan area, and how has it changed over the past decade?
> Response: The estimated population of Monterrey, Mexico, including its metropolitan area, is 5,341,171 people as of 2020. Over the past decade, the population has increased from the data provided in the national INEGI population census of 2010, where 87.3% of the total population of the state of Nuevo León lived in the Monterrey metropolitan area.
> Response eval: {'has_error': False, 'new_question': 'What is the current population of Monterrey, Mexico, including its metropolitan area, and what are the specific growth trends over the past decade?', 'explanation': "The response provides the current population of Monterrey's metropolitan area and mentions an increase over the past decade. However, it lacks specific details about the growth trends or figures over the past decade. The modified question seeks more detailed information about the growth trends to provide a comprehensive understanding of population changes."}

Response: The estimated population of Monterrey, Mexico, including its metropolitan area, is 5,341,171 people as of 2020. Over the past decade, the population has increased from the data provided in the national INEGI population census of 2010, where 87.3% of the total population of the state of Nuevo León lived in the Monterrey metropolitan area.
```

## ✨ Key Features

- **Hybrid Query System**: Seamlessly combines SQL and vector store capabilities
- **Smart Response Evaluation**: Built-in response quality assessment and query refinement
- **Wikipedia Integration**: Automatic context enrichment from Wikipedia
- **Interactive CLI Interface**: User-friendly command-line interaction

## 🚀 Quick Start

### Prerequisites

You'll need these installed:

- [Python 3.12+](https://www.python.org/downloads/release/python-3124/)
- [pip](https://pypi.org/project/pip/)
- [virtualenv](https://pypi.org/project/virtualenv/)

### Environment Setup

```bash
# Clone this repository
git clone https://github.com/poacosta/custom-agent-with-llamaindex.git

# Navigate to project directory
cd custom-agent-with-llamaindex

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Core Components Installation

# LlamaIndex's OpenAI Integration Trifecta 🎯
pip install llama-index-llms-openai        # Core LLM capabilities
pip install llama-index-embeddings-openai   # Vector embeddings magic
pip install llama-index-program-openai      # Structured outputs handling

# Knowledge Enhancement Layer 📚
pip install wikipedia                       # Wikipedia API wrapper
pip install llama-index-readers-wikipedia   # LlamaIndex's Wikipedia parser

# Or use the full package installation
pip install -r requirements.txt

# Set your OpenAI API key
export OPENAI_API_KEY="sk-..."  # On Windows: set OPENAI_API_KEY=sk-...
```

### Running the Agent

```bash
python main.py
```

## 💡 Usage Examples

Once the agent is running, you can add a query to the agent by typing it in the terminal.

If the terminal is ready will show the following message:

```bash
Initializing City Query System...
City Query System is ready. You can start asking questions.
Type 'exit' or press Ctrl+C to end the session.

Enter your question: 
```

Add your query and press `Enter`.

Query examples:

```
> Tell me about the population of Monterrey
> What's the historical significance of La Habana?
> Compare the populations of all cities in the database
```

## 🏗️ Architecture Overview

The system is built on three main pillars:

1. **SQL Database Layer**
    - Manages structured city data
    - Handles quantitative queries efficiently

2. **Vector Store Layer**
    - Processes semantic queries
    - Integrates Wikipedia knowledge

3. **Agent Layer**
    - Coordinates between query engines
    - Evaluates and improves responses
    - Manages conversation flow

## 🔧 Technical Deep Dive

### Component Architecture

Agent's capabilities are built on several key LlamaIndex integrations:

1. **LLM Layer** (`llama-index-llms-openai`)
    - Handles natural language understanding
    - Powers response generation and evaluation
    - Manages conversation flow

2. **Embeddings Layer** (`llama-index-embeddings-openai`)
    - Converts text into vector representations
    - Enables semantic search capabilities
    - Powers similarity matching

3. **Program Layer** (`llama-index-program-openai`)
    - Structures agent outputs
    - Manages function calling
    - Handles response formatting

4. **Knowledge Layer** (`llama-index-readers-wikipedia`)
    - Integrates Wikipedia content
    - Processes and chunks articles
    - Enhances response context

## 🔍 Advanced Configuration

### Customizing the Agent

You can modify the agent's behavior by adjusting:

- Prompt templates in `DEFAULT_PROMPT_STR`
- City data in `CITIES_DATA`
- LLM model selection in `MODEL`

### Performance Optimization

For production deployments, consider:

- Using connection pooling for the database
- Implementing caching for Wikipedia queries
- Adjusting batch sizes for vector operations

## 🚧 Known Limitations

- This is an educational project, not suitable for production deployment
- Designed for learning and experimentation purposes
- Currently, it supports a fixed set of cities
- Requires active internet connection for Wikipedia queries
- Response times may vary based on OpenAI API latency
- No production-grade security measures implemented
- Limited error handling for educational clarity
