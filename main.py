from typing import List, Dict, Any, Tuple

from llama_index.core import ChatPromptTemplate
from llama_index.core import VectorStoreIndex, SQLDatabase
from llama_index.core.agent import FnAgentWorker
from llama_index.core.bridge.pydantic import Field, BaseModel
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.program import FunctionCallingProgram
from llama_index.core.query_engine import NLSQLTableQueryEngine, RouterQueryEngine
from llama_index.core.selectors import PydanticSingleSelector
from llama_index.core.tools import QueryEngineTool
from llama_index.llms.openai import OpenAI
from llama_index.readers.wikipedia import WikipediaReader
from sqlalchemy import create_engine, MetaData, Table, Column, String, Integer, insert

# Constants
DEFAULT_PROMPT_STR = """
Given previous question/response pairs, please determine if an error has occurred in the response, and suggest \
    a modified question that will not trigger the error.

Examples of modified questions:
- The question itself is modified to elicit a non-erroneous response
- The question is augmented with context that will help the downstream system better answer the question.
- The question is augmented with examples of negative responses, or other negative questions.

An error means that either an exception has triggered, or the response is completely irrelevant to the question.

Please return the evaluation of the response in the following JSON format.
"""

MODEL = "gpt-4o"

CITIES = ["La Habana", "Monterrey", "Málaga"]

CITIES_DATA = [
    {"city_name": "La Habana", "population": 2106657, "country": "Cuba"},
    {"city_name": "Monterrey", "population": 5339425, "country": "Mexico"},
    {"city_name": "Málaga", "population": 570000, "country": "Spain"},
]


class CityDatabase:
    """Manages the SQLite database for storing city information."""

    def __init__(self):
        # Create an in-memory SQLite database
        self.engine = create_engine("sqlite:///:memory:", future=True)
        self.metadata = MetaData()
        self.table = self._create_table()
        self.metadata.create_all(self.engine)

    def _create_table(self):
        """Create the city_stats table schema."""
        return Table(
            "city_stats",
            self.metadata,
            Column("city_name", String(16), primary_key=True),
            Column("population", Integer),
            Column("country", String(16), nullable=False),
        )

    def insert_data(self, rows: List[Dict[str, Any]]):
        """Insert multiple rows of data into the city_stats table."""
        for row in rows:
            stmt = insert(self.table).values(**row)
            with self.engine.begin() as connection:
                connection.execute(stmt)


class ResponseEval(BaseModel):
    """Pydantic model for evaluating responses and suggesting new questions."""

    has_error: bool = Field(..., description="Whether the response has an error.")
    new_question: str = Field(..., description="The suggested new question.")
    explanation: str = Field(..., description="The explanation for the error and the new question.")


class CityQuerySystem:
    """Main class for handling city queries using SQL and vector databases."""

    def __init__(self, llm: OpenAI):
        self.llm = llm
        self.db = CityDatabase()
        self.sql_tool = self._create_sql_tool()
        self.vector_tools = self._create_vector_tools()
        self.router_query_engine = self._create_router_query_engine()

    def _create_sql_tool(self):
        """Create an SQL query tool for the city_stats table."""
        sql_database = SQLDatabase(self.db.engine, include_tables=["city_stats"])
        sql_query_engine = NLSQLTableQueryEngine(
            sql_database=sql_database, tables=["city_stats"], verbose=True, llm=self.llm
        )
        return QueryEngineTool.from_defaults(
            query_engine=sql_query_engine,
            description="Useful for translating a natural language query into a SQL query over a table containing: city_stats, containing the population/country of each city"
        )

    @staticmethod
    def _create_vector_tools():
        """Create vector tools for semantic queries about specific cities."""

        # Load Wikipedia data for the specified cities
        wiki_docs = WikipediaReader().load_data(pages=CITIES)

        vector_tools = []
        for city, wiki_doc in zip(CITIES, wiki_docs):
            vector_index = VectorStoreIndex.from_documents([wiki_doc])
            vector_query_engine = vector_index.as_query_engine()
            vector_tool = QueryEngineTool.from_defaults(
                query_engine=vector_query_engine,
                description=f"Useful for answering semantic questions about {city}"
            )
            vector_tools.append(vector_tool)
        return vector_tools

    def _create_router_query_engine(self):
        """Create a router query engine to select between SQL and vector tools."""
        return RouterQueryEngine(
            selector=PydanticSingleSelector.from_defaults(llm=self.llm),
            query_engine_tools=[self.sql_tool] + self.vector_tools,
            verbose=True,
        )

    @staticmethod
    def get_chat_prompt_template(system_prompt: str, current_reasoning: List[Tuple[str, str]]) -> ChatPromptTemplate:
        """Create a chat prompt template based on the system prompt and current reasoning."""
        system_msg = ChatMessage(role=MessageRole.SYSTEM, content=system_prompt)
        messages = [system_msg]
        for role, content in current_reasoning:
            messages.append(
                ChatMessage(role=MessageRole.USER if role == "user" else MessageRole.ASSISTANT, content=content))
        return ChatPromptTemplate(message_templates=messages)

    def retry_agent_fn(self, state: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
        """Function for the retry agent to process queries and evaluate responses."""
        task, router_query_engine = state["__task__"], self.router_query_engine
        prompt_str = state["prompt_str"]
        verbose = state.get("verbose", False)

        new_input = state.get("new_input", task.input)
        response = router_query_engine.query(new_input)

        state["current_reasoning"].extend([("user", new_input), ("assistant", str(response))])

        # Create a chat prompt template and LLM program for response evaluation
        chat_prompt_tmpl = self.get_chat_prompt_template(prompt_str, state["current_reasoning"])
        llm_program = FunctionCallingProgram.from_defaults(
            output_cls=ResponseEval,
            prompt=chat_prompt_tmpl,
            llm=self.llm,
        )

        # Evaluate the response and determine if a retry is necessary
        response_eval = llm_program(query_str=new_input, response_str=str(response))
        is_done = not response_eval.has_error
        state["new_input"] = response_eval.new_question

        if verbose:
            print(f"> Question: {new_input}")
            print(f"> Response: {response}")
            print(f"> Response eval: {response_eval.model_dump()}")

        state["__output__"] = str(response)
        return state, is_done

    def create_agent(self):
        """Create an agent worker for handling city queries."""
        return FnAgentWorker(
            fn=self.retry_agent_fn,
            initial_state={
                "prompt_str": DEFAULT_PROMPT_STR,
                "llm": self.llm,
                "router_query_engine": self.router_query_engine,
                "current_reasoning": [],
                "verbose": True,
            },
        ).as_agent()


def main():
    """Main function to run the interactive City Query System."""
    print("Initializing City Query System...")
    llm = OpenAI(model=MODEL)
    city_query_system = CityQuerySystem(llm)

    # Insert sample data into the city_stats table
    city_query_system.db.insert_data(CITIES_DATA)

    # Create an agent for handling queries
    agent = city_query_system.create_agent()

    print("City Query System is ready. You can start asking questions.")
    print("Type 'exit' or press Ctrl+C to end the session.")

    try:
        while True:
            user_input = input("\nEnter your question: ")
            if user_input.lower() == 'exit':
                break

            response = agent.chat(user_input)
            print("\nResponse:", str(response))

    except KeyboardInterrupt:
        print("\nExiting the City Query System. Goodbye!")


if __name__ == "__main__":
    main()
