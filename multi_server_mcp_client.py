import os
import getpass
import traceback
import re
import asyncio
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import MCPTool
from typing import List
import sys

load_dotenv()

tool_scores = {
    "jira": {
        "query_score": 0.0,
        "threshold": 5.0
    },
    "gaia": {
        "query_score": 0.0,
        "threshold": 3.0
    },
    # "postgres": {
    #     "score": 0.0,
    #     "threshold": 3.0
    # }
}


mcp_servers = {
    "postgres": {"command": "/Users/harshal.mittal/projects/myenv/bin/python", "args": ["/Users/harshal.mittal/projects/postgres-mcp/postgres_server.py"], "transport": "stdio"},
    "jira": {"command": "/Users/harshal.mittal/projects/myenv/bin/python", "args": ["/Users/harshal.mittal/projects/jira-mcp-server/jira_mcp_server.py"], "transport": "stdio"},
    "gaia": {"command": "/Users/harshal.mittal/projects/myenv/bin/python", "args": ["/Users/harshal.mittal/projects/gaia-mcp/gaia_mcp_server.py"], "transport": "stdio"}
}

if not os.getenv("AZURE_OPENAI_API_KEY"):
    os.environ["AZURE_OPENAI_API_KEY"] = getpass.getpass("Enter your Azure OpenAI API key: ")
if not os.getenv("AZURE_OPENAI_ENDPOINT"):
    os.environ["AZURE_OPENAI_ENDPOINT"] = getpass.getpass("Enter your Azure OpenAI endpoint: ")

model = AzureChatOpenAI(
    deployment_name="gpt-4o",  # or your deployment
    api_version="2023-07-01-preview",  # or your api version
    openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

async def score_tool_helpfulness(tool_name: str, user_input: str) -> float:
    prompt = f"""
        You are an intelligent assistant evaluating whether a user's query would benefit from access to internal tool-assisted context.

        You are scoring this specifically for the tool: **{tool_name}**

        ## Tool Descriptions:
        - **"jira"**: Use this for:
            - Past engineering issues or outages
            - Bug reports or Jira tickets
            - Error messages, logs, error codes
            - Troubleshooting recurring system or user issues reported earlier

        - **"gaia"**: Use this for:
            - Internal documentation, architecture, system behavior
            - Technical suggestions, troubleshooting steps, how an internal component works
            - FAQs or proprietary knowledge
            - Anything that would benefit from information not available publicly
            - Company-specific details (e.g., internal tools, business strategy, roadmap, telemetry, product usage)

        - **"postgres"**: Use for:
            - Queries that require **accurate, up-to-date, structured information**
            - Questions that ask about **inventory, metrics, lists, records, usage stats, logs, structured entities**
            - Examples include:
                - Lists of users, tickets, products, sales, movies, customers, teams
                - Queries like “show all <X> that meet <Y> criteria”
                - Any question where the best answer would come from querying **actual internal databases**

            Do **not** use general knowledge. If a user asks about data (e.g., “all orders last month” ), score high even if the answer seems known — it should come from **Postgres** to ensure accuracy.

        ---

        Scoring Guidelines:

        ### High Score (7–10):
        - Query is **specific and technical** (e.g., error messages, stack traces, logs, internal failures)
        - Query is **organization-specific** (mentions internal tools, team processes, system behavior)
        - Query refers to **known issues, components, tickets, or documents**
        - Even if the query sounds factual or general, if it could be more **accurately or comprehensively answered from internal DB**, score high

        ### Low Score (0–3):
        - The query is about definitions, concepts, explanations, or general-purpose reasoning
        - The answer is not dependent on **access to real data**

        ---

        Examples:

        | Query | Jira | Gaia | Postgres |
        |-------|------|------|----------|
        | What is TLS and how does it work? | 1 | 1 | 0 |
        | Error 'tcp_connect() failed: Connection reset by peer' — how to resolve it? | 9 | 8 | 2 |
        | What is Cohesity's product roadmap for backup optimization? | 2 | 10 | 1 |
        | Known bugs in deduplication in May 2024? | 10 | 4 | 2 |
        | How do our connectors handle TLS renegotiation internally? | 3 | 9 | 1 |
        | What is a Python list comprehension? | 1 | 0 | 0 |
        | How many support cases were created in April? | 9 | 3 | 9 |
        | What’s our sales trend over Q2? 2 | 4 | 9 |

        ---

        Your Task:

        Please return only a **single number from 0 to 10** indicating the likelihood that the tool `{tool_name}` will help answer the user query.
        """

    response = await model.ainvoke([
        SystemMessage(content=prompt),
        HumanMessage(content=user_input)
    ])
    try:
        return float(response.content.strip())
    except ValueError:
        return 0.0

async def get_jira_context(user_input: str, tool: MCPTool) -> str:
    jql_response = await model.ainvoke([
    SystemMessage(content="""
        You are a JQL generator expert. Your job is to translate any user query—whether it is a direct Jira search or a general question—into a valid JQL query that will fetch the most relevant issues from Jira.

        Instructions:
        - If the user query is a general question (e.g., "How do I resolve error 'Connection reset by peer'?"), generate a JQL that searches for issues containing the key terms from the query in the summary, description, or comments fields.
        - If the user query is already a JQL or a direct Jira search, simply return the appropriate JQL.
        - If you cannot generate a valid JQL for the user's query, return nothing (leave the response completely empty).
        - Only return the JQL query string. Do not explain anything, do not include code block markers, and do not add any extra text.

        Examples:
        User: How do I resolve error "Connection reset by peer"?
        JQL: summary ~ "Connection reset by peer" OR description ~ "Connection reset by peer" OR comment ~ "Connection reset by peer"

        User: Find all open bugs assigned to Alice
        JQL: type = Bug AND status = "Open" AND assignee = "Alice"

        User: project = ABC AND status = "To Do"
        JQL: project = ABC AND status = "To Do"
        """),
    HumanMessage(content=user_input)
    ])

    if jql_response:
        jql_clean = re.sub(r"^```.*?```$", "", jql_response.content, flags=re.DOTALL | re.MULTILINE)
        jql_lines = [line.strip() for line in jql_clean.splitlines() if line.strip() and not line.strip().lower().startswith("jql")]
        jql = jql_lines[0] if jql_lines else ""
        return await tool.ainvoke({"jql": jql})
    else:
        return "No valid JQL could be generated for this input."


async def get_combined_context(user_input: str, all_tools, tools_to_call):
    combined_context = ""
    if "jira" in tools_to_call:
        print("Using jira")
        tool_jira = next(t for t in all_tools if t.name == "get_issues")
        jira_context = await get_jira_context(user_input, tool_jira)
        combined_context += f"\nContext from Jira:\n{jira_context}\n"

    if "gaia" in tools_to_call:
        print("Using gaia")
        tool_gaia = next(t for t in all_tools if t.name == "ask")
        gaia_context = await tool_gaia.ainvoke({"question": user_input})
        combined_context += f"\nContext from Gaia:\n{gaia_context}\n"
    
    # print(json.dumps(tool_gaia.args_schema, indent=2))
    # print(json.dumps(tool_jira.args_schema, indent=2))

    return combined_context

async def run_chat():

    client = MultiServerMCPClient(mcp_servers)
    all_tools = await client.get_tools()
    print("Available tools:")
    for tool in all_tools:
        print("-", tool.name)

    # Use it with create_react_agent
    agent = create_react_agent(model, all_tools)
    # agent_executor = RunnableAgent(agent=agent)
    # agent_executor = initialize_agent(
    #     tools=all_tools,
    #     llm=model,
    #     agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    #     verbose=True
    # )
    print("Agent is ready. Type your message or 'exit' to quit.\n")

    combined_context = ""
    SYSTEM_PROMPT = SystemMessage(content="""
        You are a helpful assistant that uses information from multiple sources to answer user questions accurately and contextually.

        You may receive the following types of context as part of user_input:
        1. **Context from Gaia** — Technical documentation, system behavior, or internal knowledge articles.
        2. **Context from Jira** — Historical bug reports, support tickets, or issue tracking data.
        3. **Context from Postgres** — Structured data or metrics from internal databases.

        ### Instructions:
        - If no context is provided, respond conversationally and helpfully.
        - If only one type of context is provided, answer based solely on that source.
        - If multiple sources are provided, combine insights into a single, coherent answer:
            - Integrate Gaia and Jira by associating each insight from Gaia with relevant Jira issue(s), if available.
            - When Postgres data is available, use it to support or enhance insights from Gaia or Jira.
            - Avoid restating content separately by source.
            - Do not create labeled sections for each source.
            - Instead, weave the information into a unified, numbered list like this:

                1. <Insight based on Gaia/Postgres> — [Related Jira issue: <issue key or summary>]
                2. <Another insight> — [Related Jira issue: ...]
                3. <Any remaining relevant Jira issues not already referenced>
    """)
    chat_history = [SYSTEM_PROMPT]

    pg_system_prompt = """
        You are an intelligent assistant with access to tools to explore and query a PostgreSQL database.

        You should answer user questions **only using data from the database**. Never use external or prior knowledge.

        Available tools:
        - list_tables: Lists available tables.
        - describe_table: Describes the structure (columns) of a given table.
        - query: Executes SQL and returns result rows.

        ## Your Strategy:
        1. Use `list_tables` to see what tables are available.
        2. Use `describe_table` to inspect relevant tables (e.g., if user asks about movies or actors, inspect tables with those keywords).
        3. If you find a table containing useful columns (e.g., 'actor', 'title', 'cast', 'movie_name'), use `query` to fetch results.
        4. If no useful data exists, output just a single word **NO**.

        ALWAYS double-check table structure before generating SQL. Do not assume column names.

        ONLY answer using the retrieved results. Or if no useful result is found, respond **NO**.
        """

    while True:
        try:
            # Read input asynchronously (fallback to sync if needed)
            print("You: ", end="", flush=True)
            user_input = await asyncio.get_event_loop().run_in_executor(None, sys.stdin.readline)
            if not user_input:
                continue
            user_input = user_input.strip()
            if not user_input:
                continue
            if user_input.lower() == "exit":
                break

            combined_context = ""

            ## whether to call jira, gaia
            for tool in tool_scores:
                tool_scores[tool]["query_score"] = await score_tool_helpfulness(tool, user_input)
                print(f"{tool} score: {tool_scores[tool]["query_score"]}")
            tools_to_call = [
                tool for tool, info in tool_scores.items()
                if info["query_score"] >= info["threshold"]
            ]
            if tools_to_call:
                print("Using Jira/ Gaia tools")
                combined_context += await get_combined_context(user_input, all_tools, tools_to_call)


            ## always call postgres
            pg_result = await agent.ainvoke({"messages": [SystemMessage(content=pg_system_prompt), HumanMessage(content=user_input)]})
            last_pg_msg = pg_result["messages"][-1]
            pg_reply = last_pg_msg.content if isinstance(last_pg_msg.content, str) else str(last_pg_msg.content)
            ## print(f"pg_reply : {pg_reply}")
            if (pg_reply != "NO") :
                combined_context += f"\nContext from Postgres:\n{pg_reply}\n"

            ## print(f"combined_context final: {combined_context}")
            context_msg = SystemMessage(content=f"### Context for this query:\n{combined_context}")
            chat_history.append(context_msg)

            chat_history.append(HumanMessage(content=user_input))
            print("\nAgent thinking...\n")

            result = await agent.ainvoke({"messages": chat_history})
            last_msg = result["messages"][-1]
            reply = last_msg.content if isinstance(last_msg.content, str) else str(last_msg.content)

            print(f"Agent: {reply}\n")
            chat_history.append(AIMessage(content=reply))

        except Exception as err:
            print("Agent error:", type(err).__name__)
            traceback.print_exc()

    print("Chat ended.")

if __name__ == "__main__":
    asyncio.run(run_chat())
