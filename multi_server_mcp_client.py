import os
import getpass
import asyncio
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain.schema import HumanMessage, AIMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
import sys

load_dotenv()

if not os.getenv("AZURE_OPENAI_API_KEY"):
    os.environ["AZURE_OPENAI_API_KEY"] = getpass.getpass("Enter your Azure OpenAI API key: ")
if not os.getenv("AZURE_OPENAI_ENDPOINT"):
    os.environ["AZURE_OPENAI_ENDPOINT"] = getpass.getpass("Enter your Azure OpenAI endpoint: ")

async def run_chat():
    mcp_servers = {
    "postgres": {"command": "/Users/harshal.mittal/.pyenv/versions/venv-3.12/bin/python", "args": ["/Users/harshal.mittal/projects/mcp-postgres/postgres_server.py"], "transport": "stdio"},
    "jira": {"command": "/Users/harshal.mittal/.pyenv/versions/venv-3.12/bin/python", "args": ["/Users/harshal.mittal/projects/jira-mcp-server/mcp_server.py"], "transport": "stdio"}
}

    client = MultiServerMCPClient(mcp_servers)
    tools = await client.get_tools()
    print("Available tools:")
    for tool in tools:
        print("-", tool.name)

    model = AzureChatOpenAI(
        azure_deployment="gpt-4o",  # or your deployment
        api_version="2023-07-01-preview",  # or your api version
    )

    # Use it with create_react_agent
    agent = create_react_agent(model, tools)

    print("Agent is ready. Type your message or 'exit' to quit.\n")

    chat_history = []

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

            chat_history.append(HumanMessage(content=user_input))
            print("\nAgent thinking...\n")

            result = await agent.ainvoke({"messages": chat_history})
            last_msg = result["messages"][-1]
            reply = last_msg.content if isinstance(last_msg.content, str) else str(last_msg.content)

            print(f"Agent: {reply}\n")
            chat_history.append(AIMessage(content=reply))

        except Exception as err:
            print(f"Agent error: {err}")

    await client.aclose()
    print("Chat ended.")

if __name__ == "__main__":
    asyncio.run(run_chat())
