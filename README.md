## Langchain MCP Client

A simple MCP client prototype using Langchain, that can connect to multiple MCP servers.

The current code is configured to use [Jira](https://github.com/harshalmittal4/JiraMCPServer) and [Postgres](https://github.com/harshalmittal4/mcp-postgres).

### Prerequisites
Python 3.11+

### Usage
1. List your MCP servers in the client as a dict, and in the following format -
```
mcp_servers = {
    "postgres": {
      "command": "/Users/harshal.mittal/.pyenv/versions/venv-3.12/bin/python",
      "args": [
        "/Users/harshal.mittal/projects/mcp-postgres/postgres_server.py"
      ],
      "transport": "stdio"
    },
    "jira": {
      "command": "/Users/harshal.mittal/.pyenv/versions/venv-3.12/bin/python",
      "args": [
        "/Users/harshal.mittal/projects/jira-mcp-server/mcp_server.py"
      ],
      "transport": "stdio"
    }
}
```
2. Choose an LLM provider (Anthropic/ Google/ OpenAI/ Azure OpenAI etc), that will be used for client interaction. Configure it in the multi_server_mcp_client.py. 
The current client uses Azure OpenAI.

```
from langchain_openai import AzureChatOpenAI
```
 Copy .env.example to .env, and replace the API keys for your LLM provider in .env file.

3. Create and activate virtual environment, and install dependencies - ```pip install -r requirements.txt```

4. Start the MCP servers, and start the client using ```python multi_server_mcp_client.py```
