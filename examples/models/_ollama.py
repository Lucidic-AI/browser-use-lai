# Optional: Disable telemetry
# os.environ["ANONYMIZED_TELEMETRY"] = "false"

# Optional: Set the OLLAMA host to a remote server
# os.environ["OLLAMA_HOST"] = "http://x.x.x.x:11434"

import asyncio
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv

load_dotenv()

from langchain_ollama import ChatOllama

from browser_use import Agent
from browser_use.agent.views import AgentHistoryList

import lucidicai as lai

lai.init("Reddit Search for Browseruse")


async def run_search() -> AgentHistoryList:
	agent = Agent(
		task="Search for a 'browser use' post on the r/LocalLLaMA subreddit and open it.",
		llm=ChatOllama(
			model='qwen2.5:32b-instruct-q4_K_M',
			num_ctx=32000,
		),
	)
	handler = lai.LucidicLangchainHandler()
	handler.attach_to_llms(agent)

	result = await agent.run()
	lai.end_session()
	return result


async def main():
	result = await run_search()
	print('\n\n', result)


if __name__ == '__main__':
	asyncio.run(main())
