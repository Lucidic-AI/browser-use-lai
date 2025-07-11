"""
Simple try of the agent.

@dev You need to add OPENAI_API_KEY to your environment variables.
"""

import asyncio
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI

from browser_use import Agent

import lucidicai as lai

llm = ChatOpenAI(model='gpt-4o')
agent = Agent(
	task='Go to amazon.com, search for laptop, sort by best rating, and give me the price of the first result',
	llm=llm,
)

lai.init("Amazon search")


async def main():
	handler = lai.LucidicLangchainHandler()
	handler.attach_to_llms(agent)
	await agent.run(max_steps=10)
	input('Press Enter to continue...')
	lai.end_session()


asyncio.run(main())
