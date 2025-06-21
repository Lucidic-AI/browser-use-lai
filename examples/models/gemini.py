import asyncio
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv

load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import SecretStr

from browser_use import Agent
from browser_use.browser import BrowserProfile, BrowserSession

import lucidicai as lai

api_key = os.getenv('GOOGLE_API_KEY')
if not api_key:
	raise ValueError('GOOGLE_API_KEY is not set')

llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash', api_key=SecretStr(api_key))

browser_session = BrowserSession(
	browser_profile=BrowserProfile(
		viewport_expansion=0,
		user_data_dir='~/.config/browseruse/profiles/default',
		highlight_elements=False,
	)
)

task = "On ESPN, find the nba team with the highest average points scored in the current season"

mass_sim_id = "<FILL THIS IN>"

lai.init(
	"Browser Use ESPN NBA top scoring team",
	# lucidic_api_key=os.getenv('LUCIDIC_API_KEY'), -> Set this in .env
	# agent_id=os.getenv('LUCIDIC_AGENT_ID'), -> Set this in .env
	task=task,
	mass_sim_id=mass_sim_id
)

async def run_search():
	agent = Agent(
		task=task,
		llm=llm,
		max_actions_per_step=4,
		browser_session=browser_session,
	)
	handler = lai.LucidicLangchainHandler()
	handler.attach_to_llms(agent)

	await agent.run(max_steps=20)


if __name__ == '__main__':
	asyncio.run(run_search())
