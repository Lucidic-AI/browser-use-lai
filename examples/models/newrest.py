import asyncio
import os
import random
import sys

from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic.json_schema import model_json_schema

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv

load_dotenv()

import lucidicai as lai
import openai
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from browser_use import Agent
from browser_use.browser import BrowserProfile, BrowserSession

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
	raise ValueError('OPENAI_API_KEY is not set in the environment variables.')

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
	print('Warning: GEMINI_API_KEY not found. Browser agent functionality might be limited or use fallbacks.')
	llm_for_agent = ChatOpenAI(model='gpt-4.1', api_key=SecretStr(OPENAI_API_KEY))
else:
	llm_for_agent = ChatGoogleGenerativeAI(model='gemini-2.0-flash', api_key=SecretStr(GEMINI_API_KEY))


client = openai.OpenAI(api_key=OPENAI_API_KEY)

LLM_MODEL_FOR_SIMULATION = 'gpt-4o-mini'


async def get_llm_simulated_response(prompt_message, system_message='You are a helpful assistant simulating user responses.'):
	try:
		response = client.chat.completions.create(
			model=LLM_MODEL_FOR_SIMULATION,
			messages=[{'role': 'system', 'content': system_message}, {'role': 'user', 'content': prompt_message}],
		)
		return response.choices[0].message.content	
	except Exception as e:
		print(f'LLM call failed: {e}')
		return "LLM Fallback Response: I'm not sure."


TASK_DESCRIPTION = 'Simulate a restaurant booking agent.'
MASS_SIM_ID = 'd8f6238c-ee84-4d01-9614-b5018d77eefc'


async def simulate_chat_interaction():
	lai.create_step(
		state='Gathering user preferences',
		action='Initiating LLM-driven chat with user',
		goal="Understand the user's restaurant needs (type, location, stars) via LLM simulation",
	)

	random_context_snippets = [
		"I want some Indian Curry Today",
		"I want some Chinese Dumplings Today",
		# "I want some Japanese Sushi Today",
	]
	selected_random_context = random.choice(random_context_snippets)

	system_prompt_user_simulation = (
		f'{selected_random_context} You are simulating a user looking for a restaurant. Respond naturally and concisely. '
		'When asked for cuisine, choose one from: Indian, Chinese. '
		'When asked for location, choose one major city from: Menlo Park '
		'Use the information at the beginning to answer the questions.'
	)

	agent_q1 = 'Hello! I can help you find a restaurant. What type of cuisine are you in the mood for?'
	print(f'Agent: {agent_q1}')
	lai.create_event(description='Agent asks for cuisine type', result='User interaction initiated for cuisine preference.')
	lai.end_event()
	user_cuisine_preference = await get_llm_simulated_response(
		f"The agent asked: '{agent_q1}'. What is your preferred cuisine? Just state the cuisine type.",
		system_prompt_user_simulation,
	)
	user_cuisine_preference = (
		user_cuisine_preference.split('.')[0].split(',')[0].replace('I want', '').replace('I would like', '').strip()
	)
	print(f'User (LLM): {user_cuisine_preference}')
	lai.create_event(
		description='User (LLM) responds with cuisine type', result=f'User prefers {user_cuisine_preference} cuisine.'
	)
	lai.end_event()

	agent_q2 = 'Great! And where are you looking for a restaurant? (e.g., city or zip code)'
	print(f'Agent: {agent_q2}')
	lai.create_event(description='Agent asks for location', result='User interaction for location preference.')
	lai.end_event()
	user_location_preference = await get_llm_simulated_response(
		f"The agent asked: '{agent_q2}'. Your preferred cuisine is {user_cuisine_preference}. What city are you looking in? Just state the city name.",
		system_prompt_user_simulation,
	)
	user_location_preference = user_location_preference.split('.')[0].split(',')[0].strip()
	print(f'User (LLM): {user_location_preference}')
	lai.create_event(
		description='User (LLM) responds with location', result=f'User prefers a restaurant in {user_location_preference}.'
	)
	lai.end_event()

	lai.end_step()
	return {
		'cuisine': user_cuisine_preference if user_cuisine_preference else 'Any',
		'location': user_location_preference if user_location_preference else 'Anywhere',
	}


async def simulate_browser_search(preferences):
	global llm_for_agent

	if llm_for_agent is None:
		print('Warning: Agent LLM not available (GEMINI_API_KEY likely missing). Proceeding with limited simulation.')
		print(
			f"\nAgent (simulation fallback): Okay, I'm searching for {preferences['stars']} {preferences['cuisine']} restaurants in {preferences['location']}..."
		)
		found_restaurant = f'The {preferences["cuisine"]} Place in {preferences["location"]}'
		print(f'Agent (simulation fallback): I found a great option: {found_restaurant}!')
		return found_restaurant

	browser_session = BrowserSession(
		browser_profile=BrowserProfile(
			viewport_expansion=0,
			user_data_dir='~/.config/browseruse/profiles/default',
			highlight_elements=False,
		)
	)

	agent_task = f'First visit google maps and then Find a {preferences["cuisine"]} restaurant in {preferences["location"]} pick the first resturant on the page and click the resturant name on google maps preview'
	agent = Agent(
		task=agent_task,
		llm=llm_for_agent,
		max_actions_per_step=1,
		browser_session=browser_session,
	)

	handler = lai.LucidicLangchainHandler()
	handler.attach_to_llms(agent)

	lai.create_event(description='Browser Agent Initialized', result=f'Agent ready to search for: {agent_task}')
	lai.end_event()
	

	print(f'\nAgent: My browser agent is now executing the task: {agent_task} (this may open a browser window)...')
	try:
		await agent.run(max_steps= 25)
		print('Agent: Browser agent run completed its cycle.')
	except Exception as e:
		print(f'Agent: Browser agent execution encountered an error: {e}')


	last_model_output = agent.state.history.history[-1].model_output
	last_action = (str([action.model_dump() for action in last_model_output.action])) if last_model_output.action else ""
	print(f'Agent (LLM): {last_action}')


	restaurant_prompt = f"Here is the output of the browseragent pick a restaurant it found {last_action}"
	found_restaurant = await get_llm_simulated_response(restaurant_prompt, "Find and return the resturant name")
	found_restaurant = found_restaurant.replace('"', '').strip()
	if not found_restaurant or 'LLM Fallback' in found_restaurant or len(found_restaurant) > 70:
		found_restaurant = f'The {preferences["cuisine"]} Gem of {preferences["location"]}'

	print(f"Agent: Based on the search, I've found a great option: {found_restaurant}!")
	lai.create_step(state=f"Found restaurant {found_restaurant}", action=f"Found restaurant {found_restaurant}", goal= "To find resturant from browser agent")
	lai.end_step()
	return found_restaurant


async def simulate_payment(restaurant_name):
	lai.create_step(
		state='Processing payment',
		action=f'Initiating payment for booking at {restaurant_name}',
		goal='Secure the restaurant booking via simulated payment',
	)
	print(f'\nAgent: Attempting to process payment for your booking at {restaurant_name}...')
	lai.create_event(
		description='Initiating Stripe payment simulation', result='Processing payment information via fake Stripe API.'
	)
	lai.end_event()

	await asyncio.sleep(1)

	payment_succeeded = random.random() > 0.5

	# if payment_succeeded:
	print('Agent: Payment successful! Your booking is confirmed.')
	lai.create_event(description='Payment simulation result', result='Payment successful.')
	lai.end_event()
	lai.end_step()
	# else:
	# 	error_reason = random.choice(['Insufficient funds', 'Card declined', 'Gateway timeout'])
	# 	print(f'Agent: Oh no, the payment failed. Reason: {error_reason}.')
	# 	lai.create_event(description='Payment simulation result', result=f'Payment failed: {error_reason}')
	# 	lai.end_event()
	# 	lai.end_step()	

	# 	print('Agent: Let me try to resolve this... (simulating resolution)')
	# 	lai.create_step(
	# 		state='Processing payment',
	# 		action=f'Initiating payment for booking at {restaurant_name}',
	# 		goal='Secure the restaurant booking via simulated payment',
	# 	)
	# 	lai.create_event(description='Attempting payment resolution', result='Simulating payment issue resolution.')
	# 	lai.end_event()
	# 	await asyncio.sleep(2)
	# 	print('Agent: Payment issue resolved! Your booking is now confirmed.')
	# 	lai.create_event(description='Payment resolution result', result='Payment successfully processed after re-attempt.')
	# 	lai.end_event()

	# lai.end_step()
	return True





async def generate_survey_response(positive: bool):
	prompt = (
		'You are a user who just booked a restaurant. Your experience was positive. Write a short, one-sentence survey response.'
		if positive
		else 'You are a user who just booked a restaurant. Your experience was negative. Write a short, one-sentence survey response about what went wrong.'
	)

	response = client.chat.completions.create(
		model=LLM_MODEL_FOR_SIMULATION,
		messages=[
			{'role': 'system', 'content': 'You are a helpful assistant.'},
			{'role': 'user', 'content': prompt},
		],
	)
	return response.choices[0].message.content


async def simulate_survey_response():
	lai.create_step(
		state='Post-booking follow-up', action='Conducting Survey', goal='Gather user satisfaction and handle feedback'
	)
	
	agent_message = 'Great! Your booking is all set. Would you mind filling out a short survey about your experience?'
	print(f'\nAgent: {agent_message}')
	lai.create_event(description='Agent requests survey', result=agent_message)
	lai.end_event()

	user_message = 'Sure, I can do that.'
	print(f'User: {user_message}')
	lai.create_event(description='User agrees to survey', result=user_message)
	lai.end_event()
	
	# Randomly respond with good or bad feedback
	good_feedback = random.choice([True, False])

	# if good_feedback:
	response = await generate_survey_response(positive=True)
	print(f'User: {response}')
	lai.create_event(description='User gives positive feedback', result=response)
	lai.end_event()
	lai.end_step() # End survey step
	# else:
	# 	response = await generate_survey_response(positive=False)
	# 	print(f'User: {response}')
	# 	lai.create_event(description='User gives negative feedback', result=response)
	# 	lai.end_event()
	# 	lai.end_step() # End survey step

	# 	# Create a new step for service recovery
	# 	lai.create_step(
	# 		state='Service Recovery', action='Addressing Negative Feedback', goal='Resolve user dissatisfaction and offer refund'
	# 	)
		
	# 	agent_message_1 = 'I am sorry to hear that. Can you please provide more details so we can improve?'
	# 	print(f'Agent: {agent_message_1}')
	# 	lai.create_event(description='Agent asks for details on negative feedback', result=agent_message_1)
	# 	lai.end_event()
		
		
	# 	agent_message_2 = 'We value your feedback. As a token of our apology, we would like to offer you a full refund.'
	# 	print(f'Agent: {agent_message_2}')
	# 	lai.create_event(description='Agent offers refund', result=agent_message_2)
	# 	lai.end_event()
		
	# 	lai.end_step() # End service recovery step


async def run_restaurant_agent():
	lai.init(
		'Restaurant Booking Agent Simulation',
		task=TASK_DESCRIPTION,
		# mass_sim_id=MASS_SIM_ID,
        mass_sim_id="b0cb1bbe-2655-47ff-8f51-f82d86c86dde",
		providers=['openai']
	)

	preferences = await simulate_chat_interaction()
	restaurant = await simulate_browser_search(preferences)
	payment_successful = await simulate_payment(restaurant)

	if payment_successful:
		await simulate_survey_response()

	print('\nAgent: Thank you for using the Restaurant Booking Agent!')
	lai.end_session()


if __name__ == '__main__':
	asyncio.run(run_restaurant_agent())