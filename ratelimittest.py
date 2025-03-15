import lucidicai as lai
import asyncio
import httpx
from lucidicai import Client
from dotenv import load_dotenv

# Modify this with your actual test endpoint
NUM_REQUESTS = 100  # Total number of requests to send
CONCURRENT_WORKERS = 50  # Number of concurrent requests at a time

async def make_request(client, i):
    try:
        response_json = Client().make_request('verifyapikey', 'GET', {})
        print(f"[{i}] Status: {response_json}")
    except Exception as e:
        print(f"[{i}] Error: {e}")

async def main():
    async with httpx.AsyncClient(timeout=10.0) as client:
        tasks = []
        for i in range(NUM_REQUESTS):
            task = make_request(client, i)
            tasks.append(task)
            # Limit concurrent tasks
            if len(tasks) >= CONCURRENT_WORKERS:
                await asyncio.gather(*tasks)
                tasks = []
        # Run any remaining tasks
        if tasks:
            await asyncio.gather(*tasks)

if __name__ == "__main__":
    load_dotenv()
    lai.init("lmao")
    asyncio.run(main())
