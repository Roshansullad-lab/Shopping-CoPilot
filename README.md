# Shopping-CoPilot

Creating virtual python environment on windows

python.exe -m venv work

work\Scripts\activate

work\Scripts\deactivate

Install the required packages using below command

pip install -r requirements.txt

then add you OpenAI API key to the ReturnAisstant.py which can be generated from openAI website.
without this functionality will not work.

Running the FastAPI server which has both client and server built in to it 

uvicorn ReturnAssistant:app --reload

Once server starts we can open any browser and go to the URL.

http://localhost:8000

Then continue with chat prompts still need to train the prompt to update the details correctly.

But as of now we can use prompts like

i want to return laptop

i want to return order no 1008

still report and forecast need to be wired with new LLM flow.
