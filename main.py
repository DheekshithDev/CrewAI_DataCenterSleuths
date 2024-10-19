from crewai import Crew, Process, LLM
from dotenv import load_dotenv
import os
# from langchain_nvidia_ai_endpoints import ChatNVIDIA
# from langchain_groq import ChatGroq
from agents import SleuthAgents
from tasks import SleuthAgentsTasks
from tools import SleuthAgentsTools


load_dotenv()

# # Llama 3.2 90B served by GroqCloud
# client_groq_90b = ChatGroq(
#     api_key=os.getenv("GROQ_API_KEY"),
#     model="llama-3.2-90b-text-preview",
#     temperature=0.75,
#     top_p=0.9,
#     max_tokens=3500
# )
#
# # Llama 3.1 70B served by GroqCloud
# client_groq_70b = ChatGroq(
#     api_key=os.getenv("GROQ_API_KEY"),
#     model="llama-3.1-70b-versatile",
#     temperature=0.75,
#     top_p=0.9,
#     max_tokens=3500
# )

# Llama 3.1 405B served by Nvidia NIM
client_nvidia_405b = LLM(
    base_url="https://integrate.api.nvidia.com/v1",
    model="meta/llama-3.1-405b-instruct",
    api_key=os.getenv("OPENAI_API_KEY"),
    # Might need to remove these 3 values coz I'm using LLM() not ChatNVIDIA()
    temperature=0.80,
    top_p=0.9,
    max_tokens=8192,
)


# Instantiate agents
overseer = SleuthAgents.overseer_agent()  # M
dc_finder = SleuthAgents.dc_agent()  # 1
ews_finder = SleuthAgents.ews_agent()  # 2
ewc_finder = SleuthAgents.ewc_agent()  # 3
dc_users_finder = SleuthAgents.dc_users_agent()  # 4
compiler = SleuthAgents.compiler_agent()  # C

# Instantiate tasks
dc_task = SleuthAgentsTasks.fetch_dc_task(agent=dc_finder)  # 1
ews_task = SleuthAgentsTasks.fetch_ews_task(agent=ews_finder, context=[dc_task])  # 2
ewc_task = SleuthAgentsTasks.fetch_ewc_task(agent=ewc_finder, context=[ews_task])  # 3
dc_users_task = SleuthAgentsTasks.dc_users_task(agent=dc_users_finder, context=[dc_task])  # 4  # Async-OFF
compile_task = SleuthAgentsTasks.compile_task(agent=compiler, context=[ewc_task, dc_users_task])  # C

# Establishing the crew with a hierarchical process and additional configurations
crew = Crew(
    agents=[dc_finder, ews_finder, ewc_finder, dc_users_finder, compiler],
    tasks=[dc_task, ews_task, ewc_task, dc_users_task, compile_task],  # Tasks to be delegated and executed under the manager's supervision
    manager_llm=client_nvidia_405b,  # Mandatory if manager_agent is not set
    process=Process.hierarchical,  # Specifies the hierarchical management approach
    respect_context_window=True,  # Enable respect of the context window for tasks
    memory=True,  # Enable memory usage for enhanced task execution
    manager_agent=overseer,  # Optional: explicitly set a specific agent as manager instead of the manager_llm
    planning=True,  # Enable planning feature for pre-execution strategy,
    verbose=True,
    max_rpm=25
)

# Kick off the crew's work
results = crew.kickoff()

# Print the results
print("Crew Work Results:::::::::::::::::: \n")
print(results)

# LLMs
# Llama3.1
# ChatGPT3.5



