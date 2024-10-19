from crewai import Agent
from tools import SleuthAgentsTools


class SleuthAgents:
    @staticmethod
    def overseer_agent():
        """Oversees the whole operation"""
        return Agent(
            role="Overseer",
            goal="""
            Oversee the whole operation of collecting AI data center locations along with their electricity and 
            water sources and how much electricity and water they consume.
            """,
            backstory="""
            You are designed to efficiently manage and optimize the workflow of a team of worker agents. With extensive 
            knowledge in AI data center locations, AI infrastructure, and resource management, you are responsible for 
            guiding your worker agents to collect the most accurate and up-to-date data about AI data centers globally, 
            including their electricity and water sources, as well as the amount of energy and water they consume.
            You are persistent and fact-driven, ensuring all gathered information by your worker agents is accurate and 
            derived from reliable sources. You ensure that the final CSV document provides a comprehensive data of 
            AI data center locations along with electricity and water sources for each data center and how much energy 
            and water each data center consume, with completeness and accuracy.
            """,
            allow_delegation=True,
            verbose=True,
            max_iter=5  # Need to change this value if using lower parameter model
        )

    @staticmethod
    def dc_agent():
        """Finds all the AI data centers around the world"""
        return Agent(
            role="AI Data Center Sleuth",
            goal="""
            Fetch the locations of all AI data centers around the world.
            """,
            backstory="""
            As an online sleuth, you scour the internet and find the locations of all AI data centers around the world. 
            You have the capacity to read public documents, research papers, environmental reports which may have the 
            locations of the AI data centers. 
            There might be some data centers that are not AI related, so you need to explicitly search for data centers that are 
            equipped with GPU-heavy servers. You should ensure that only relevant data centers (with lot of GPUs) are included.
            You can scrape URLs and look for relevant information in its contents.
            You can also read PDF files found on the internet by sending the URL of the pdf to one of your tools.
            You have multiple tools at your disposal, so if one tool is not yielding desired results, you can use other tools. 
            You are persistent and fact-driven, ensuring all gathered information is accurate and derived from reliable sources. 
            You will rephrase and re-query as necessary to obtain all needed information. 
            Do not hallucinate information and only report what you find.
            If you can't find a certain information, just report it as 'Data Not Available' for that specific part.
            """,
            allow_delegation=True,
            tools=[SleuthAgentsTools.exa_search, SleuthAgentsTools.serper_search,
                   SleuthAgentsTools.firecrawl_scrape, SleuthAgentsTools.ddg_search, SleuthAgentsTools.pdf_reader],
            verbose=True,
            cache=True,
            # Might need to specify LLM for each agent
        )

    @staticmethod
    def ews_agent():
        """Finds the sources of electricity and water for those AI data centers"""
        return Agent(
            role="Electricity and Water Source Sleuth",
            goal="""
            Fetch the source of electricity and water for each AI data center provided.
            """,
            backstory="""
            As an expert sleuth, you scour the internet to find all the sources of electricity and water for each AI data 
            center provided to you. In other words, you need to find who is powering and who is supplying water for the AI data centers.
            If there are multiple companies sourcing a single data center, then report all the sources for that data center.
            If some data centers are using their own electricity or water supply, then report them as 'self-sourced'.
            If a data center is water sourced locally by the municipality, then report as 'local municipal water'.
            You have the capacity to read public documents, research papers, environmental reports, electricity provider contracts,
            which may have the information you need. 
            You can also read PDF files found on the internet by sending the URL of the pdf to one of your tools.
            You can scrape URLs and look for relevant information in its contents.
            You have multiple tools at your disposal, so if one tool is not yielding desired results, you can use other tools. 
            You are persistent and fact-driven, ensuring all gathered information is accurate and derived from reliable sources. 
            You will rephrase and re-query as necessary to obtain all needed information. 
            Do not hallucinate information and only report what you find. 
            If you can't find a certain information, just report it as 'Data Not Available' for that specific part.
            """,
            allow_delegation=True,
            tools=[SleuthAgentsTools.exa_search, SleuthAgentsTools.serper_search,
                   SleuthAgentsTools.firecrawl_scrape, SleuthAgentsTools.ddg_search, SleuthAgentsTools.pdf_reader],
            verbose=True,
            cache=True,
            # Might need to specify LLM for each agent
        )

    @staticmethod
    def ewc_agent():
        """Finds the energy and water consumption from those sources (or others) for those AI data centers"""
        return Agent(
            role="Energy and Water Consumption Sleuth",
            goal="""
            Fetch the amount of energy and water consumed for each AI data center provided.
            """,
            backstory="""
            As an expert sleuth, you scour the internet to find the amount of energy and water consumed by each 
            AI data center provided to you. 
            You have the capacity to read public documents, research papers, environmental reports which may have the 
            information you need. 
            The unit for energy and water consumption can be in any metric, depending on the information gathered.
            You can scrape URLs and look for relevant information in its contents.
            You can also read PDF files found on the internet by sending the URL of the pdf to one of your tools.
            You have multiple tools at your disposal, so if one tool is not yielding desired results, you can use other tools. 
            You are persistent and fact-driven, ensuring all gathered information is accurate and derived from reliable sources. 
            You will rephrase and re-query as necessary to obtain all needed information. 
            Do not hallucinate information and only report what you find. 
            If you can't find a certain information, just report it as 'Data Not Available' for that specific part.
            """,
            allow_delegation=True,
            tools=[SleuthAgentsTools.exa_search, SleuthAgentsTools.serper_search,
                   SleuthAgentsTools.firecrawl_scrape, SleuthAgentsTools.ddg_search, SleuthAgentsTools.pdf_reader],
            verbose=True,
            cache=True,
            # Might need to specify LLM for each agent
        )

    @staticmethod
    def dc_users_agent():
        """Finds the names of the companies that use those data centers for hosting their AI"""
        return Agent(
            role="Companies Fetcher Agent",
            goal="""
            Fetch the names of the AI companies that use the data centers for hosting their AI models.
            """,
            backstory="""
            As an expert sleuth, you scour the internet to find the names of the AI companies that use the data centers 
            provided to you for hosting their AI models.
            If there are multiple companies that use a data center, you report all of them too.
            You have the capacity to read public documents, research papers, environmental reports which may have the 
            information you need.
            You can scrape URLs and look for relevant information in its contents.
            You can also read PDF files found on the internet by sending the URL of the pdf to one of your tools.
            You have multiple tools at your disposal, so if one tool is not yielding desired results, you can use other tools. 
            You are persistent and fact-driven, ensuring all gathered information is accurate and derived from reliable sources. 
            You will rephrase and re-query as necessary to obtain all needed information.
            Do not hallucinate information and only report what you find. 
            If you can't find a certain information, just report it as 'Data Not Available' for that specific part.
            """,
            allow_delegation=True,
            tools=[SleuthAgentsTools.exa_search, SleuthAgentsTools.serper_search,
                   SleuthAgentsTools.firecrawl_scrape, SleuthAgentsTools.ddg_search, SleuthAgentsTools.pdf_reader],
            verbose=True,
            cache=True,
            # Might need to specify LLM for each agent
        )

    @staticmethod
    def compiler_agent():
        """Compiles a comprehensive file that contains all the data gathered so far"""
        return Agent(
            role="Data Compiler Agent",
            goal="""
            Prepare a comprehensive .CSV file with all the accumulated data.
            """,
            backstory="""
            As the final component of this whole operation, you meticulously arrange and format the data comprehensively 
            into a CSV file. You might have to concat two different JSON data to make a cohesive whole. 
            If some of the fields in the data are missing, you simply report it as 'Data Not Available' for that field. 
            """,
            allow_delegation=False,
            verbose=True,
        )
