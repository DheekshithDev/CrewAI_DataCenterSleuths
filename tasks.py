from crewai import Task
from tools import SleuthAgentsTools


class SleuthAgentsTasks:
    @staticmethod
    def fetch_dc_task(agent):
        """Fetch data centers task for dc_finder_agent"""
        return Task(
            description="""
            Fetch the locations of all AI data centers around the world. 
            """,
            agent=agent,
            async_execution=False,
            expected_output="""
            A list of AI data centers along with companies that own each of them.
                Example Output: 
                [
                    {  "datacenter": 'xAI', 
                    "locations": ['Memphis']
                    },
                    {  "datacenter": 'Microsoft Azure', 
                    "locations": ['North Central US', 'Chicago']
                    }, 
                    {{...}}
                ]
            """,
            tools=[SleuthAgentsTools.exa_search, SleuthAgentsTools.serper_search,
                   SleuthAgentsTools.firecrawl_scrape, SleuthAgentsTools.ddg_search, SleuthAgentsTools.pdf_reader]
        )

    @staticmethod
    def fetch_ews_task(agent, context):
        """Fetch electricity and water sources task for ews_agent"""
        return Task(
            description="""
            Fetch who is powering and who is supplying water for the AI data centers.
            """,
            agent=agent,
            async_execution=False,
            context=context,
            expected_output="""
            A JSON list of AI data centers with their respective sources of electricity and water supply appended to 
            previous task's JSON output.
                Example Output: 
                [
                    {  "datacenter": 'xAI', 
                    "locations": ['Memphis'],
                    "electricity_sources": ['self-sourced'],
                    "water_sources": ['Data Not Available']
                    },
                    {  "datacenter": 'Microsoft Azure', 
                    "locations": ['North Central US', 'Chicago'],
                    "electricity_sources": ['ENGIE', 'Sun Streams 2 Solar Project-Longroad Energy'],
                    "water_sources": ['local municipal water', 'self-sourced']
                    }, 
                    {{...}}
                ]
            """,
            tools=[SleuthAgentsTools.exa_search, SleuthAgentsTools.serper_search,
                   SleuthAgentsTools.firecrawl_scrape, SleuthAgentsTools.ddg_search, SleuthAgentsTools.pdf_reader]
        )

    @staticmethod
    def fetch_ewc_task(agent, context):
        """To find the energy and water consumption for ewc_agent"""
        return Task(
            description="""
            Get the amount of consumption of energy and water for AI data centers.
            """,
            agent=agent,
            async_execution=False,
            context=context,
            expected_output="""
            A JSON list of AI data centers with their respective energy and water consumption appended to the
            previous task's JSON output.
                Example Output: 
                [
                    {  "datacenter": 'xAI', 
                    "locations": ['Memphis'],
                    "electricity_sources": ['self-sourced'],
                    "water_sources": ['Data Not Available'],
                    "energy_consumption": '120 MW',
                    "water_consumption": '2M liters per day'
                    },
                    {  "datacenter": 'Microsoft Azure', 
                    "locations": ['North Central US', 'Chicago'],
                    "electricity_sources": ['ENGIE', 'Sun Streams 2 Solar Project-Longroad Energy'],
                    "water_sources": ['local municipal water', 'self-sourced'],
                    "energy_consumption": '85 MW',
                    "water_consumption": '1.5M liters per week'
                    }, 
                    {{...}}
                ]
            """,
            tools=[SleuthAgentsTools.exa_search, SleuthAgentsTools.serper_search,
                   SleuthAgentsTools.firecrawl_scrape, SleuthAgentsTools.ddg_search, SleuthAgentsTools.pdf_reader]
        )

    @staticmethod
    def dc_users_task(agent, context):
        """To find the names of the companies for dc_users_agent"""
        return Task(
            description="""
            Get the names of the AI companies that use the data center for hosting their AI models.
            """,
            agent=agent,
            async_execution=False,
            context=context,
            expected_output="""
            A JSON list of AI data centers with AI companies that use the data center for their AI model hosting.
                Example Output: 
                [
                    {  "datacenter": 'xAI', 
                    "locations": ['Memphis'],
                    "companies": ['Data Not Available']
                    },
                    {  "datacenter": 'Microsoft Azure', 
                    "locations": ['North Central US', 'Chicago'],
                    "companies": ['OpenAI', 'Bloom', 'AlphaGo']
                    }, 
                    {{...}}
                ]
            """,
            tools=[SleuthAgentsTools.exa_search, SleuthAgentsTools.serper_search,
                   SleuthAgentsTools.firecrawl_scrape, SleuthAgentsTools.ddg_search, SleuthAgentsTools.pdf_reader]
        )

    @staticmethod
    def compile_task(agent, context):
        """Compile final output file for compiler_agent"""
        return Task(
            description="""
            Create a comprehensive CSV file with all the data gathered. 
            """,
            agent=agent,
            context=context,
            expected_output="""
            A complete comprehensive data including all AI data center locations, their sources of electricity and water, 
            and the amount of energy and water consumption by them in CSV format.
            The data entry should have the following fields: datacenter, location, electricity_source, water_source, 
            energy_consumption, water_consumption, companies.
            """,
            # callback=callback_function,
            output_file="Data3.csv"
        )
