# from crewai_tools import PDFSearchTool
from langchain_community.document_loaders import PyPDFLoader
# from langchain_community.tools import DuckDuckGoSearchRun
# from langchain_core.tools import Tool
# from langchain.utilities import GoogleSearchAPIWrapper
from crewai_tools import tool, SerperDevTool, EXASearchTool
from firecrawl import FirecrawlApp
from serpapi import GoogleSearch
# from exa_py import Exa
# from langchain.tools import tool
# from pydantic import BaseModel
import os


class SleuthAgentsTools:

    # Web Scraper
    @staticmethod
    @tool('WebScraperTool')
    def firecrawl_scrape(url: str):
        """To scrape a website and return its contents. Use this tool if you want to go to a URL and extract its
        contents, so you can go through it and grab relevant information."""
        print("Website scraping....")
        app = FirecrawlApp(api_key=os.getenv("FIRECRAWL_API_KEY"))
        return app.scrape_url(url)

    # PDF READER TOOL
    @staticmethod
    @tool('PDFReaderTool')
    def pdf_reader(url: str):
        """To read any PDF files found on the internet.
        This tool will take the URL of the pdf as its argument. Only one pdf can be read at once.
        This tool needs to be called multiple times if there are multiple pdfs to read from the internet."""
        print("PDF reading....")
        loader = PyPDFLoader(url, extract_images=False)
        doc = loader.lazy_load()
        pages = ""
        for page in doc:
            # Extract the content of the current page
            content = page.page_content
            # Check for the keyword that indicates the start of the references section
            if "References" in content or "Bibliography" in content:
                continue
            # Concatenate the current page content to the 'pages' string
            pages += content
        return pages

    ###########  SEARCH TOOLS  ##############
    # Langchain Google API Search
    # @staticmethod
    # def lang_goog_search():  # Unused
    #     search = GoogleSearchAPIWrapper(k=25)
    #     lg_tool = Tool(
    #         name="GoogleSearch",
    #         description="Search the internet using Google for information.",
    #         func=search.run,
    #         num_results=25
    #     )
    #     return lg_tool

    # Google Serper API Search
    @staticmethod
    @tool('GoogleSerperSearch')
    def serper_search(query: str):
        """Search the internet using Google Serper for information and return relevant results."""
        print("Serper searching....")
        serp_tool = SerperDevTool(
            search_url="https://google.serper.dev/search",
            n_results=25,
        )
        return serp_tool.run(search_query=query)

    # EXA Search
    @staticmethod
    @tool("Exa Search")
    def exa_search(query: str):
        """Tool using Exa's Python SDK to run semantic search and return relevant URLs."""
        print("Exa searching....")
        exa_tool = EXASearchTool()
        exa_tool.n_results = 2
        return exa_tool.run(search_query=query)

    # DuckDuckGo Search powered by Serpapi
    @staticmethod
    @tool('DuckDuckGoSearch')
    def ddg_search(query: str):
        """Search the internet using DuckDuckGo for information and return relevant results."""
        print("DuckDuckGo searching....")
        params = {
            "engine": "duckduckgo",
            "q": query,
            "kl": "us-en",
            "api_key": os.getenv("SERPAPI_API_KEY")
        }
        search = GoogleSearch(params)
        results = search.get_dict()
        fin_result = {}
        # Extract "organic_results" if available
        if "organic_results" in results:
            fin_result["organic_results"] = results["organic_results"]
        # Extract "news_results" if available
        if "news_results" in results:
            fin_result["news_results"] = results["news_results"]
        # Extract "knowledge_graph" if available
        if "knowledge_graph" in results:
            fin_result["knowledge_graph"] = results["knowledge_graph"]
        # organic_results = results["organic_results"]
        # return DuckDuckGoSearchRun().run(query)
        return fin_result

    # CSV CALLBACK











