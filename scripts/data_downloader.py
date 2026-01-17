import aiohttp
import asyncio
import certifi
import os
import ssl
from urllib.parse import parse_qs, urlparse

from quant_sandbox.common import get_base_dir

ssl_context = ssl.create_default_context(cafile=certifi.where())


async def download_file(download_dir, session, url, semaphore):
    parsed_url = urlparse(url)
    match parsed_url.hostname:
        case "fred.stlouisfed.org":
            params = parse_qs(parsed_url.query)
            filename = params["id"][0] + ".csv"
            download_dir = os.path.join(download_dir, parsed_url.hostname)
            filepath = os.path.join(download_dir, filename)
        case _:
            filename = url.split("/")[-1] or f"file_{hash(url)}"
            filepath = os.path.join(download_dir, filename)
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)
    async with semaphore:
        try:
            async with session.get(url, ssl=ssl_context) as response:
                if response.status == 200:
                    content = await response.read()
                    with open(filepath, "wb") as f:
                        f.write(content)
                    print(f"Successfully downloaded: {filename} from {parsed_url.hostname}")
                else:
                    print(f"Failed to download {url}: Status {response.status}")
        except Exception as e:
            print(f"Error downloading {url}: {e}")


async def main(download_dir, urls, concurrency=5):
    semaphore = asyncio.Semaphore(concurrency)
    async with aiohttp.ClientSession() as session:
        tasks = [download_file(download_dir, session, url, semaphore) for url in urls]
        await asyncio.gather(*tasks)


if __name__ == '__main__':
    base_dir = get_base_dir()
    urls_to_fetch = [
        # Brave-Butters-Kelley Real Gross Domestic Product
        "https://fred.stlouisfed.org/graph/fredgraph.csv?id=BBKMGDP",
        # Brave-Butters-Kelley Leading Index
        "https://fred.stlouisfed.org/graph/fredgraph.csv?id=BBKMLEIX",
        # Sticky Price Consumer Price Index less Food and Energy
        "https://fred.stlouisfed.org/graph/fredgraph.csv?id=CORESTICKM159SFRBATL",
        # Federal Funds Effective Rate
        "https://fred.stlouisfed.org/graph/fredgraph.csv?id=FEDFUNDS",
        # Real Gross Domestic Product
        "https://fred.stlouisfed.org/graph/fredgraph.csv?id=GDPC1",
        # Real M2 Money Stock
        "https://fred.stlouisfed.org/graph/fredgraph.csv?id=M2REAL",
        # University of Michigan: Inflation Expectation
        "https://fred.stlouisfed.org/graph/fredgraph.csv?id=MICH",
        # All Employees, Total Nonfarm
        "https://fred.stlouisfed.org/graph/fredgraph.csv?id=PAYEMS",
        # Personal Consumption Expenditures
        "https://fred.stlouisfed.org/graph/fredgraph.csv?id=PCE",
        # Unemployment Rate - Job Losers
        "https://fred.stlouisfed.org/graph/fredgraph.csv?id=U2RATE",
        # Total Unemployed, Plus All Persons Marginally Attached to the Labor Force, Plus Total Employed Part Time
        # for Economic Reasons, as a Percent of the Civilian Labor Force Plus All Persons Marginally Attached to the
        # Labor Force
        "https://fred.stlouisfed.org/graph/fredgraph.csv?id=U6RATE",
        # Average Weeks Unemployed
        "https://fred.stlouisfed.org/graph/fredgraph.csv?id=UEMPMEAN",
        # University of Michigan: Consumer Sentiment
        "https://fred.stlouisfed.org/graph/fredgraph.csv?id=UMCSENT",
        # Unemployment Rate
        "https://fred.stlouisfed.org/graph/fredgraph.csv?id=UNRATE",
    ]

    asyncio.run(main(os.path.join(base_dir, 'data'), urls_to_fetch))
