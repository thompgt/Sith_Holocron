import json
import os
import sys

import requests
from bs4 import BeautifulSoup

#: The path src/main.py and scripts/synthesize_dataset.py actually read. This
#: script used to write data/raw/wookieepedia_lore.json, which nothing loads --
#: so a successful re-scrape left the stale lore in place and reported success.
LORE_PATH = "data/raw/lore.json"


def scrape_fandom_page(url):
    print(f"Scraping {url}...")
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        print(f"Failed to retrieve {url} (Status Code: {response.status_code})")
        return None

    soup = BeautifulSoup(response.content, 'html.parser')

    # Remove unwanted elements
    for script in soup(["script", "style", "aside"]):
        script.extract()

    content = soup.find(id="mw-content-text")
    if not content:
        return None

    # Extract text from paragraphs
    paragraphs = content.find_all('p')
    text = "\n".join([p.get_text().strip() for p in paragraphs if p.get_text().strip()])

    return {
        "url": url,
        "title": soup.find(id="firstHeading").get_text() if soup.find(id="firstHeading") else url,
        "content": text
    }

def main():
    urls = [
        "https://starwars.fandom.com/wiki/Sith",
        "https://starwars.fandom.com/wiki/Code_of_the_Sith",
        "https://starwars.fandom.com/wiki/Sith_philosophy",
        "https://starwars.fandom.com/wiki/Rule_of_Two",
        "https://starwars.fandom.com/wiki/Protocol_droid",
        "https://starwars.fandom.com/wiki/C-3PO",
        "https://starwars.fandom.com/wiki/Darth_Sidious",
        "https://starwars.fandom.com/wiki/Darth_Vader"
    ]

    results = []
    for url in urls:
        data = scrape_fandom_page(url)
        if data:
            results.append(data)

    # Fandom blocks scrapers periodically and every page failure is swallowed
    # above, so an empty result set is a realistic outcome -- it is what produced
    # the stray 2-byte wookieepedia_lore.json in this repo. Overwriting the
    # corpus with [] would silently empty the lore half of the index, so refuse.
    if not results:
        print(
            f"Scraped 0 of {len(urls)} pages; refusing to overwrite {LORE_PATH}.",
            file=sys.stderr,
        )
        return 1

    if len(results) < len(urls):
        print(
            f"WARNING: only {len(results)} of {len(urls)} pages scraped.",
            file=sys.stderr,
        )

    os.makedirs("data/raw", exist_ok=True)
    with open(LORE_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    print(f"Scraped {len(results)} pages and saved to {LORE_PATH}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
