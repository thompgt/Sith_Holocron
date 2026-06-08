import requests
from bs4 import BeautifulSoup
import json
import os

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
    
    os.makedirs("data/raw", exist_ok=True)
    with open("data/raw/wookieepedia_lore.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    
    print(f"Scraped {len(results)} pages and saved to data/raw/wookieepedia_lore.json")

if __name__ == "__main__":
    main()
