import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import time

seed_urls = [
    "https://iitj.ac.in/",
    "https://www.iitj.ac.in/bioscience-bioengineering",
    "https://www.iitj.ac.in/computer-science-engineering/",
    "https://www.iitj.ac.in/mathematics/",
    "https://www.iitj.ac.in/physics/",
    "https://www.iitj.ac.in/computer-science-engineering/en/Research-Archive"
]

base_domain = "iitj.ac.in"

all_text = ""

max_per_seed = 6   # pages per seed

for seed in seed_urls:
    visited = set()
    to_visit = [seed]

    print(f"\nStarting crawl for: {seed}")

    while to_visit and len(visited) < max_per_seed:
        url = to_visit.pop(0)

        if url in visited:
            continue

        try:
            res = requests.get(url, timeout=5)
            soup = BeautifulSoup(res.text, "html.parser")

            print("Scraping:", url)

            # extract text
            page_text = soup.get_text(separator=" ")
            all_text += page_text + "\n"

            visited.add(url)

            # extract links
            for link in soup.find_all("a", href=True):
                full_url = urljoin(url, link['href'])

                if base_domain in full_url and full_url not in visited:
                    if any(keyword in full_url.lower() for keyword in [
                        "academics", "research", "department", "faculty", "program", "course"
                    ]):
                        if full_url not in to_visit:
                            to_visit.append(full_url)

        except Exception as e:
            print("Error:", url)
            continue

        time.sleep(1)  # polite crawling

# save raw data
with open("raw.txt", "w", encoding="utf-8") as f:
    f.write(all_text)

print("\nDone scraping.")
