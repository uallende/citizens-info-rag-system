from bs4 import BeautifulSoup
import requests, pdfkit, os

def get_main_sections(soup):
    main_section = soup.find('div', {'class': 'page_content'})
    links = main_section.select('a[href^="/en/"]')
    sub_nav_block = soup.find('div', {'class': 'sub-nav'})
    links = [link['href'] for link in sub_nav_block.select('a[href^="/en/"]')]
    return links

def get_sub_sections(root, main_sections, headers):
    sub_sections = []

    for section in main_sections:
        url = f'{root}{section}'
        page_to_scrape = requests.get(url, headers=headers)
        soup = BeautifulSoup(page_to_scrape.content, 'html.parser')

        topic_section = soup.find("div", {"class": "topic"})
        if topic_section:
            href_links = [a["href"] for a in topic_section.find_all("a", href=True)]
            sub_sections.append(href_links)
        else:
            pass

    # flatten sub_sections
    return [item for sublist in sub_sections for item in sublist]

def scrape_all_links(root, soup, headers):
    links = get_main_sections(soup)
    all_links = []
    while links:
        new_links = []
        for link in links:
            url = f'{root}{link}'
            page_to_scrape = requests.get(url, headers=headers)
            soup = BeautifulSoup(page_to_scrape.content, 'html.parser')
            topic_section = soup.find("div", {"class": "topic"})
            if topic_section:
                href_links = [a["href"] for a in topic_section.find_all("a", href=True)]
                new_links.extend(href_links)
            else:
                pass
            
        links = [link for link in new_links if link not in all_links]
        all_links.extend(links)
    return all_links

root = 'https://www.citizensinformation.ie'
headers = {
    'User-Agent': 'My Scraper Bot (contact: [allende.rev@gmail.com](mailto:allende.rev@gmail.com))'
}

page_to_scrape = requests.get(root, headers=headers)
soup = BeautifulSoup(page_to_scrape.content, 'html.parser')
all_links = scrape_all_links(root, soup, headers)
config = pdfkit.configuration(wkhtmltopdf='/usr/bin/wkhtmltopdf')

# Create the output directory if it doesn't exist
output_dir = '../app/pdf_docs'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for link in all_links:
    url = f'{root}{link}'
    output_file = os.path.join(output_dir, f"{link.replace('/', '_')}.pdf")
    pdfkit.from_url(url, output_file, configuration=config)
    print(f"PDF saved as {output_file}")