from selenium import webdriver
from bs4 import BeautifulSoup
import os
import requests
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

options = Options()
options.add_argument('--headless')
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')

driver = webdriver.Chrome(options=options)


# Base URL for the website
base_url = "https://www.citizensinformation.ie/en/"

# Folder to save PDF files
pdf_folder = "pdf_docs"

# Create the folder if it doesn't exist
if not os.path.exists(pdf_folder):
    os.makedirs(pdf_folder)

# Function to extract article titles and links from a page
def extract_articles(url):
    driver.get(url)
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    articles = []
    for article in soup.find_all('div', {'class': 'article'}):
        title = article.find('h2').text.strip()
        link = article.find('a')['href']
        articles.append({'title': title, 'link': base_url + link})
    return articles

# Function to download and save a PDF file
def download_pdf(url, title):
    response = requests.get(url)
    print(response)
    if response.headers['Content-Type'] == 'application/pdf':
        pdf_file = f"{pdf_folder}/{title}.pdf"
        with open(pdf_file, 'wb') as f:
            f.write(response.content)
        print(f"Saved {title} to {pdf_file}")
    else:
        print(f"Error: {title} is not a PDF file")

# Start at the home page and extract articles
articles = extract_articles(base_url)

# Loop through each article and download the PDF file
for article in articles:
    print(f"Downloading {article['title']}")
    download_pdf(article['link'], article['title'])

# Close the Selenium webdriver
driver.quit()