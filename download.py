import os
import requests
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup  # Install BeautifulSoup using: pip install beautifulsoup4

def download_file(url, local_path):
    response = requests.get(url)
    if response.status_code == 200:
        with open(local_path, 'wb') as file:
            file.write(response.content)
        print(f"File downloaded successfully to {local_path}")
    else:
        print(f"Failed to download file. Status code: {response.status_code}")

def download_all_files(base_url, local_directory):
    response = requests.get(base_url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        for link in soup.find_all('a', href=True):
            file_url = urljoin(base_url, link['href'])
            file_name = os.path.join(local_directory, os.path.basename(urlparse(file_url).path))
            download_file(file_url, file_name)
    else:
        print(f"Failed to fetch page. Status code: {response.status_code}")

# Example usage:
web_base_url = 'https://www.pythonanywhere.com/user/hbnnarzullayev/files/home/hbnnarzullayev/mysite/'
local_directory = 'D:/HOME/'

# Ensure the local directory exists
os.makedirs(local_directory, exist_ok=True)

# Download all files from the web to the local directory
download_all_files(web_base_url, local_directory)
