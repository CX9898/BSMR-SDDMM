import os
import tarfile
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

BASE_URL = 'https://sparse.tamu.edu/'

def get_matrix_links(collection='HB'):
    url = f'{BASE_URL}{collection}'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    links = set()  # 使用集合来去重
    for row in soup.find_all('tr'):
        for link in row.find_all('a', href=True):
            if link['href'].endswith('.mtx.gz') or link['href'].endswith('.tar.gz'):
                full_link = urljoin(BASE_URL, link['href'])
                links.add(full_link)

    return list(links)

def download_matrix(url, save_path='matrices'):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    filename = url.split('/')[-1]
    file_path = os.path.join(save_path, filename)

    # 如果文件已存在，跳过下载
    if os.path.exists(file_path):
        print(f"File already exists, skipping: {filename}")
        return

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    response = requests.get(url, headers=headers)
    if response.status_code == 200 and 'text/html' not in response.headers.get('Content-Type', ''):
        with open(file_path, 'wb') as f:
            f.write(response.content)
        print(f'Downloaded: {filename}')
    else:
        print(f"Failed to download {filename}. Status code: {response.status_code}")

    if file_path.endswith('.tar.gz'):
        with tarfile.open(file_path, 'r:gz') as tar:
            tar.extractall(path=save_path)
            print(f'Extracted: {filename}')

def main():
    collection = input("Enter the collection name (e.g., HB, SNAP, UF): ").strip()
    links = get_matrix_links(collection)

    if not links:
        print("No matrices found.")
        return

    for url in links:
        download_matrix(url)

if __name__ == '__main__':
    main()
