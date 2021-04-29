import requests
import os
from tqdm import tqdm
from bs4 import BeautifulSoup as bs
from urllib.parse import urljoin, urlparse
import re
import sys


def is_valid(url):
    """
    Checks whether `url` is a valid URL.
    """
    parsed = urlparse(url)
    return bool(parsed.netloc) and bool(parsed.scheme)

def get_all_images(url):
    """
    Returns all image URLs on a single `url`
    """
    soup = bs(requests.get(url).content, "html.parser")

    urls = soup.find_all("a", href=True)
    #print (urls)
    urls = [ elem["href"] for elem in urls if re.search (".+(jpg|png|jpeg)", elem["href"]) ]# and is_valid (elem["href"]) ]

    base_url = re.search("^(?:https?:\/\/)?(?:[^@\n]+@)?(?:www\.)?([^:\/\n?]+)", url).group(0)

    urls = [ elem.strip("\r\n\t ") for elem in urls ]
    urls = [ elem if elem[0] != "/" else base_url+elem for elem in urls ]

    #print (urls)
    return urls

def download(url, pathname):
    """
    Downloads a file given an URL and puts it in the folder `pathname`
    """
    # if path doesn't exist, make that path dir
    if not os.path.isdir(pathname):
        os.makedirs(pathname)
    # download the body of response by chunk, not immediately
    response = requests.get(url, stream=True)
    # get the total file size
    file_size = int(response.headers.get("Content-Length", 0))
    # get the file name
    filename = os.path.join(pathname, url.split("/")[-1])
    # progress bar, changing the unit to bytes instead of iteration (default by tqdm)
    progress = tqdm(response.iter_content(1024), f"Downloading {filename}", total=file_size, unit="B", unit_scale=True, unit_divisor=1024)
    with open(filename, "wb") as f:
        for data in progress:
            # write data read to the file
            f.write(data)
            # update the progress bar manually
            progress.update(len(data))


def get_sources():
    with open("image_sources.txt", "r") as f:
        sources = f.readlines()
        sources = [ source.strip () for source in sources ]
        sources = [ source for source in sources if source[0] != "#" ]
    return sources

path = "data"

print (get_sources())

#download("http://archive.wizards.com/dnd/images/mapofweek/June2007/01_June2007_72_fc2wdj_ppi.jpg", path)
#download("http://archive.wizards.com/leaving.asp?url=/dnd/images/mapofweek/April2007/04_Apr2007_72_ppi_54hge.jpg&origin=dnd_mw_20070404arch", path)

for url in get_sources():
    #break
    if not is_valid(url):
        print ("Link not valid: {0}".format (url))
        continue
    # get all images
    print ("Looking at {0}".format (url))
    imgs = get_all_images(url)
    print ("{0} images found".format (len(imgs)))

    for img in imgs:
        # for each image, download it
        #print(repr(img))
        download(img, path)
        #break
