from bs4 import BeautifulSoup
import os
import io
import re
import requests
import sys
import json
import time

# URLs for the General and Class Development US forums on battle.net
start_urls = ["https://us.battle.net/forums/en/wow/22814068/", "https://us.battle.net/forums/en/wow/984270/"]
num_pages_to_visit = 10
visited_urls = []

# Attempt to fetch URL, handling timeout, and retrying
def fetch_url(url):
    while True:
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            return r
        except requests.exceptions.HTTPError as eh:
            print ("HTTP Error: ", eh)
        except requests.exceptions.ConnectionError as ec:
            print ("Connection Error: ", ec)
        except requests.exceptions.Timeout as et:
            print ("Timeout: ", et)
        except requests.exceptions.RequestException as ef:
            print ("Fatal Error: ", ef)
            break
        print("Retrying in 10 seconds.")
        time.sleep(10)

# This function loads a URL and throws it into BeautifulSoup for parsing
# This function returns a BeautifulSoup object
def get_page_source(url):
    global visited_urls
    page = fetch_url(url)
    if url not in visited_urls:
        visited_urls.append(url)
    html_code = page.content
    soup = BeautifulSoup(html_code, "lxml")
    return soup

# Iterates the list of posts on each forum main page
# This returns a list of URLs found
# Note that it skips blue "sticky" posts found at the top of each page
def process_threadlist(url):
    ret = []
    base_domain = "https://us.battle.net"
    soup = get_page_source(url)
    for p in soup.find_all('a', class_="ForumTopic"):
        # Skip forum stickies
        if "ForumTopic--sticky" in p["class"]:
            print "Found sticky post. Skipping"
        else:
            link = p.get('href')
            if link is not None:
                link = base_domain + link
                if link not in ret:
                    ret.append(link)
    return ret

# This function looks for a next button on a forum thread page and returns it, if found
def get_next_button(soup):
    next_url = None
    next_button = soup.find('a', class_="Pagination-button Pagination-button--next")
    if next_button is not None:
        next_url = next_button.get('href')
    return next_url

# Scrape posts from each page of a thread
# This function looks for a "next" button and returns the sub-url of it, if found
# It also returns a list of all posts found on the page
def process_forum_topic(url):
    ret = []
    soup = get_page_source(url)
    # Get "next" link, if it exists
    next_button = get_next_button(soup)
    posts = soup.find_all("div", class_="TopicPost-bodyContent")
    if posts is not None:
        for p in posts:
            # Skip quote
            if p.find(class_="quote-public"):
                continue
            # Skip blue quote
            if p.find(class_="quote-blizzard"):
                continue
            # This bit of code makes sure spaces are handled correctly
            for r in p:
                if r.string is None:
                    r.string = " "
            t = p.get_text()
            # And this bit of code makes sure there aren't extraneous spaces
            ' '.join(t.split())
            t = t.strip()
            if t is not None:
                ret.append(t)
    return ret, next_button

# Serialize all collected posts to disk, so that if we need to restart,
# we can pick up from where we left
def dump_data(posts):
    filename = "data/data.json"
    print("Writing data file: " + filename)
    with open(filename, "w") as file:
        json.dump(posts, file, indent=4)

# Serialize the list of URLs visited to disk, so that if we need to restart,
# we don't visit them again
def dump_visited():
    filename = "data/visited.json"
    print("Writing data file: " + filename)
    with open(filename, "w") as file:
        json.dump(visited_urls, file, indent=4)

# This will be used as the training set for our model
def dump_raw(posts):
    filename = "data/input.txt"
    print("Writing data file: " + filename)
    with io.open(filename, "w", encoding="utf-8") as handle:
        for p in posts:
            handle.write(p)
            handle.write(u"\n")

# Main routine
# Assembles URLs, processes them, writes output
if __name__ == '__main__':
    print("Starting")
    posts = []
    if not os.path.exists("data"):
        print("Creating data dir")
        os.makedirs("data")
    if os.path.exists("data/visited.json"):
        print("Loading visited.json")
        with open("data/visited.json", "r") as file:
            visited_urls = json.load(file)
    if os.path.exists("data/data.json"):
        print("Loading data.json")
        with open("data/data.json", "r") as file:
            posts = json.load(file)
    for url in start_urls:
        for page in range(num_pages_to_visit):
            page_url = url + "?page=" + str(page + 1)
            print("Getting threads from URL: " + page_url)
            thread_urls = process_threadlist(page_url)
            print("Got: " + str(len(thread_urls)) + " thread URLs.")
            thread_count = 1
            for u in thread_urls:
                if u in visited_urls:
                    print("URL: " + u + " has already been scraped. Skipping.")
                else:
                    next_button = ""
                    sub_pages = 0
                    post_count = 0
                    print("[" + str(thread_count) + "]: " + u)
                    thread_count += 1
                    while next_button is not None:
                        next_url = u + next_button
                        print("Getting posts for url: " + next_url)
                        current_posts, next_button = process_forum_topic(next_url)
                        if current_posts is not None:
                            for c in current_posts:
                                if c not in posts:
                                    post_count += 1
                                    posts.append(c)
                                else:
                                    print("Post already seen. Skipping")
                        print("New posts collected so far in this thread: " + str(post_count))
                        if next_button is not None:
                            sub_pages += 1
                    print("For URL: " + u)
                    print("Sub pages: " + str(sub_pages))
                    print("Posts: " + str(post_count))
                    dump_data(posts)
                    dump_visited()
                    dump_raw(posts)
    print("Done collecting")


