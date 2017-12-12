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

def dump_data(var, name):
    filename = "conv_data/" + name
    with open(filename, "w") as file:
        json.dump(var, file, indent=4)

def load_data(name):
    filename = "conv_data/" + name
    if os.path.exists(filename):
        with open(filename, "r") as file:
            return json.load(file)

# Attempt to fetch URL, handling timeout, and retrying
def fetch_url(url):
    retries = 0
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
        if retries > 5:
            return None
        print("Retrying in 10 seconds.")
        time.sleep(10)
        retries += 1

# This function loads a URL and throws it into BeautifulSoup for parsing
# This function returns a BeautifulSoup object
def get_page_source(url):
    global visited_urls
    page = fetch_url(url)
    if page is not None:
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
    post_count = 0
    base_domain = "https://us.battle.net"
    soup = get_page_source(url)
    for p in soup.find_all('a', class_="ForumTopic"):
        context = {}
        t = p.find("span", class_="ForumTopic-title")
        context["title"] = t.get_text().strip()
        a = p.find("span", class_="ForumTopic-author")
        context["author"] = a.get_text().strip()
        dft = json.loads(p["data-forum-topic"])
        if "lastPosition" in dft:
            context["posts"] = dft["lastPosition"]
            post_count += dft["lastPosition"]
            if dft["lastPosition"] == 0:
                continue
        link = p.get('href')
        if link is not None:
            context["link"] = base_domain + link
        ret.append(context)
    return ret, post_count

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
    if soup is None:
        return None, None
    # Get "next" link, if it exists
    next_button = get_next_button(soup)
    full_info = soup.find_all("div", class_="TopicPost")
    for info in full_info:
        post = {}
        dtp = json.loads(info["data-topic-post"])
        post["post_id"] = dtp["id"]
        post["author_name"] = dtp["author"]["name"]
        post["author_id"] = dtp["author"]["id"]
        content = info.find("div", class_="TopicPost-bodyContent")
        quote = content.find("blockquote", class_="quote-public")
        if quote is not None:
            post["replied_to"] = quote["data-quote"]
        quote = content.find(class_="quote-blizzard")
        if quote is not None:
            post["replied_to"] = quote["data-quote"]
        for c in content:
            if c.string is None:
                c.string = " "
        post["post_content"] = content.get_text().strip()
        ret.append(post)
    return ret, next_button

def get_threads(url_list, count):
    threads = []
    post_count_total = 0
    for url in url_list:
        for page in range(count):
            page_url = url + "?page=" + str(page + 1)
            print("Getting threads from URL: " + page_url)
            new_threads, post_count = process_threadlist(page_url)
            threads += new_threads
            thread_total = len(threads)
            post_count_total += post_count
            print("Got: " + str(thread_total) + " threads, containing " + str(post_count_total) + " posts.")
            dump_data(threads, "threads.json")
    return threads, post_count_total

def organize_conversation(posts):
    conv = []
    first_post = posts[0]["post_content"]
    first_post_id = posts[0]["post_id"]
    replied_ids = {}
    posts_map = {}
    replied_ids[first_post_id] = []
    for p in posts:
        if "replied_to" in p:
            replied_ids[p["replied_to"]] = []
        posts_map[p["post_id"]] = p["post_content"]
    for p in posts:
        if "replied_to" in p:
            replied_ids[p["replied_to"]].append(p["post_id"])
        else:
            replied_ids[first_post_id].append(p["post_id"])
    for key, values in replied_ids.iteritems():
        if key in posts_map:
            question = posts_map[key]
            for v in values:
                if v in posts_map:
                    answer = posts_map[v]
                    if question != answer:
                        qa = [question, answer]
                        conv.append(qa)
    return conv

def scrape_thread(url):
    next_button = ""
    sub_pages = 0
    post_count = 0
    thread_posts = []
    while next_button is not None:
        next_url = url + next_button
        print("Getting posts for url: " + next_url)
        current_posts, next_button = process_forum_topic(next_url)
        if current_posts is not None:
            post_count += len(current_posts)
            thread_posts += current_posts
        print("New posts collected so far in this thread: " + str(post_count))
        if next_button is not None:
            sub_pages += 1
    return thread_posts

def test():
    #test_urls = ["https://us.battle.net/forums/en/wow/topic/20759616677", "https://us.battle.net/forums/en/wow/topic/20759478451"]
    test_urls = ["https://us.battle.net/forums/en/wow/topic/20759478451"]
    posts = []
    conv = []
    for t in test_urls:
        posts += scrape_thread(t)
        conv += organize_conversation(posts)
    dump_data(posts, "test.json")
    dump_data(conv, "test_conv.json")
    sys.exit(0)

def get_just_text(all_posts):
    ret = []
    for p in all_posts:
        if p["post_content"] not in ret:
            ret.append(p["post_content"])
    return ret

def get_authors(all_posts):
    ret = []
    for p in all_posts:
        if p["author_name"] not in ret:
            ret.append(p["author_name"])
    return ret

# Main routine
# Assembles URLs, processes them, writes output
if __name__ == '__main__':
    print("Starting")
    #test()
    if not os.path.exists("conv_data"):
        print("Creating data dir")
        os.makedirs("conv_data")
    threads, post_count = get_threads(start_urls, num_pages_to_visit)
    thread_total = len(threads)
    print("Enumerated a total of: " + str(thread_total) + " threads, with " + str(post_count) + " posts.")
    dump_data(threads, "threads.json")
    thread_count = 1
    post_count_total = 0
    all_authors = []
    all_posts = []
    all_conversations = []
    for t in threads:
        url = t["link"]
        title = t["title"]
        print("Thread [" + str(thread_count) + "/" + str(thread_total) + "] Posts [" + str(post_count_total) + "/" + str(post_count) + "] " + title)
        thread_count += 1
        thread_posts = scrape_thread(url)
        if len(thread_posts) > 0:
            post_count_total += len(thread_posts)
            all_posts += thread_posts
            conv = organize_conversation(thread_posts)
            all_conversations += conv
            just_text = get_just_text(all_posts)
            authors = get_authors(all_posts)
            dump_data(authors, "authors.json")
            dump_data(all_conversations, "conv.json")
            dump_data(just_text, "data.json")
            dump_data(all_posts, "full_data.json")
            dump_data(visited_urls, "visited.json")
    print("Done collecting")


