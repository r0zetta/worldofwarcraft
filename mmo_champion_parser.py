from bs4 import BeautifulSoup
import os
import io
import re
import requests
import sys
import json
import time

# URLs for the General and Class Development US forums on battle.net
save_dir = "mmo_champion_data"
start_urls = ["https://www.mmo-champion.com/forums/266-General-Discussions"]
num_pages_to_visit = 3
visited_urls = []

def dump_data(var, name):
    filename = os.path.join(save_dir, name)
    with open(filename, "w") as file:
        json.dump(var, file, indent=4)

def load_data(name):
    filename = os.path.join(save_dir, name)
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
    base_domain = "https://www.mmo-champion.com/"
    soup = get_page_source(url)
    threads = soup.find('ol', class_="threads")
    threadlist = threads.find_all('li', class_="threadbit")
    for t in threadlist:
        context = {}
        p = t.find('a', class_="title")
        link = p.get('href')
        if link is not None:
            context["link"] = base_domain + link
        title = p.get_text().strip()
        if title is not None:
            m = re.search("Sticky\:\s.+", title)
            if m is not None:
                print("Skipping sticky post: " + title)
                return None, 0
            context["title"] = title
        s = t.find("ul", class_="threadstats td alt")
        if s is not None:
            stats = t.get_text().strip()
            context["posts"] = 0
            m = re.search("Replies:\s([0-9\,]+)", stats)
            if m is not None:
                posts = m.group(1)
                pc = int(posts.replace(",", ""))
                pc += 1
                context["posts"] = pc
                post_count += pc
        ret.append(context)
        print("Got thread: " + context["title"] + " with " + str(context["posts"]) + " posts.")
    return ret, post_count

# This function looks for a next button on a forum thread page and returns it, if found
def get_next_button(soup):
    next_url = None
    prev_next = soup.find_all("a", rel="next")
    if prev_next is not None:
        for p in prev_next:
            nu = p.get('href')
            m = re.search("\/page[0-9]+$", nu)
            if m is not None:
                next_url = nu
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
    posts = soup.find_all("li", class_="postbitlegacy")
    for p in posts:
        post = {}
        nodecontrols = p.find("span", class_="nodecontrols")
        if nodecontrols is not None:
            link = nodecontrols.find("a")
            if link is not None:
                post_id = link.get("name")
                m = re.search("post([0-9]+)", post_id)
                if m is not None:
                    post["post_id"] = m.group(1)
        username = p.find("a", class_="username")
        if username is not None:
            name_url = username.get("href")
            m = re.search("^members\/([0-9]+)\-(\w+)$", name_url)
            if m is not None:
                user_id = m.group(1)
                author = m.group(2)
                post["author_name"] = author
                post["author_id"] = user_id
        else:
            post["author_name"] = "guest"
            post["author_id"] = "66666666"
        content = p.find("div", class_="content")
        if content is not None:
            quote_container = content.find("div", class_="quote_container")
            if quote_container is not None:
                quote_url = quote_container.find("a")
                if quote_url is not None:
                    u = quote_url.get("href")
                    m = re.search("^showthread.php\?p=([0-9]+)\#post([0-9]+)$", u)
                    if m is not None:
                        replied_id = m.group(1)
                        post["replied_to"] = replied_id
                quote_text = quote_container.find("div", class_="message")
                if quote_text is not None:
                    t = quote_text.get_text().strip()
                    post["replied_to_text"] = t
            t = content.get_text().strip()
            if "replied_to_text" in post:
                if len(post["replied_to_text"]) > 0:
                    t = t.replace(post["replied_to_text"], "")
            post["post_content"] = t
        ret.append(post)
    return ret, next_button

def get_threads(url_list, count):
    threads = []
    post_count_total = 0
    for url in url_list:
        for page in range(count):
            page_url = url + "/page" + str(page + 1)
            print("Getting threads from URL: " + page_url)
            new_threads, post_count = process_threadlist(page_url)
            if new_threads is not None:
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
    test_urls = ["https://www.mmo-champion.com/threads/2355838-To-all-the-delusional-tanks"]
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
        if "author_name" in p:
            if p["author_name"] not in ret:
                ret.append(p["author_name"])
    return ret

# Main routine
# Assembles URLs, processes them, writes output
if __name__ == '__main__':
    print("Starting")
    #test()
    if not os.path.exists(save_dir):
        print("Creating save dir")
        os.makedirs(save_dir)
    threads, post_count = get_threads(start_urls, num_pages_to_visit)
    thread_total = len(threads)
    print("Enumerated a total of: " + str(thread_total) + " threads, with " + str(post_count) + " posts.")
    dump_data(threads, "threads.json")
    thread_count = 1
    post_count_total = 0
    all_authors = []
    all_posts = []
    all_conversations = []
    all_threads = []
    for t in threads:
        url = t["link"]
        title = t["title"]
        all_threads_item = {}
        all_threads_item["title"] = title
        all_threads_item["link"] = url
        print("Thread [" + str(thread_count) + "/" + str(thread_total) + "] Posts [" + str(post_count_total) + "/" + str(post_count) + "] " + title)
        thread_count += 1
        thread_posts = scrape_thread(url)
        if len(thread_posts) > 0:
            thread_post_count = len(thread_posts)
            all_threads_item["count"] = thread_post_count
            post_count_total += thread_post_count
            all_posts += thread_posts
            conv = organize_conversation(thread_posts)
            all_threads_item["posts"] = thread_posts
            all_threads_item["conversations"] = conv
            all_conversations += conv
            just_text = get_just_text(all_posts)
            all_threads_item["text"] = just_text
            authors = get_authors(all_posts)
            all_threads_item["authors"] = authors
            all_threads.append(all_threads_item)
            dump_data(authors, "authors.json")
            dump_data(all_conversations, "conv.json")
            dump_data(just_text, "data.json")
            dump_data(all_posts, "full_data.json")
            dump_data(visited_urls, "visited.json")
            dump_data(all_threads, "all_threads.json")
    print("Done collecting")

