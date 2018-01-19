from bs4 import BeautifulSoup
import os
import io
import re
import requests
import sys
import json
import time

save_dir = "reddit_data"
#start_urls = ["https://www.reddit.com/r/wow/"]
start_urls = ["https://www.reddit.com/r/The_Donald/"]
num_pages_to_visit = 2
visited_urls = []

def dump_gephi_file():
    outfile = os.path.join(save_dir, "interactions.csv")
    handle = io.open(outfile, "w", encoding="utf-8")
    handle.write(u"Source,Target\n")
    for n, l in interactions.iteritems():
        for x in l:
            if x != "guest" and x != n:
                handle.write(n + u"," + x + u"\n")
    handle.close

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
    headers = {'User-Agent': 'User-Agent: python parser for generating gephi graphs:v1 (by /u/r0zetta)'}

    retries = 0
    while True:
        try:
            r = requests.get(url, headers=headers, timeout=30)
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
    print("Processing URL: " + url)
    ret = []
    post_count = 0
    base_domain = "https://www.reddit.com"
    soup = get_page_source(url)
    next_button = get_next_button_thread_list_page(soup)
    print next_button
    for p in soup.find_all('div', class_="thing"):
        context = {}
        context["author"] = p["data-author"]
        context["posts"] = p["data-comments-count"]
        post_count += int(context["posts"])
        title_block = p.find("a", class_="title")
        context["title"] = title_block.get_text().strip()
        link = p["data-permalink"]
        if link is not None:
            context["link"] = base_domain + link
        ret.append(context)
    return ret, post_count, next_button

# This function looks for a next button on a thread list page and returns it, if found
def get_next_button_thread_list_page(soup):
    next_url = None
    next_button = soup.find('div', class_="nav-buttons")
    if next_button is not None:
        url = next_button.find("a")
        if url is not None:
           next_url = url.get('href')
    return next_url

def recurse_comments(outer_comment, parent_metadata, post_data, posts_collected):
    global interactions
    post = {}
    try:
        post_id = outer_comment["id"]
    except:
        #print("ERROR: No post id found")
        return post_data
    #print("Post id: " + post_id)
    if post_id not in posts_collected:
        posts_collected.append(post_id)
        post["post_id"] = post_id
        try:
            post["author_id"] = outer_comment["data-author-fullname"]
        except:
            post["author_id"] = "66666666"
        #print("Author id: " + post["author_id"])
        try:
            post["author_name"] = outer_comment["data-author"]
        except:
            post["author_name"] = "guest"
        #print("Author name: " + post["author_name"])
        if parent_metadata is not None:
            if "post_id" in parent_metadata:
                post["replied_to"] = parent_metadata["post_id"]
            if "author_name" in parent_metadata:
                replied_to = parent_metadata["author_name"]
                if post["author_name"] not in interactions:
                    interactions[post["author_name"]] = []
                if replied_to not in interactions[post["author_name"]]:
                    interactions[post["author_name"]].append(replied_to)

        text_area = outer_comment.find("div", class_="usertext-body")
        if text_area is not None:
            post["post_content"] = text_area.get_text().strip()
        else:
            post["post_content"] = ""
        post_data.append(post)
        inner_comments = outer_comment.find_all("div", class_="thing")
        if inner_comments is not None and len(inner_comments) > 0:
            for c in inner_comments:
                if c is not None:
                    recurse_comments(c, post, post_data, posts_collected)
    return post_data

# Scrape posts from each page of a thread
# This function looks for a "next" button and returns the sub-url of it, if found
# It also returns a list of all posts found on the page
def process_forum_topic(url):
    ret = []
    soup = get_page_source(url)
    if soup is None:
        return None, None
    comments = soup.find("div", class_="commentarea")
    outer_comments = comments.find_all('div', class_="thing")
    print("Found " + str(len(outer_comments)) + " outer comments.")
    post_data = []
    posts_collected = []
    for x, c in enumerate(outer_comments):
        #print("Outer comment: " + str(x))
        new_data = recurse_comments(c, None, post_data, posts_collected)
        for d in new_data:
            if d not in ret:
                ret.append(d)
    print("Got " + str(len(ret)) + " posts.")
    return ret

def get_threads(url_list, count):
    threads = []
    post_count_total = 0
    for url in url_list:
        next_url = url
        for page in range(count):
            page_url = next_url
            print("Getting threads from URL: " + page_url)
            new_threads, post_count, next_url = process_threadlist(page_url)
            print("Next URL: " + next_url)
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
    first_post_author = posts[0]["author_name"]
    replied_ids = {}
    replied_authors = {}
    posts_map = {}
    replied_ids[first_post_id] = []
    post_id_to_author = {}
    for p in posts:
        if "replied_to" in p:
            replied_ids[p["replied_to"]] = []
        posts_map[p["post_id"]] = p["post_content"]
        post_id_to_author[p["post_id"]] = p["author_name"]
    for p in posts:
        post_author = p["author_name"]
        replied_post_id = ""
        if "replied_to" in p:
            replied_ids[p["replied_to"]].append(p["post_id"])
            replied_post_id = p["replied_to"]
        else:
            replied_ids[first_post_id].append(p["post_id"])
            replied_post_id = first_post_id
        replied_to_author = ""
        if replied_post_id in post_id_to_author:
            replied_to_author = post_id_to_author[replied_post_id]
        elif first_post_id in post_id_to_author:
            replied_to_author = first_post_id
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
    next_url = url
    print("Getting posts for url: " + next_url)
    current_posts = process_forum_topic(next_url)
    if current_posts is not None:
        post_count += len(current_posts)
        thread_posts += current_posts
    print("New posts collected so far in this thread: " + str(post_count))
    return thread_posts

def test():
    test_urls = ["https://www.reddit.com/r/wow/comments/7qgr8m/500k_subscribers_giveaway_party_thread/", "https://www.reddit.com/r/wow/comments/7rftj4/firepower_friday_your_weekly_dps_thread/", "https://www.reddit.com/r/wow/comments/7rdpul/this_weeks_affixes/"]
    #test_urls = ["https://www.reddit.com/r/wow/comments/7rdpul/this_weeks_affixes/"]
    posts = []
    conv = []
    for t in test_urls:
        posts += scrape_thread(t)
        dump_data(posts, "test.json")
        conv += organize_conversation(posts)
        dump_data(conv, "test_conv.json")
    dump_data(interactions, "interactions.json")
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
    interactions = {}
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
            dump_gephi_file()
            dump_data(interactions, "interactions.json")
            dump_data(authors, "authors.json")
            dump_data(all_conversations, "conv.json")
            dump_data(just_text, "data.json")
            dump_data(all_posts, "full_data.json")
            dump_data(visited_urls, "visited.json")
            dump_data(all_threads, "all_threads.json")
    print("Done collecting")


