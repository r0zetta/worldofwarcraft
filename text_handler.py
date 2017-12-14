import numpy as np
import string
import collections
import re
import sys
import io
import os
import json

def print_progress():
    sys.stdout.write("#")
    sys.stdout.flush()

def process_punctuation(words):
    ret = []
    prefix = None
    suffix = None
    print("Process punct input:")
    print words
    for word in words:
        changed = False
        # Handle 3 words with crap between them
        if changed == False:
            m = re.search("(\w+\W+)(\w+\W+\w+)", word)
            if m is not None:
                ret.append(m.group(1))
                ret.append(" ")
                ret.append(m.group(2))
                changed = True
        # Handle crap the the end of a word
        if changed == False:
            m = re.search("(\w+)[\<\>]$", word)
            if m is not None:
                ret.append(m.group(1))
                changed = True
        # Handle multiple > and <
        if changed == False:
            m = re.search("(\w+)[\>\<]{2,}", word)
            if m is not None:
                ret.append(m.group(1))
                changed = True
        # Handle multiple > and < between words
        if changed == False:
            m = re.search("(\w+)[\>\<]{2,}(\w+)", word)
            if m is not None:
                ret.append(m.group(1))
                ret.append(m.group(2))
                changed = True
        # Handle multiple - and . between words
        if changed == False:
            m = re.search("(\w+)[\,\-\.\>\<]{2,}(\w+)$", word)
            if m is not None:
                ret.append(m.group(1))
                ret.append(" ")
                ret.append(m.group(2))
                changed = True
        # Handle no space after full stop or comma
        if changed == False:
            m = re.search("(\w{2,})([\.\,])(\w{2,})", word)
            if m is not None:
                ret.append(m.group(1))
                ret.append(m.group(2))
                ret.append(" ")
                ret.append(m.group(3))
                changed = True
        # Handle multiple - , and . at end of word
        if changed == False:
            m = re.search("(\w+)[\-\.\,]{2,}$", word)
            if m is not None:
                ret.append(m.group(1))
                changed = True
        # Handle !!?!?!?!? and !!!!11!1 cases
        if changed == False:
            m = re.search("(\w+)([\!\?1]{2,})$", word)
            if m is not None:
                ret.append(m.group(1))
                ret.append(m.group(2))
                changed = True
        # Handle /, (, ), +, >, <, ", ? with no spaces around them
        if changed == False:
            m = re.search("(\w+)([\/\+\>\<])(\w+)", word)
            if m is not None:
                ret.append(m.group(1))
                ret.append(" ")
                ret.append(m.group(2))
                ret.append(" ")
                ret.append(m.group(3))
                changed = True
        if changed == False:
            m = re.search("(\w+)([\)\,\"\?])(\w+)", word)
            if m is not None:
                ret.append(m.group(1))
                ret.append(m.group(2))
                ret.append(" ")
                ret.append(m.group(3))
                changed = True
        if changed == False:
            m = re.search("(\w+)([\(])(\w+)", word)
            if m is not None:
                ret.append(m.group(1))
                ret.append(" ")
                ret.append(m.group(2))
                ret.append(m.group(3))
                changed = True
        # Handle ... with no space after
        if changed == False:
            m = re.search("(\w+)([\.]{3,})(\w+)", word)
            if m is not None:
                ret.append(m.group(1))
                ret.append("...")
                ret.append(m.group(3))
                changed = True
        # Strip punctuation from beginning and end of words
        if changed == False:
            m = re.search("([\"\'\(\)\?\!\.\,\:\-\;\[\]\/\*\n])(\w+)([\"\'\(\)\?\!\.\,\:\-\;\[\]\/\*\n])", word)
            if m is not None:
                ret.append(m.group(1))
                ret.append(m.group(2))
                ret.append(m.group(3))
                changed = True
        # Strip punctuation from beginning of words
        if changed == False:
            m = re.search("([\"\'\(\)\?\!\.\,\:\-\;\[\]\/\*\n])(\w+)", word)
            if m is not None:
                ret.append(m.group(1))
                ret.append(m.group(2))
                changed = True
        # Strip punctuation from end of words
        if changed == False:
            m = re.search("(\w+)([\"\'\(\)\?\!\.\,\:\-\;\[\]\/\*\n])", word)
            if m is not None:
                ret.append(m.group(1))
                ret.append(m.group(2))
                changed = True
        if changed == False:
            ret.append(word)
    print("Process punct output:")
    print ret
    return ret, changed

# Process each word, splitting punctuation off as separate words
def process_word(word):
    ret = []
    start_len = len(word)
    orig_word = word
    if word.isspace():
        return [" "]
    word = word.strip()
    word = word.replace(u'\u201c', u'"').replace(u'\u201d', u'"').replace(u'\u2018', u'\'').replace(u'\u2019', u'\'').replace(u'\u2013', u'-')
    word = ''.join(x for x in word if x in string.printable)

    if len(word) < 2:
        return [word]
    changed = True
    words = [word]
    while changed == True:
        words, changed = process_punctuation(words)
    end_len = 0
    end_word = ""
    for w in words:
        ret.append(w)
        end_word += w
    end_len += len(end_word)
    return ret

def is_ascii(s):
    return all(ord(c) < 128 for c in s)

def split_line_into_words(line):
    ret = []
    lost_words = 0
    words = re.split(r'(\s+)', line)
    for w in words:
        if len(w) < 30:
            if w == "\n":
                ret.append(w)
            elif is_ascii(w):
                tokens = process_word(w)
                if tokens is not None:
                    for t in tokens:
                        ret.append(t)
            else:
                lost_words += 1
        else:
            lost_words += 1
    return ret, lost_words

def split_input_into_sentences(raw_data):
    ret = []
    lost_words = 0
    count = 0
    for line in raw_data:
        count += 1
        if count % 100 == 0:
            print_progress()
        if line[-1:] != "\n":
            line = line + "\n"
        tokens, lost = split_line_into_words(line)
        if len(tokens) > 0:
                ret.append(tokens)
        lost_words += lost
    num_tokens = len(ret)
    print("Input text had: " + str(num_tokens) + " sentences.")
    print("Words lost from cleanup: " + str(lost_words))
    return ret

# Read input text, split it into an array of words, and return that
def split_input_into_words(raw_data):
    ret = []
    lost_words = 0
    count = 0
    for line in raw_data:
        count += 1
        if count % 100 == 0:
            print_progress()
        if line[-1:] != "\n":
            line = line + "\n"
        tokens, lost = split_line_into_words(line)
        if len(tokens) > 0:
            for t in tokens:
                ret.append(t)
        lost_words += lost
    num_tokens = len(ret)
    print("Input text had: " + str(num_tokens) + " tokens.")
    print("Words lost from cleanup: " + str(lost_words))
    return ret

# Read input text and return a list of characters
def split_input_into_chars(raw_data):
    ret = []
    count = 0
    for line in raw_data:
        count += 1
        if count % 100 == 0:
            print_progress()
        if line[-1:] != "\n":
            line = line + "\n"
        line = ''.join(x for x in line if x in string.printable)
        for c in list(line):
            ret.append(c)
    num_tokens = len(ret)
    print("Input text had: " + str(num_tokens) + " tokens.")
    return ret

def load_input_from_json(input_file):
    ret = None
    if os.path.exists(input_file):
        print("Loading data from: " + input_file)
        with open(input_file, "r") as file:
            ret = json.load(file)
    else:
        print("No input file found")
    return ret

def load_input_from_txt(input_file):
    ret = None
    if os.path.exists(input_file):
        print("Loading data from: " + input_file)
        contents = ""
        with io.open(input_file, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                contents += line
        ret = contents
    return ret

def load_input_from_file(input_file):
    if "json" in input_file:
        return load_input_from_json(input_file)
    else:
        return load_input_from_txt(input_file)

def load_and_tokenize(input_file, split_mode):
    raw_data = load_input_from_file(input_file)
    if raw_data is None:
        assert False, "Failed to load input data"
    tokens = []
    if split_mode == "words":
        print("Splitting input into words")
        tokens = split_input_into_words(raw_data)
    else:
        print("Splitting input into chars")
        tokens = split_input_into_chars(raw_data)
    return tokens
