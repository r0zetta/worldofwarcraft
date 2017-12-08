import numpy as np
import collections
import re
import io
import os
import json

punct = ["\"", "\'", "(", ")", "?", "!", ".", ",", ":", "-", ";", "[", "]", "/", "*", "\n"]

def process_punctuation(words):
    ret = []
    prefix = None
    suffix = None
    changed = False
    for word in words:
        # Handle multiple - and . between words
        m = re.search("(\w+)[\-\.]{2,}(\w+)$", word)
        if m is not None:
            #print "[0]: " + word
            return [m.group(1), m.group(2)], True
        # Handle no space after full stop or comma
        m = re.search("(\w{2,})[\.\,](\w{2,})", word)
        if m is not None:
            #print "[1]: " + word
            return [m.group(1), ".", m.group(2)], True
        # Handle multiple - , and . at end of word
        m = re.search("(.+)[\-\.\,]{2,}$", word)
        if m is not None:
            #print "[2]: " + word
            return [m.group(1)], True
        # Handle !!?!?!?!? and !!!!11!1 cases
        m = re.search("(.+)([\!\?1]{2,}$)", word)
        if m is not None:
            #print "[3]: " + word
            return [m.group(1), m.group(2)], True
        # Handle /, (, ), +, >, <, ", ? with no spaces around them
        m = re.search("(\w+)([\/\)\(\+\>\<\,\"\?])(\w+)", word)
        if m is not None:
            #print "[4]: " + word
            return [m.group(1), m.group(2), m.group(3)], True
        # Handle ... with no space after
        m = re.search("(\w+)([\.]{3,})(\w+)", word)
        if m is not None:
            #print "[5]: " + word
            return [m.group(1), m.group(2), m.group(3)], True
        # Strip punctuation from beginning and end of words
        for p in punct:
            if len(word) > 1:
                if word.startswith(p):
                    prefix = p
                    word = word[1:]
                    changed = True
        for p in punct:
            if len(word) > 1:
                if word.endswith(p):
                    suffix = p
                    word = word[:-1]
                    changed = True
        if prefix is not None:
            ret.append(prefix)
        ret.append(word)
        if suffix is not None:
            ret.append(suffix)
    return ret, changed

# Process each word, splitting punctuation off as separate words
def process_word(word):
    ret = []
    word = word.strip()
    if len(word) < 2:
        return [word]
    if "http" in word:
        return "_URL_"
    if "@" in word:
        return "_EMAIL_"
    if word.startswith("multimedia:"):
        return None
    changed = True
    words = [word]
    while changed == True:
        words, changed = process_punctuation(words)
    for word in words:
        ret.append(word)
    return ret

def is_ascii(s):
    return all(ord(c) < 128 for c in s)

def split_line_into_words(line):
    ret = []
    lost_words = 0
    words = line.split()
    for w in words:
        if len(w) < 30:
            if is_ascii(w):
                tokens = process_word(w)
                if tokens is not None:
                    for t in tokens:
                        ret.append(t)
            else:
                lost_words += 1
        else:
            lost_words += 1
    return ret, lost_words

# Read input text, split it into an array of words, and return that
def split_input_into_words(raw_data):
    ret = []
    lost_words = 0
    print("Splitting input into words.")
    for line in raw_data:
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
    print("Splitting input into chars.")
    for line in raw_data:
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
    print("Loading data from " + input_file)
    raw_data = load_input_from_file()
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
