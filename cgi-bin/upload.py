#!/usr/bin/env python3

import cgi
import cgitb
import datetime
import json
import os
import pandas as pd
import polars as pl
import regex
import spacy
from spacy.matcher import Matcher
import subprocess
import sys


cgitb.enable()  # Show detailed error messages in the browser


def print_html_header():
    print("Content-Type: text/html\n")
    print("""
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <title>Processing results</title>
</head>
<body>
<p><a href="/enriching">Home</a></p>
""")


def make_patterns(entities_path, entities_filename):
    patterns = {}
    if entities_path != "":
        try:
            entities_df = pl.read_csv(entities_path, encoding="utf-8")
        except:
            print(f"ERROR. Could not read file {entities_filename}. Is it a .csv file?")
            sys.exit(1)
        for row in entities_df.iter_rows():
            entity_label = row[0]
            entity_text = row[1]
            if entity_label not in patterns:
                patterns[entity_label] = []
            patterns[entity_label].append([{"TEXT": token} for token in entity_text.strip().split()])
    return patterns


def check_entity_overlap(entity_list, entity_start, entity_end):
    for entity in entity_list:
        if entity_start < entity[1] and entity_end > entity[0]:
            return True
    return False


def print_label(entity_label):
    if entity_label in ["PER", "PERSON"]:
        print(f"<font color=\"red\"><strong>{entity_label}</strong></font>", end="")
    elif entity_label in ["GPE", "LOC", "FAC"]:
        print(f"<font color=\"green\"><strong>{entity_label}</strong></font>", end="")
    else:
        print(f"<font color=\"black\"><strong>{entity_label}</strong></font>", end="")


def process_response_string(response_string):
    entity_label = ""
    for token in response_string.split():
        if token in ["LOC", "PER"]:
            if entity_label != "":
                print("; ")
            entity_label = token
            print_label(entity_label)
            print(":", end="")
        elif entity_label != "" and not regex.search("^\d+$", token):
            print(" ", token)


def process_form_data():
    form = cgi.FieldStorage()
    file_item = form["file"] if "file" in form else None
    column_names = [column_name.strip() for column_name in form.getfirst("column", "").strip().split(",")]
    method = form.getfirst("method", "").strip()
    try:
        max_processed = int(form.getfirst("max_processed", "").strip())
    except:
        print("ERROR. Please go back and specify number of cells to process")
        sys.exit(1)
    entities_file = form["entities_file"] if "entities_file" in form else None
    if entities_file is None or entities_file == "" or os.path.basename(entities_file.filename) == "":
        entities_path = ""
    else:
        entities_filename = os.path.basename(entities_file.filename)
        entities_path = f"/tmp/{entities_filename}"
        with open(entities_path, "wb") as f:
            f.write(entities_file.file.read())
            f.close()
    
    if "file" not in form or not form["file"].filename:
        print("ERROR. No file uploaded")
        sys.exit(1)
    elif len(column_names) == 0:
        print("ERROR. Column name not provided")
        sys.exit(1)
    
    file_name = os.path.basename(file_item.filename)
    file_path = f"/tmp/{file_name}"
    
    with open(file_path, "wb") as f:
        f.write(file_item.file.read())
        f.close()
    return file_name, file_path, column_names, max_processed, method, entities_filename, entities_path


def read_data_file(file_name, file_path, column_names):
    try:
        data_df = pl.read_csv(file_path, encoding="utf-8")
    except:
        print(f"ERROR. Could not read file {file_name}. Is it a .csv file?")
        sys.exit(1)
    if len(data_df.columns) == 0:
        print(f"ERROR. File {file_name} has np columns")
        sys.exit(1)
    for column_name in column_names:
        if column_name not in data_df.columns:
            print(f"ERROR. Cannot find column with name {column_name} in file {file_name}. The available columns are: {sorted(list(data_df.columns))}")
            sys.exit(1)
    return data_df


def show_processing_parameters(file_path, column_names, max_processed, method, entities_path):
    print(f"<p>File saved as: {file_path}</p>")
    print(f"<p>Column(s) of interest: <strong>{column_names}</strong></p>")
    print(f"<p>Processing the first {max_processed} cells</p>")
    print(f"<p>System: {method}</p>")
    print(f"<p>Entities file: {entities_path}</p>")


def setup_spacy(entities_path, entities_filename):
    nlp = spacy.load("en_core_web_trf", disable=["tagger", "parser", "lemmatizer"])
    matcher = Matcher(nlp.vocab)
    patterns = make_patterns(entities_path, entities_filename)
    for entity_label in patterns:
        matcher.add(entity_label, patterns[entity_label])
    return nlp, matcher


llama_prompt = "Can you find the person names and locations in this text and list them in combination with the character offset with respect to the beginning of the text? Please mention only the found strings in the format: type offset string, for example: PER 23 John and LOC 45 Paris. Here is the text:"


def process_with_llama(llama_prompt, text):
    result = subprocess.run("curl http://localhost:11434/api/generate -d '{" + f"\"model\": \"llama3.3\", \"prompt\": \"{llama_prompt} {text}\"" +  "}'", shell=True, capture_output=True, text=True)
    print("<br>Entities: ")

    response_string = ""
    for line in result.stdout.split("\n"):
        try:
            my_object = json.loads(line)
            if "response" in my_object:
                response_string += my_object["response"] + " "
        except:
            pass
    process_response_string(response_string)


def process_with_spacy(text):
    result = nlp(text)
    matches = matcher(result)
    matched_entities = []
    print("<br>Entities: ")
    for match_id, start, end in matches:
        string_id = nlp.vocab.strings[match_id]
        span = result[start:end]
        print_label(string_id)
        print(f": {span.text};")
        matched_entities.append([start, end, string_id, span.text])
    for entity in result.ents:
        if not check_entity_overlap(matched_entities, entity.start, entity.end):
            print_label(entity.label_)
            print(f": {entity.text};")


def report_time_taken(start_time):
    end_time = datetime.datetime.now()
    time_taken = end_time - start_time
    print(f"Processing time: {round(time_taken.total_seconds(), 2)} seconds")


def print_html_footer():
    print("</body>\n</html>")


print_html_header()
file_name, file_path, column_names, max_processed, method, entities_filename, entities_path = process_form_data()
data_df = read_data_file(file_name, file_path, column_names)
if method == "spacy":
    nlp, matcher = setup_spacy(entities_path, entities_filename)
show_processing_parameters(file_path, column_names, max_processed, method, entities_path)
counter = 0
print("<ol>")
start_time = datetime.datetime.now()
for row_list in data_df.iter_rows():
    row_dict = dict(zip(data_df.columns, row_list))
    text = '\n'.join([str(row_dict[column_name] or "").strip() + "." for column_name in column_names if str(row_dict[column_name] or "").strip() != ""])
    print("<li><i>", text, "</i>")
    if method == "llama33":
        process_with_llama(llama_prompt, text)
    elif method == "spacy":
        process_with_spacy(text)
    else:
        print("Unknown text processing method: {method}!")
    print("</li><br>")
    counter += 1
    if counter >= max_processed:
        break
print("</ol>")
report_time_taken(start_time)
print_html_footer()
