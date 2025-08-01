#!/usr/bin/env python3

import cgi
import cgitb
import datetime
import json
import os
import pandas as pd
import regex
import spacy
import subprocess


cgitb.enable()  # Show detailed error messages in the browser

print("Content-Type: text/html\n")


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

form = cgi.FieldStorage()

file_item = form["file"] if "file" in form else None
column_name = form.getfirst("column", "").strip()
method = form.getfirst("method", "").strip()
max_processed = int(form.getfirst("max_processed", "").strip())

if "file" not in form or not form["file"].filename:
    print("<h1>No file uploaded</h1>")
elif not column_name:
    print("<h1>Column name not provided</h1>")
else:
    file_name = os.path.basename(file_item.filename)
    file_path = f"/tmp/{file_name}"

    with open(file_path, "wb") as f:
        f.write(file_item.file.read())

    print(f"<p>File saved as: {file_path}</p>")
    print(f"<p>Column of interest: <strong>{column_name}</strong></p>")
    print(f"<p>Processing the first {max_processed} cells</p>")

    data_df = pd.read_csv(file_path)
    prompt = "Can you find the person names and locations in this text and list them in combination with the character offset with respect to the beginning of the text? Please mention only the found strings in the format: type offset string, for example: PER 23 John and LOC 45 Paris. Here is the text:"
    print("<ol>")
    counter = 0
    if method == "spacy":
        nlp = spacy.load("en_core_web_trf", disable=["tagger", "parser", "lemmatizer"])
    start_time = datetime.datetime.now()
    for text in data_df[column_name]:
        print("<li><i>", text, "</i>")
        if method == "llama33":
            # result = subprocess.run(f"echo \"{prompt} {text}\" | ollama run llama3.3", shell=True, capture_output=True, text=True)
            result = subprocess.run("curl http://localhost:11434/api/generate -d '{" + f"\"model\": \"llama3.3\", \"prompt\": \"{prompt} {text}\"" +  "}'", shell=True, capture_output=True, text=True)
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
        elif method == "spacy":
            result = nlp(text)
            print("<br>Entities: ")
            for entity in result.ents:
                print_label(entity.label_)
                print(f": {entity.text};")
        else:
            print("Unknown text processing method: {method}!")
        print("</li><br>")
        counter += 1
        if counter >= max_processed:
            break
    print("</ol>")
    end_time = datetime.datetime.now()
    time_taken = end_time - start_time
    print(f"Processing time: {round(time_taken.total_seconds(), 2)} seconds")
