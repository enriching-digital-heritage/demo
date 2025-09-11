#!/usr/bin/env python3
# inspect_runs.py: compare analysis of systems with gold analysis
# usage: inspect_runs.py
# 20250822 e.tjongkimsang@esciencecenter.nl


import cgi
import cgitb
import datetime
import polars as pl
import regex
import sys
import utils


cgitb.enable()  # Show detailed error messages in the browser


def print_html_header():
    print("Content-Type: text/html\n")
    print("""
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <title>Processing results</title>
  <base href="/enriching/" />
</head>
<body>
<p>
  <a href="/">Home</a>
  | British Museum: <a href="bm.csv">text</a>; <a href="bm.txt">gold labels</a>; <a href="disambiguation_annotation.csv">gold links</a>;
  | Egyptian Museum: <a href="em.csv">text</a>; <a href="em.txt">gold labels</a>
  | <a href="tagset.txt">Label set explanation</a>
  | <a href="prompts.txt">Prompts</a>
</p>
""")


def print_label(entity_label):
    if entity_label in ["PER", "PERSON"]:
        print(f"<font style=\"color: red;\"><strong>{entity_label}</strong></font>", end="")
    elif entity_label in ["GPE", "LOC", "FAC"]:
        print(f"<font style=\"color: green;\"><strong>{entity_label}</strong></font>", end="")
    else:
        print(f"<font style=\"color: black;\"><strong>{entity_label}</strong></font>", end="")


def report_time_taken(start_time):
    end_time = datetime.datetime.now()
    time_taken = end_time - start_time
    print(f"Processing time: {round(time_taken.total_seconds(), 2)} seconds")


def print_html_footer():
    print("</body>\n</html>")


def process_form_data():
    form = cgi.FieldStorage()
    return form.getfirst("task", "").strip(), form.getfirst("data_source", "").strip(), form.getlist("system")


def print_error_message(text):
    print(f"<font style=\"color:darkred;\">ERROR: {text}</font><br>")


GOLD_DATA_FILE_RECOGNITION_BM = "/home/etjongkims/projects/enriching/data/gethin/bm-dataset-cut-random-100-annotations.txt"
MACHINE_DATA_FILE_RECOGNITION_BM_SPACY = "/home/etjongkims/projects/enriching/data/gethin/spacy_trf_output_100_with_locations.txt"
MACHINE_DATA_FILE_RECOGNITION_BM_NAMETAG3 = "/home/etjongkims/projects/enriching/data/gethin/nametag3_output_evaluate.txt"
MACHINE_DATA_FILE_RECOGNITION_BM_LLAMA = "/home/etjongkims/projects/enriching/data/gethin/llama_output_100.txt"
MACHINE_DATA_FILE_RECOGNITION_BM_GPTOSS = "/home/etjongkims/projects/enriching/data/gethin/gpt-oss_output.txt"
MACHINE_DATA_FILE_RECOGNITION_BM_DANDELION = "/home/etjongkims/projects/enriching/data/gethin/dandelion_output.txt"
GOLD_DATA_FILE_RECOGNITION_EM = "/home/etjongkims/projects/enriching/data/rossana/Wikimedia-random-100-annotations.txt"
MACHINE_DATA_FILE_RECOGNITION_EM_SPACY = "/home/etjongkims/projects/enriching/data/rossana/spacy_trf_output_100.txt"
MACHINE_DATA_FILE_RECOGNITION_EM_NAMETAG3 = "/home/etjongkims/projects/enriching/data/rossana/nametag3_output_evaluate.txt"
MACHINE_DATA_FILE_RECOGNITION_EM_LLAMA = "/home/etjongkims/projects/enriching/data/rossana/llama3_output.txt"
MACHINE_DATA_FILE_RECOGNITION_EM_GPTOSS = "/home/etjongkims/projects/enriching/data/rossana/gpt-oss_output.txt"
MACHINE_DATA_FILE_RECOGNITION_EM_DANDELION = "/home/etjongkims/projects/enriching/data/rossana/dandelion_output.txt"
GOLD_DATA_FILE_DISAMBIGUATION_BM = "/home/etjongkims/projects/enriching/data/gethin/disambiguation_annotation.csv"
MACHINE_DATA_FILE_DISAMBIGUATION_BM_DANDELION = "/home/etjongkims/projects/enriching/data/gethin/dandelion_disambiguation.csv"
MACHINE_DATA_FILE_DISAMBIGUATION_BM_NAMETAG3 = "/home/etjongkims/projects/enriching/data/gethin/nametag3_output_disambiguation.csv"


def read_gold_data(task, data_source):
    if task == "recognition" and data_source == "bm":
        texts, gold_entities = utils.read_annotations(GOLD_DATA_FILE_RECOGNITION_BM)
        return texts, gold_entities, pl.DataFrame([])
    elif task == "recognition" and data_source == "em":
        texts, gold_entities = utils.read_annotations(GOLD_DATA_FILE_RECOGNITION_EM)
        return texts, gold_entities, pl.DataFrame([])
    elif task == "disambiguation" and data_source == "bm":
        texts, gold_entities = utils.read_annotations(GOLD_DATA_FILE_RECOGNITION_BM)
        disambiguation_df = utils.read_disambiguation_analysis(GOLD_DATA_FILE_DISAMBIGUATION_BM)
        return texts, gold_entities, disambiguation_df
    else:
        print_error_message(f"Either task \"{task}\" or data source \"{data_source}\" or their combination is not known")
        sys.exit(0)


def read_machine_data(task, data_source, system_list):
    machine_entities = []
    for system in system_list:
        entities = []
        if task == "recognition" and data_source == "bm" and system == "spacy":
            entities = utils.read_machine_analysis(MACHINE_DATA_FILE_RECOGNITION_BM_SPACY)
        elif task == "recognition" and data_source == "bm" and system == "nametag3":
            entities = utils.read_machine_analysis(MACHINE_DATA_FILE_RECOGNITION_BM_NAMETAG3)
        elif task == "recognition" and data_source == "bm" and system == "llama":
            entities = utils.read_machine_analysis(MACHINE_DATA_FILE_RECOGNITION_BM_LLAMA)
        elif task == "recognition" and data_source == "bm" and system == "gptoss":
            entities = utils.read_machine_analysis(MACHINE_DATA_FILE_RECOGNITION_BM_GPTOSS)
        elif task == "recognition" and data_source == "bm" and system == "dandelion":
            entities = utils.read_machine_analysis(MACHINE_DATA_FILE_RECOGNITION_BM_DANDELION)
        elif task == "recognition" and data_source == "em" and system == "spacy":
            entities = utils.read_machine_analysis(MACHINE_DATA_FILE_RECOGNITION_EM_SPACY)
        elif task == "recognition" and data_source == "em" and system == "nametag3":
            entities = utils.read_machine_analysis(MACHINE_DATA_FILE_RECOGNITION_EM_NAMETAG3)
        elif task == "recognition" and data_source == "em" and system == "llama":
            entities = utils.read_machine_analysis(MACHINE_DATA_FILE_RECOGNITION_EM_LLAMA)
        elif task == "recognition" and data_source == "em" and system == "gptoss":
            entities = utils.read_machine_analysis(MACHINE_DATA_FILE_RECOGNITION_EM_GPTOSS)
        elif task == "recognition" and data_source == "em" and system == "dandelion":
            entities = utils.read_machine_analysis(MACHINE_DATA_FILE_RECOGNITION_EM_DANDELION)
        elif task == "disambiguation" and data_source == "bm" and system == "dandelion":
            entities = utils.read_disambiguation_analysis(MACHINE_DATA_FILE_DISAMBIGUATION_BM_DANDELION)
        elif task == "disambiguation" and data_source == "bm" and system == "nametag3":
            entities = utils.read_disambiguation_analysis(MACHINE_DATA_FILE_DISAMBIGUATION_BM_NAMETAG3)
        else:
            print_error_message(f"Either task \"{task}\" or data source \"{data_source}\" or system \"{system}\" or their combination is not known")
        if entities:
            if not machine_entities:
                machine_entities = [{system: entities_list} for entities_list in entities]
            else:
                machine_entities = [dict(entities_dict, **{system: entities_list}) for entities_list, entities_dict in zip(entities, machine_entities)]
    return machine_entities


def find_entity_in_text(text, entity_text, offsets):
    pattern = regex.compile(rf"{entity_text}")
    char_pos = -1
    for m in pattern.finditer(text):
        if m.start() not in offsets:
            char_pos = m.start()
            break
    return char_pos


def guess_offsets(text, entities_list, line_counter, system):
    offsets = {}
    seen = {}
    for entity_label in entities_list:
        if entity_label in ['p', 'l', 'PER', 'LOC']:
            for entity_text in entities_list[entity_label]:
                if len(entity_text) > 1:
                    for counter in range(0, entities_list[entity_label][entity_text]):
                        if entity_text in seen:
                            seen[entity_text] += 1
                        else:
                            seen[entity_text] = 1
                        char_pos = find_entity_in_text(text, entity_text, offsets)
                        if char_pos < 0:
                            alternative_entity_text = regex.sub("(of|van|de|de|the)", " \g<1>", entity_text)
                            char_pos = find_entity_in_text(text, alternative_entity_text, offsets)
                            #print("<br>", system, entity_text, alternative_entity_text, char_pos, text)
                            if char_pos >= 0:
                                entity_text = alternative_entity_text
                        if char_pos < 0:
                            alternative_entity_text = regex.sub(" ", "", entity_text)
                            char_pos = find_entity_in_text(text, alternative_entity_text, offsets)
                            if char_pos >= 0:
                                entity_text = alternative_entity_text
                        if char_pos < 0:
                            alternative_entity_text = regex.sub("'s", " 's", entity_text)
                            alternative_entity_text = regex.sub("el-", "el - ", alternative_entity_text)
                            alternative_entity_text = regex.sub("n-R", "n - R", alternative_entity_text)
                            alternative_entity_text = regex.sub("-R", "- R", alternative_entity_text)
                            alternative_entity_text = regex.sub("([a-z])([A-Z])", "\g<1> \g<2>", alternative_entity_text)
                            char_pos = find_entity_in_text(text, alternative_entity_text, offsets)
                            if char_pos >= 0:
                                entity_text = alternative_entity_text
                        if char_pos < 0:
                            freq = "" if seen[entity_text] == 1 else f" ({seen[entity_text]})"
                            print_error_message(f"cannot find entity \"{entity_text}\"{freq} on line {line_counter} for system {system}!")
                        else:
                            entity_char_pos = 0
                            for token in entity_text.split(" "):
                                if entity_char_pos == 0:
                                    offsets[char_pos] = [entity_text, entity_label]
                                elif char_pos + entity_char_pos in offsets:
                                    print_error_message(f"position {char_pos + entity_char_pos} of entity \"{entity_text}\" is already in an entity for line {line_counter} of system \"{system}\"")
                                    del(offsets[char_pos])
                                    break
                                else:
                                    offsets[char_pos + entity_char_pos] = [entity_text[entity_char_pos:], None]
                                entity_char_pos += len(token) + 1
    return offsets


ENTITY_COLORS = {"p": "red",
                 "l": "green",
                 "PER": "red",
                 "LOC": "green"}


def print_text_with_entities(text, offsets):
    for offset in sorted(offsets.keys(), reverse=True):
        entity_start = offset
        entity_end = offset + len(offsets[offset][0])
        entity_label = offsets[offset][1]
        if entity_label:
            text = text[:entity_end] + "</font>" + text[entity_end:]
            text = text[:entity_start] + f"<font style=\"color:{ENTITY_COLORS[entity_label]}\">" + text[entity_start:]
    print(text)


def show_text(text, gold_entities_list, machine_entities_dict, line_counter):
    gold_offsets = guess_offsets(text, gold_entities_list, line_counter, system="Gold")
    print("<td>Gold</td><td>")
    print_text_with_entities(text, gold_offsets)
    print("</td></tr>")
    for system in sorted(machine_entities_dict):
        machine_offsets = guess_offsets(text, machine_entities_dict[system], line_counter, system)
        print(f"<tr><td></td><td>{system}</td><td>")
        print_text_with_entities(text, machine_offsets)
        print("</td></tr>")


def get_http_page_name(url):
    try:
        return url.split("/")[-1]
    except:
        return ""


print_html_header()
start_time = datetime.datetime.now()
task, data_source, system_list = process_form_data()
texts, gold_entities, disambiguation_list = read_gold_data(task, data_source)
machine_entities = read_machine_data(task, data_source, system_list)
table_style = "style=\"border: 1px; border-style: dotted; border-collapse: collapse; padding: 5px;\""
print(f"<h1>Results task \"{task}\"</h1>")
if task == "recognition":
    print(f"<p>Meaning of colors: red: <font style=\"color:red;\">PERSON</font>; green: <font style=\"color:green;\">LOCATION</font>")
print("<table><tr><th align=\"left\">Id</th><th align=\"left\">System</th><th align=\"left\">Text</th></tr>")
counter = 1
if task == "recognition":
    if len(machine_entities) == 0:
        machine_entities = len(gold_entities) * [[]]
    for text, gold_entities_list, machine_entities_dict in zip(texts, gold_entities, machine_entities):
        print(f"<tr><td align=\"right\">{counter}</td>")
        show_text(text, gold_entities_list, machine_entities_dict, counter)
        counter += 1
else:
    system_names = sorted(machine_entities[0].keys())
    for text, disambiguation_dict in zip(texts, disambiguation_list):
        print("<tr><td>", counter, "</td><td>", text, "</td></tr>")
        print(f"<tr><td></td><td><table {table_style}><tr><th {table_style}>Text</th><th style=\"border: 1px; border-style: dotted;\">Gold")
        for system_name in system_names:
            print(f"</th><th {table_style}>{system_name}")
        print("</th></tr>")
        seen = {}
        for entity in disambiguation_dict:
            if entity["entity_text"] not in seen:
                print(f"<tr><td {table_style}>", entity["entity_text"], f"</td><td {table_style}>")
                dbpedia_uri = entity["dbpedia_uri"]
                if dbpedia_uri:
                    print(f"<a href=\"{dbpedia_uri}\">{get_http_page_name(dbpedia_uri)}</a>")
                for system_name in system_names:
                    print(f"</td><td {table_style}>")
                    for machine_entity in machine_entities[counter-1][system_name]:
                        if machine_entity["entity_text"] == entity["entity_text"]:
                            machine_dbpedia_uri = entity["dbpedia_uri"]
                            if machine_dbpedia_uri:
                                print(f"<a href=\"{machine_dbpedia_uri}\">{get_http_page_name(machine_dbpedia_uri)}</a>")
                            break
                print("</td></tr>")
                seen[entity["entity_text"]] = True
        # add code for processing false machine positives
        print("</table></td></tr>")
        counter += 1
print("</table>")
report_time_taken(start_time)
print_html_footer()

sys.exit(0)
