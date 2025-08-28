#!/usr/bin/env python3
# inspect_runs.py: compare analysis of systems with gold analysis
# usage: inspect_runs.py
# 20250822 e.tjongkimsang@esciencecenter.nl


import cgi
import cgitb
import datetime
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
</head>
<body>
<p><a href="/enriching">Home</a></p>
""")


def print_label(entity_label):
    if entity_label in ["PER", "PERSON"]:
        print(f"<font color=\"red\"><strong>{entity_label}</strong></font>", end="")
    elif entity_label in ["GPE", "LOC", "FAC"]:
        print(f"<font color=\"green\"><strong>{entity_label}</strong></font>", end="")
    else:
        print(f"<font color=\"black\"><strong>{entity_label}</strong></font>", end="")


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
    print(f"<br><font style=\"color:red\">ERROR: {text}</font>")


GOLD_DATA_FILE_RECOGNITION_BM = "/home/etjongkims/projects/enriching/data/gethin/bm-dataset-cut-random-100-annotations.txt"
MACHINE_DATA_FILE_RECOGNITION_BM_SPACY = "/home/etjongkims/projects/enriching/data/gethin/spacy_trf_output_100_with_locations.txt"
MACHINE_DATA_FILE_RECOGNITION_BM_NAMETAG3 = "/home/etjongkims/projects/enriching/data/gethin/nametag3_output_evaluate.txt"
MACHINE_DATA_FILE_RECOGNITION_BM_LLAMA = "/home/etjongkims/projects/enriching/data/gethin/llama_output_100.txt"
MACHINE_DATA_FILE_RECOGNITION_BM_GPTOSS = "/home/etjongkims/projects/enriching/data/gethin/gpt-oss_output.txt"
MACHINE_DATA_FILE_RECOGNITION_BM_DANDELION = "/home/etjongkims/projects/enriching/data/gethin/dandelion_output.txt"


def read_gold_data(task, data_source):
    if task == "recognition" and data_source == "bm":
        texts, gold_entities = utils.read_annotations(GOLD_DATA_FILE_RECOGNITION_BM)
        return texts, gold_entities
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
        else:
            print_error_message(f"Either task \"{task}\" or data source \"{data_source}\" or system \"{system}\" or their combination is not known")
        if entities:
            if not machine_entities:
                machine_entities = [{system: entities_list} for entities_list in entities]
            else:
                machine_entities = [dict(entities_dict, **{system: entities_list}) for entities_list, entities_dict in zip(entities, machine_entities)]
    return machine_entities


def guess_offsets(text, entities_list, line_counter, system):
    offsets = {}
    text = regex.sub(" 's", "'s", text)
    for entity_label in entities_list:
        if entity_label in ['p', 'l', 'PER', 'LOC']:
            for entity_text in entities_list[entity_label]:
                if len(entity_text) > 1:
                    pattern = regex.compile(rf"{entity_text}")
                    for counter in range(0, entities_list[entity_label][entity_text]):
                        char_pos = -1
                        for m in pattern.finditer(text):
                            if m.start() not in offsets:
                                char_pos = m.start()
                                break
                        if char_pos < 0:
                            print_error_message(f"cannot find entity \"{entity_text}\" on line {line_counter} for system {system}!")
                        else:
                            entity_char_pos = 0
                            for token in entity_text.split(" "):
                                if entity_char_pos == 0:
                                    offsets[char_pos] = [entity_text, entity_label]
                                elif char_pos + entity_char_pos in offsets:
                                    print_error_message(f"position {char_pos + entity_char_pos} is already in offsets for line {line_counter} of system \"{system}\"")
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
        #print("<tr><td></td><td></td><td>", machine_offsets, "</td></tr>")
        print(f"<tr><td></td><td>{system}</td><td>")
        print_text_with_entities(text, machine_offsets)
        print("</td></tr>")


print_html_header()
start_time = datetime.datetime.now()
task, data_source, system_list = process_form_data()
texts, gold_entities = read_gold_data(task, data_source)
machine_entities = read_machine_data(task, data_source, system_list)
print(f"<h1>Results task \"{task}\"</h1>")
print("<table><tr><th align=\"left\">Id</th><th align=\"left\">System</th><th align=\"left\">Text</th></tr>")
counter = 1
for text, gold_entities_list, machine_entities_dict in zip(texts, gold_entities, machine_entities):
    print(f"<tr><td align=\"right\">{counter}</td>")
    show_text(text, gold_entities_list, machine_entities_dict, counter)
    counter += 1
print("</table>")
report_time_taken(start_time)
print_html_footer()

sys.exit(0)
