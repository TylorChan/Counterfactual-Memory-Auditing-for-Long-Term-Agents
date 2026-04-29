import re

SESSION_NAMES = ["first", "second", "third", "fourth", "fifth"]


def get_session_field_name(episode, session_num: int, suffix: str) -> str:
    ordinal_name = None
    if 1 <= session_num <= len(SESSION_NAMES):
        ordinal_name = f"{SESSION_NAMES[session_num-1]}_session_{suffix}"
        if ordinal_name in episode:
            return ordinal_name
    generic_name = f"session_{session_num}_{suffix}"
    if generic_name in episode:
        return generic_name
    if ordinal_name is not None:
        return ordinal_name
    return generic_name


def get_history_session_count(episode) -> int:
    if "history_session_count" in episode:
        return int(episode["history_session_count"])
    count = 0
    while True:
        key = get_session_field_name(episode, count + 1, "dialogue")
        if key not in episode:
            break
        count += 1
    return count


def get_total_session_count(episode) -> int:
    if "total_session_count" in episode:
        return int(episode["total_session_count"])
    return get_history_session_count(episode)


def parse_session_num_from_memory_key(key: str) -> int:
    match = re.match(r"s(\d+)-", key)
    if not match:
        raise ValueError(f"Invalid memory key: {key}")
    return int(match.group(1))


def get_dialogue_(dialogue, session_num):
    dialogue_key = get_session_field_name(dialogue, session_num, "dialogue")
    speakers_key = get_session_field_name(dialogue, session_num, "speakers")
    dialogue_lst = dialogue[dialogue_key]
    speakers = dialogue[speakers_key]
    return dialogue_lst, speakers


def get_dialogue(episode, session_num) -> str:
    dialogue_key = get_session_field_name(episode, session_num, "dialogue")
    speakers_key = get_session_field_name(episode, session_num, "speakers")
    dialogue = episode[dialogue_key]
    speakers = episode[speakers_key]
    dialogue_input = ""
    for idx, sentence in enumerate(dialogue):
        dialogue_input += f"{speakers[idx]}: {sentence}\n"
    return dialogue_input.strip()


def to_dic(retrieved_memory):
    retrieved_memory_dic = {}
    key_lst = []
    for d in retrieved_memory:
        key = list(d.keys())[0]
        retrieved_memory_dic[key] = d[key]
        key_lst.append(key)
    return retrieved_memory_dic, key_lst
