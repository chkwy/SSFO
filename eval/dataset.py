
import json


def load_data(datapath):
    print("loading data from %s" % datapath)
    try:
        with open(datapath, "r") as f:
            data_list = json.load(f)
    except json.JSONDecodeError:
        data_list = []
        with open(datapath, "r") as f:
            for line in f:
                data_list.append(json.loads(line))

    return data_list


def reformat_question(turn_list, dataset_name):

    # only take the lastest 7 turns
    turn_list = turn_list[-7:]
    assert turn_list[-1]['role'] == 'user'

    long_answer_dataset_list = [
        "openbooknqopen",
        "doc2dial",
        "quac",
        "qrecc",
        "inscit",
        "doqa_movies",
        "doqa_travel",
        "doqa_cooking",
        "hybridial",
        "convfinqa"]
    long_and_short_dataset_list = ["topiocqa"]
    entity_dataset_list = ["sqa"]
    short_dataset_list = ["coqa", 'nqopen', 'triviaqa']

    if dataset_name in long_answer_dataset_list:
        for item in turn_list:
            if item['role'] == 'user':
                # only needs to add it on the first user turn
                item['content'] = 'Please give a full and complete answer for the question. ' + item['content']
                break

    elif dataset_name in long_and_short_dataset_list:
        turn_list[-1]['content'] = "Answer the following question with a short span, or a full and complete answer. " + \
            turn_list[-1]['content']

    elif dataset_name in entity_dataset_list:
        turn_list[-1]['content'] = "Answer the following question with one or a list of items. " + \
            turn_list[-1]['content']

    elif dataset_name in short_dataset_list:
        turn_list[-1]['content'] = "Answer the following question with a short span. The answer needs to be just in a few words. " + turn_list[-1]['content']

    else:
        raise Exception("please input a correct dataset name!")

    question = ""
    for item in turn_list:
        if item["role"] == "user":
            # question += "User: " + item["content"] + "\n\n"
            question += item["content"] + "\n\n"
        else:
            assert item["role"] == "assistant"
            # question += "Assistant: " + item["content"] + "\n\n"
            question += item["content"] + "\n\n"

    # question += "Assistant:"
    question += "Answer:"

    return question


def get_inputs(
        data_list,
        dataset_name,
        tokenizer,
        num_ctx,
        max_output_len,
        max_seq_length=4096):

    # system = "System: This is a chat between a user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions based on the context. The assistant should also indicate when the answer cannot be found in the context. \n Please give an answer in just one sentence."

    prompt_list = []
    for item in data_list:
        if dataset_name == 'nqopen' or dataset_name == 'triviaqa':
            system = "Please give an answer in just one sentence."
            turn_list = [
                {
                    "role": "user",
                    "content": item.get('question', item.get('Question', None))
                }
            ]
        else:
            turn_list = item['messages']
        question_formatted = reformat_question(turn_list, dataset_name)

        if dataset_name == 'nqopen':
            ctx_list = ["title: " + ctx["title"] + ", source: " +
                        ctx["text"] for ctx in item['ctxs'][:num_ctx]]
        else:
            ctx_list = ["title: " + ctx["Title"] + ", source: " +
                        ctx["Description"] for ctx in item['SearchResults'][:1]]
        context = "\n\n".join(ctx_list)

        context_tokens = tokenizer.encode(context)
        question_tokens = tokenizer.encode(question_formatted)
        system_tokens = tokenizer.encode(system)

        if len(context_tokens) + len(question_tokens) + \
                len(system_tokens) + max_output_len >= max_seq_length:
            context_tokens = context_tokens[:max_seq_length -
                                            max_output_len -
                                            len(question_tokens) -
                                            len(system_tokens)]
            context = tokenizer.decode(
                context_tokens, skip_special_tokens=True)

        model_input = system + "\n\n" + context + "\n\n" + question_formatted
        model_input = [
            {"role": "system", "content": ""},
            {"role": "user", "content": model_input, }
        ]
        model_input = tokenizer.apply_chat_template(
            model_input, tokenize=False, add_generation_prompt=True)
        prompt_list.append(model_input)
    print(prompt_list[0])

    return prompt_list
