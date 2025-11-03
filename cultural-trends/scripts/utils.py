import re
import os
import yaml
import json
import time
import requests
import numpy as np
import pandas as pd

import scipy.stats
from itertools import product
from collections import Counter

MAX_ATTEMPTS = 10

def retry_request(url, payload, headers):
    for i in range(MAX_ATTEMPTS):
        try:
            response = requests.post(url, data=json.dumps(
                payload), headers=headers, timeout=90)
            json_response = json.loads(response.content)
            if "error" in json_response:
                print(json_response)
                print(f"> Sleeping for {2 ** i}")
                time.sleep(2 ** i)
            else:
                return json_response
        except:
            print(f"> Sleeping for {2 ** i}")
            time.sleep(2 ** i)  # exponential back off
    raise TimeoutError()

def convert_to_percentages(answers, options, answer_map=None, is_scale=False):
    answers_mapped = []
    for ans in answers:
        if ans == -1: continue
        if ans not in options and answer_map is not None:
            answers_mapped += [str(answer_map[ans])]
        elif not is_scale:
            answers_mapped += [options[ans-1]]
        else:
            answers_mapped += [ans]

    # Count the occurrences of each answer
    answer_counts = Counter(answers_mapped)
    # Calculate the total number of answers
    total_answers = len(answers)
    # Calculate the percentage for each unique answer and store it in a dictionary
    percentages = {answer: (count / total_answers) * 100 for answer, count in answer_counts.items()}
    labels = list(percentages.keys())
    values = [percentages[label] if label in percentages else 0 for label in labels]
    return options, values

def parse_range(data):
    """
    Turns a dictionary with number ranges as keys into one with single numbers as keys.

    Parameters:
        data (dict): A dictionary with keys as strings like "1-3" or "5", and any values.

    Returns:
        dict: A new dictionary with numbers as keys and the same values copied over.
    """
    data_dict = {}
    for q_range in data:
        if "-" in q_range:
            q_start, q_end = tuple(map(int, q_range.split("-")))
        else:
            q_start = q_end = int(q_range)

        for q_idx in range(q_start, q_end+1):
            data_dict[q_idx] = data[q_range]

    return data_dict

def cartesian_product(lists):
    return list(product(*lists))

def read_file(path):
    with open(path, 'r', encoding="utf-8") as fin:   # Open file for reading
        data = fin.readlines()                       # Read all lines into a list
    return data

def read_raw(path):
    with open(path, 'r', encoding="utf-8") as fin:
        data = fin.read()
    return data

def read_json(path):
    with open(path, 'r', encoding="utf-8") as fin:
        data = json.load(fin)
    return data

def read_yaml(path):
    with open(path, 'r', encoding="utf-8") as fin:
        data = yaml.load(fin, Loader=yaml.FullLoader)
    return data

def write_file(path, data):
    with open(path, 'w', encoding="utf-8") as fout:
        fout.write('\n'.join(data))

def write_json(path, data):
    with open(path, 'w', encoding="utf-8") as fout:
        json.dump(data, fout, ensure_ascii=False)

def kl_divergence(p, q):
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))

def jensen_shannon_distance(p, q):
    """
    method to compute the Jenson-Shannon Distance
    between two probability distributions
    """

    # convert the vectors into numpy arrays in case that they aren't
    p = np.array(p)
    q = np.array(q)

    # calculate m
    m = (p + q) / 2

    # compute Jensen Shannon Divergence
    divergence = (scipy.stats.entropy(p, m) + scipy.stats.entropy(q, m)) / 2

    # compute the Jensen Shannon Distance
    distance = np.sqrt(divergence)

    return distance

def append_row(
    data,
    **cols,
):
    for k, v in cols.items():
        data[k].append(v)


def parse_response_wvs(response: str, question_options: list):
  """
    Parse a raw model-generated response into a 1-based index for a World Values Survey question.

    Parameters:
        response (str): The raw text produced by the model.
        question_options (list[str]): The list of valid answer option strings, in order.

    Returns:
        int: A 1-based index corresponding to one of the `question_options`, or -1 if parsing fails.
  """

  # Make response case-insensitive
  response = response.lower().strip()

  # Matches a number enclosed in parentheses
  pattern = r"\(\d+\)"

  # Search for the number within response
  match = re.search(pattern, response)

  if match:

      # Extract number from match and convert it to integer
      match_text = match.group()
      stripped_text = match_text[1:-1]
      answer = int(stripped_text)

      return answer if 1 <= answer <= len(question_options) else -1

  else:
      # for option_idx, option in enumerate(question_options):
      #     if response == option.lower().strip():
      #         return option_idx+1
      # question_options = question_options[::-1]
      # for option_idx, option in enumerate(question_options):
      #     if response == option.lower().strip():
      #         return len(question_options)-option_idx


      # Match model response to one of the  valid options
      for option_idx, option in enumerate(question_options):
          if response in option.lower().strip():
              return option_idx+1

  # Handle cases where closing parenthesis is missing
  pattern = r"\(\d+"
  match = re.search(pattern, response)
  if match:
      answer = int(match.group()[1:])
      return answer if 1 <= answer <= len(question_options) else -1

  # Takes the first number of the response
  pattern = r"\d+"
  match = re.search(pattern, response)
  if match:
      answer = int(match.group())
      return answer if 1 <= answer <= len(question_options) else -1

  #  Unable to parse a valid index
  return -1

def parse_response(res: str, options: list):
    if type(res) == int:
        return res

    res = res.strip()
    pattern = r"\d+"
    match = re.search(pattern, res)
    if match:
        answer = int(match.group())
        if 1 <= answer <= len(options):
            return answer

    num_words = len(res.split())
    for i, option in enumerate(options):
        space_idx = option.index(" ")
        if res == option or \
           res == option.replace(".", "").strip() or \
           res == option[space_idx+1:].strip() or \
           res == option[space_idx+1:].strip().replace(".", "") or \
           res == ' '.join(option[:num_words]):
            return i+1

    for i in range(1, len(options)+1):
        if str(i) in res:
            return i
    return -1

def parse_question(q: dict, questions_en=None):
    index = '.'.join(str(x) for x in q['index'])
    text = q['questions'][0]

    if questions_en is not None:
        options = questions_en[index]["options"]
    else:
        options = q["options"]

    qparams = q["question_parameters"] if "question_parameters" in q else None

    return {
        'index': index,
        'text': text,
        'options': options,
        "qparams": qparams
    }

def append_data(qidx, data, questions, columnar_data):

    invalid_ans = 0
    for row in data:
        try:
            if "Error" in row:
                continue
            persona = row['persona']
            qid = '.'.join(str(x) for x in row['question']['id'])
            vid = row['question']['variant']
            responses = row['response']
            qparams = row["question"]["params"]
            key_qparam = list(qparams.keys())[0] if len(qparams) > 0 else None
            if qidx == 6 and qparams[key_qparam] in ["Corporations", "Public Companies","Local Government", "Electoral Process"]:
                continue

            for response in responses:

                question = questions[qid]
                options = question["options"]
                answer = parse_response(response, options)
                if answer == -1:
                    invalid_ans += 1
                    continue

                if key_qparam is not None:
                    qparam_idx = str(question["qparams"][key_qparam].index(qparams[key_qparam]) + 1)
                else:
                    qparam_idx = "0"

                if qidx == 10 and qparam_idx == "2":
                    # to remove the extra variant Nael added
                    continue

                # breakpoint()

                append_row(
                    columnar_data,
                    qid=qid, vid=vid, response=answer,
                    question_text=question['text'],
                    response_text=question['options'][answer-1],
                    qparam_id=qparam_idx,
                    **persona,
                )
        except:
            breakpoint()
            raise

    print('='*50)
    print(f"> {invalid_ans} Invalid Answers")
    print('='*50)
    return columnar_data, invalid_ans

def read_question(path, qidx, questions_en=None):
    questions = {}
    with open(path, 'r', encoding='utf-8') as fp:
        q_data = yaml.safe_load(fp)['dataset']
        for row in q_data:
            if row["index"][0] != qidx: continue
            q = parse_question(row, questions_en)
            questions[q['index']] = q
    return questions

def get_results_path(filesuffix, model_name, lang, version, m1):
    for v_num in range(version, 0, -1):
        if m1:
            v_num = f"{v_num}m1"
        results_path = f'results/{model_name}/{lang}/preds_{filesuffix}_v{v_num}.json'
        if os.path.exists(results_path):
            return results_path
    return None

def append_response(
    model_data:list[dict],
    row:dict,
    response_int:int,
    response_id:int,
    persona_id: int,
    q_responses:list[int]
    ):

  """
  Append a single response record to the flattened model dataset,
  optionally performing majority-voting over multiple parsed answers.

  Parameters:
    model_data (list[dict]): A new record is appended.
    row (dict): Raw entry containing keys:
            - "persona": sub-dict with demographic fields.
            - "question": sub-dict with "id" and "variant" (0-based) fields.
    response_int (int):
        The initially parsed integer response (1-based).
    response_id (int):
        Index (0-based) of this particular generation attempt within the variant.
    persona_id (int):
        Sequential index of the persona/variant block in the original dataset (used
        for tracking but not stored in the appended record).
    q_responses (list[int] | None):
        If not None, a list of multiple parsed integer responses collected for
        majority-voting.

  Returns:
      list[dict]:
          The same `model_data` list with one new appended response record.

  Raises:
      AssertionError:/
          If the final `response_int` (after voting) is not greater than zero.
  """


  # If q_response was provided, ignore response_int and do majority voting
  if q_responses is not None:
      # Count each unique response
      response_counter = Counter(q_responses)

      # Get them sorted by frecuency
      most_common_responses = response_counter.most_common()

      # First is the highest frecuency as it is sorted
      first_freq = most_common_responses[0]

      # Extract its frecuency
      max_freq = first_freq[1]

      # Collect tied winners
      max_responses = []

      # Iterate list frecuency till counts drop below max_freq
      for most_common in most_common_responses:
          if most_common[1] == max_freq:
              # Collect all tied for first
              max_responses += [most_common[0]]
          else:
              break
      # If multiple equally common answers, pick one at random
      response_int = np.random.choice(max_responses)

  assert response_int > 0

  # Append structured response to model data

  model_data += [{
      "persona.region": row["persona"]["region"],
      "persona.sex": row["persona"]["sex"],
      "persona.age": row["persona"]["age"],
      "persona.country": row["persona"]["country"],
      "persona.marital_status": row["persona"]["marital_status"],
      "persona.education": row["persona"]["education"],
      "persona.social_class": row["persona"]["social_class"],
      "question.id": row["question"]["id"],
      "question.variant": row["question"]["variant"],
      "response.id": response_id, # Which generation
      "response.answer": response_int
  }]
  return model_data


def convert_to_dataframe(
    model_data:list[dict],
    question_options:list[str],
    demographic_map: dict[str,str],
    eval_method: str = "mv_all",
    language: str = "en",
    is_scale_question: bool = False
    ):

  """ Flatten and evaluate raw model outputs into a structured DataFrame.

    Parameters:
        model_data (list[dict]):
            Raw output for a single question across all personas and variants.

        question_options (list[str]):
            The ordered list of valid option strings for this question, used
            to validate and parse model outputs.

        eval_method (str, default="mv_all")

        language (str, default="en")

        is_scale_question (bool, default=False):
            If True, collapse 10-point scale answers into 5 bins via ceil(ans/2).

    Returns:
        tuple[pd.DataFrame, int]:
            - DataFrame with columns:
                persona.region, persona.sex, persona.age, persona.country,
                persona.marital_status, persona.education, persona.social_class,
                question.id, question.variant,
                response.id, response.answer
              One row per persona-question according to `eval_method`.
            - invalid_count (int): number of generations that could not be
              parsed into a valid answer.

    Raises:
        AssertionError:
            If `eval_method` is invalid or if data ordering is wrong
            (expects variants 0–3 in sequence), or if a voted response
            is not a positive integer.

  """

  assert eval_method in {"flatten", "mv_sample", "mv_all", "first"}

  model_data_flat = []
  invalid_count = 0
  # Collect parsed ints for voting
  q_responses = []

  # Loop every raw row of model output
  for row_idx, row in enumerate(model_data):

      #if language != "en":
      #    row["persona"] = {d_text: (demographic_map[d_text][d_value] if d_text != "region" else demographic_map[d_text][d_value]) if d_text != "age" else d_value for d_text, d_value in row["persona"].items()}

      # Clear out any old responses at the start of each row.
      if eval_method == "mv_sample":
          q_responses = []

      # Reset every 4 rows (one set of variants)
      if row_idx % 4 == 0 and eval_method == "mv_all":
          q_responses = []

      # Verify data is in groups of 4 (1 persona - 4 variants)
      # assert row_idx % 4 == row["question"]["variant"]

      # Parse each generation (5 of them per row)
      for response_id, response in enumerate(row["response"]):

          # Turns the row string into 1..N or -1
          response_int = parse_response_wvs(response, question_options)

          if is_scale_question:
              response_int = int(np.ceil(response_int/2))

          if eval_method == "first":

              # If first parse response is invalid, just count and stop
              if response_int <= 0:
                  invalid_count += 1
              else:
              # Append the one and break (ignoring the other 4 generations)
                  model_data_flat = append_response(model_data_flat, row, response_int, response_id, row_idx, q_responses=None)
              break

          if eval_method == "flatten":
              # Every valid generation becomes a row
              if response_int <= 0:
                  invalid_count += 1
                  continue
              model_data_flat = append_response(model_data_flat, row, response_int, response_id, row_idx, q_responses=None)

          elif response_int > 0 and "mv" in eval_method:
              # Collect valid responses for later voting
              q_responses += [response_int]

      if eval_method == "mv_sample":
          # if no valid responses, count it
          if len(q_responses) == 0:
              # breakpoint()
              invalid_count += 1
              continue

          # Pass q_response list for voting
          model_data_flat = append_response(model_data_flat, row, -1, response_id, row_idx, q_responses)

      elif eval_method == "mv_all" and row_idx % 4 == 3: # vote in 4 variant

          if len(q_responses) == 0:
              invalid_count += 1
              continue
          # Vote over all 20 responses
          model_data_flat = append_response(model_data_flat, row, -1, response_id, row_idx, q_responses)

  return pd.DataFrame(model_data_flat), invalid_count

def create_wvs_question_map(headers:list[str], selected_questions:list[str]):
  """
    Creates a mapping from WVS question indices to their corresponding column names
    in the dataset, filtered by a list of selected question indices.

    Args:
        headers (List[str]): List of column names from the WVS dataset.
        selected_questions (List[int]): List of numeric question indices to retain.

    Returns:
        Dict[int, str]: A dictionary mapping question indices to column names.
    """
  wvs_question_map = {}
  for column in headers:
    match = re.search(r"Q(\d+)[\w]? (.+)", column) #A number following "Q" (Q(\d+)) / word character after the number ([\w]?)
    if match:
      qidx = int(match.group(1))
      if qidx in selected_questions:
        wvs_question_map[qidx] = column
  return wvs_question_map
