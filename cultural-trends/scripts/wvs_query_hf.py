#updated
import os
import torch
import argparse

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from transformers import BitsAndBytesConfig, AutoConfig

from wvs_dataset import WVSDataset
from utils import read_json, write_json, parse_response_wvs

from transformers.utils import logging
logging.set_verbosity(50)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


OFFLOAD_DIR = "/content/offload"
os.makedirs(OFFLOAD_DIR, exist_ok=True)

#bnb_config = BitsAndBytesConfig(
#   load_in_8bit=True,
#    llm_int8_enable_fp32_cpu_offload=True
#)

def query_hf(
    qid: str,
    *,
    model_name: str = 'bigscience/mt0-small',
    version: int = 1,
    lang: str = 'en',
    max_tokens: int = 8,
    temperature: float = 0.7,
    n_gen: int = 5,
    batch_size: int = 4,
    fewshot: int = 0,
    cuda: int = 0,
    greedy: bool = False,
    generator = None,
    tokenizer = None,
    no_persona = False,
    subset = None,
    country: str = "egypt",
    max_retries: int = 10,
):

    model_name_ = model_name.split("/")[-1]
    savedir = f"../results_wvs_2/{model_name_}/{lang}"
    if not os.path.isdir(savedir):
        os.makedirs(savedir)

    filepath = f"../dataset/wvs_template.{lang}.yml"

    dataset = WVSDataset(filepath,
        language=lang,
        country=country,
        api=False,
        model_name=model_name_,
        use_anthro_prompt=False
    )

    device = torch.device(f'cuda:{cuda}' if torch.cuda.is_available() else 'cpu')
    print(f"Language={lang} | Temperature={temperature} | Tokens={max_tokens} | N={n_gen} | Batch={batch_size} | Version={version}")
    print(f"> Device {device}")

    if qid <= 0:
        question_ids = dataset.question_ids
    else:
        question_ids = [f"Q{qid}"]

    print(f"> Running {len(question_ids)} Qs")
    model_path = model_name
    if "mt0" in model_name_:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map="cpu", torch_dtype=torch.float16).to(device)
    else:
        # model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, trust_remote_code=True).to(device)
        if "AceGPT" in model_name:
            model_path = "/mnt/u14157_ic_nlp_001_files_nfs/nlpdata1/home/bkhmsi/models/models--FreedomIntelligence--AceGPT-13B-chat/snapshots/ab87ccbc2c4a05969957755aaabc04400bb20052"
        elif "Llama" in model_name:
            model_path = "eri00eli/llama2-13b-8bit"
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                #quantization_config=bnb_config,
                device_map="auto",
                low_cpu_mem_usage=True,
                #offload_folder=OFFLOAD_DIR,
                trust_remote_code=True,
                #local_files_only=True,
            )
            #model.eval()





       # model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cpu", torch_dtype=torch.float16).to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_path,
                                              trust_remote_code=True
                                              #local_files_only=True
                                              )

    if"Llama-2-13b-chat-hf" in model_name or "AceGPT-13B-chat" in model_name:
        print("> Changing padding side")
        tokenizer.padding_side = "left"

    if model_name == "gpt2" or "Sheared-LLaMA-1.3B" in model_name or "Llama-2-13b" in model_name or "AceGPT-13B-chat" in model_name:
        tokenizer.pad_token = tokenizer.eos_token

    for qid in question_ids:
        qid = int(qid[1:])
        dataset.set_question(index=qid)

        filesuffix = f"q={str(qid).zfill(2)}_lang={lang}_country={country}_temp={temperature}_maxt={max_tokens}_n={n_gen}_v{version}_fewshot={fewshot}"
        print(filesuffix)

        preds_path = os.path.join(savedir, f"preds_{filesuffix}.json")

        completions = []
        if os.path.exists(preds_path):
            completions = read_json(preds_path)

        if len(completions) >= len(dataset):
            print(f"Skipping Q{qid}")
            continue

        if len(completions) > 0:
            print(f"> Trimming Dataset from {len(completions)}")
            dataset.trim_dataset(len(completions))

        remaining_personas = len(dataset)
        persona_entries = dataset.persona_qid[f"Q{qid}"]
        question_entries = dataset.question_info[f"Q{qid}"]

        if fewshot > 0:
            fewshot_examples, _ = dataset.fewshot_examples()
        else:
            fewshot_examples = ""

        print(f"> Prompting {model_name} with Q{qid}")

        processed_offset = len(completions)

        for local_idx in tqdm(range(remaining_personas)):
            prompt_text = dataset[local_idx]
            persona = persona_entries[local_idx]
            q_info = question_entries[local_idx]

            question_id = q_info["id"] if isinstance(q_info["id"], str) and q_info["id"].startswith("Q") else f"Q{q_info['id']}"
            question_options = dataset.wvs_questions.get(question_id, {}).get("options", [])

            attempt_log = []
            invalid_attempts = []
            collected_responses = []
            valid_response_text = None

            attempts_made = 0

            while attempts_made < max_retries and valid_response_text is None:
                attempts_made += 1

                if fewshot_examples:
                    full_prompt = fewshot_examples + prompt_text
                else:
                    full_prompt = prompt_text

                tokens = tokenizer([full_prompt], padding=True, return_tensors="pt").to(device)

                generation_kwargs = {
                    "max_new_tokens": max_tokens,
                    "temperature": temperature,
                    "do_sample": (not greedy),
                    "num_return_sequences": n_gen,
                }

                if not generation_kwargs["do_sample"] and n_gen > 1:
                    generation_kwargs["num_beams"] = n_gen

                with torch.no_grad():
                    gen_outputs = model.generate(**tokens, **generation_kwargs)

                generated_tokens = gen_outputs[:, tokens["input_ids"].shape[-1]:]
                decoded_output = tokenizer.batch_decode(
                    generated_tokens,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )

                for response_text in decoded_output:
                    response_text = response_text.strip()
                    collected_responses.append(response_text)
                    parsed_value = parse_response_wvs(response_text, question_options)
                    attempt_log.append({
                        "attempt": attempts_made,
                        "response": response_text,
                        "parsed": parsed_value,
                    })
                    if parsed_value > 0 and valid_response_text is None:
                        valid_response_text = response_text
                        break
                    if parsed_value <= 0 and response_text:
                        invalid_attempts.append(response_text)

            if valid_response_text is not None:
                stored_responses = [valid_response_text]
            else:
                stored_responses = collected_responses if collected_responses else ["No valid response after retries"]
                print(
                    f"> No valid response after {attempts_made} call(s) for persona index {processed_offset + local_idx}"
                )

            response_entries_checked = [
                entry for entry in attempt_log if "response" in entry
            ]

            completions += [{
                "persona": persona,
                "question": q_info,
                "response": stored_responses,
                "attempt_log": attempt_log,
                "invalid_attempts": invalid_attempts,
                "attempt_summary": {
                    "attempts_made": attempts_made,
                    "max_retries": max_retries,
                    "n_gen": n_gen,
                    "valid": valid_response_text is not None,
                    "responses_checked": len(response_entries_checked),
                },
            }]

            write_json(preds_path, completions)

if __name__ == "__main__":

    # python wvs_query_hf_generate.py --qid -1 --model FreedomIntelligence/AceGPT-13B-chat --lang en --fewshot 3 --cuda 0
    # python wvs_query_hf_generate.py --qid -1 --model princeton-nlp/Sheared-LLaMA-1.3B --max-tokens 10 --lang en --fewshot 3 --cuda 0
    # python wvs_query_hf_generate.py --qid -1 --model princeton-nlp/Sheared-LLaMA-1.3B --max-tokens 5 --lang ar --fewshot 3 --cuda 0 --n-gen 1
    # python wvs_query_hf_generate.py --qid -1 --model meta-llama/Llama-2-13b-chat-hf --max-tokens 16 --lang en --fewshot 0 --cuda 0 --n-gen 5 --batch-size 4

    # python wvs_query_hf_generate.py --qid -1 --model meta-llama/Llama-2-13b-chat-hf --max-tokens 32 --lang ar --fewshot 0 --cuda 1 --n-gen 5 --batch-size 4
    # python wvs_query_hf_generate.py --qid -1 --model FreedomIntelligence/AceGPT-13B-chat --max-tokens 5 --lang ar --fewshot 0 --cuda 0 --n-gen 5 --batch-size 4
    # python wvs_query_hf_generate.py --qid -1 --model FreedomIntelligence/AceGPT-13B-chat --max-tokens 5 --lang en --fewshot 0 --cuda 0 --n-gen 5 --batch-size 4
    # python wvs_query_hf_generate.py --qid -1 --model bigscience/mt0-xxl --max-tokens 5 --lang en --fewshot 0 --cuda 1 --n-gen 5 --batch-size 4 --country us

    # cp -r /home/bkhmsi/.cache/huggingface/hub/models--FreedomIntelligence--AceGPT-13B-chat /mnt/u14157_ic_nlp_001_files_nfs/nlpdata1/home/bkhmsi/models
    parser = argparse.ArgumentParser()

    parser.add_argument('--qid', required=True, type=int, help='question index')
    parser.add_argument('--model', default="bigscience/mt0-small", help='model to use')
    parser.add_argument('--version', default=1, help='dataset version number')
    parser.add_argument('--lang', default="en", help='language')
    parser.add_argument('--max-tokens', default=4, type=int, help='maximum number of output tokens')
    parser.add_argument('--temperature', default=0.7, type=float, help='temperature')
    parser.add_argument('--n-gen', default=5, type=int, help='number of generations')
    parser.add_argument('--batch-size', default=4, type=int, help='batch size')
    parser.add_argument('--fewshot', default=0, type=int, help='fewshot examples')
    parser.add_argument('--cuda', default=0, type=int, help='cuda device number')
    parser.add_argument('--greedy', action="store_true", help='greedy decoding')
    parser.add_argument('--country', type=str, help='country')
    parser.add_argument('--max-retries', default=10, type=int, help='maximum number of generation attempts per sample')

    args = parser.parse_args()

    if args.greedy:
        args.n_gen = 1
        args.temperature = 1.0

    qid = int(args.qid)

    query_hf(
        qid=qid,
        model_name=args.model,
        version=args.version,
        lang=args.lang,
        max_tokens=args.max_tokens,
        temperature=float(args.temperature),
        n_gen=int(args.n_gen),
        batch_size=int(args.batch_size),
        fewshot=int(args.fewshot),
        cuda=args.cuda,
        greedy=args.greedy,
        country=args.country,
        max_retries=int(args.max_retries),
    )
    ##updated
