import base64
import os
import argparse
import time
import logging
import re
import unicodedata
import io
import soundfile as sf
import numpy as np
import torch
# import torchaudio
import datasets
import kaldialign
from tqdm import tqdm
from collections import defaultdict
from typing import Iterable, Tuple, List, TextIO, Dict
from openai import OpenAI
from tn.chinese.normalizer import Normalizer as ZhNormalizer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8091/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

SEED = 42

def store_transcripts(
    filename: str, texts: Iterable[Tuple[str, str, str]]
) -> None:
    """Save predicted results and reference transcripts to a file."""
    with open(filename, "w") as f:
        for cut_id, ref, hyp in texts:
            print(f"{cut_id}:\tref={ref}", file=f)
            print(f"{cut_id}:\thyp={hyp}", file=f)

def write_error_stats(
    f: TextIO,
    test_set_name: str,
    results: List[Tuple[str, str]],
    enable_log: bool = True,
) -> float:
    """Write statistics based on predicted results and reference transcripts."""
    subs: Dict[Tuple[str, str], int] = defaultdict(int)
    ins: Dict[str, int] = defaultdict(int)
    dels: Dict[str, int] = defaultdict(int)

    words: Dict[str, List[int]] = defaultdict(lambda: [0, 0, 0, 0, 0])
    num_corr = 0
    ERR = "*"
    for cut_id, ref, hyp in results:
        ali = kaldialign.align(ref, hyp, ERR)
        for ref_word, hyp_word in ali:
            if ref_word == ERR:
                ins[hyp_word] += 1
                words[hyp_word][3] += 1
            elif hyp_word == ERR:
                dels[ref_word] += 1
                words[ref_word][4] += 1
            elif hyp_word != ref_word:
                subs[(ref_word, hyp_word)] += 1
                words[ref_word][1] += 1
                words[hyp_word][2] += 1
            else:
                words[ref_word][0] += 1
                num_corr += 1
    
    # Calculate ref_len based on characters for Chinese (assuming space separation if provided, else len)
    # Replicating infer_batch.py logic:
    ref_len = sum([len(r) for _, r, _ in results])

    sub_errs = sum(subs.values())
    ins_errs = sum(ins.values())
    del_errs = sum(dels.values())
    tot_errs = sub_errs + ins_errs + del_errs
    if ref_len > 0:
        tot_err_rate = "%.2f" % (100.0 * tot_errs / ref_len)
    else:
        tot_err_rate = "0.00"

    if enable_log:
        logging.info(
            f"[{test_set_name}] %WER {tot_errs / ref_len:.2%} "
            f"[{tot_errs} / {ref_len}, {ins_errs} ins, "
            f"{del_errs} del, {sub_errs} sub ]"
        )

    print(f"%WER = {tot_err_rate}", file=f)
    print(
        f"Errors: {ins_errs} insertions, {del_errs} deletions, "
        f"{sub_errs} substitutions, over {ref_len} reference "
        f"words ({num_corr} correct)",
        file=f,
    )
    print(
        "Search below for sections starting with PER-UTT DETAILS:, "
        "SUBSTITUTIONS:, DELETIONS:, INSERTIONS:, PER-WORD STATS:",
        file=f,
    )

    print("", file=f)
    print("PER-UTT DETAILS: corr or (ref->hyp)  ", file=f)
    for cut_id, ref, hyp in results:
        ali = kaldialign.align(ref, hyp, ERR)
        # Simple alignment printing
        print(
            f"{cut_id}:\t"
            + " ".join(
                (
                    ref_word if ref_word == hyp_word else f"({ref_word}->{hyp_word})"
                    for ref_word, hyp_word in ali
                )
            ),
            file=f,
        )

    print("", file=f)
    print("SUBSTITUTIONS: count ref -> hyp", file=f)
    for count, (ref, hyp) in sorted([(v, k) for k, v in subs.items()], reverse=True):
        print(f"{count}   {ref} -> {hyp}", file=f)

    print("", file=f)
    print("DELETIONS: count ref", file=f)
    for count, ref in sorted([(v, k) for k, v in dels.items()], reverse=True):
        print(f"{count}   {ref}", file=f)

    print("", file=f)
    print("INSERTIONS: count hyp", file=f)
    for count, hyp in sorted([(v, k) for k, v in ins.items()], reverse=True):
        print(f"{count}   {hyp}", file=f)

    print("", file=f)
    print("PER-WORD STATS: word  corr tot_errs count_in_ref count_in_hyp", file=f)
    for _, word, counts in sorted(
        [(sum(v[1:]), k, v) for k, v in words.items()], reverse=True
    ):
        (corr, ref_sub, hyp_sub, ins, dels) = counts
        tot_errs = ref_sub + hyp_sub + ins + dels
        ref_count = corr + ref_sub + dels
        hyp_count = corr + hyp_sub + ins
        print(f"{word}   {corr} {tot_errs} {ref_count} {hyp_count}", file=f)
        
    if ref_len > 0:
        return float(tot_errs) / float(ref_len) * 100.0
    else:
        return 0.0

def normalize_text_alimeeting(text: str) -> str:
    """Text normalization similar to M2MeT challenge baseline."""
    text = text.replace('\u00A0', '') 
    text = text.replace(" ", "")
    text = text.replace("<sil>", "")
    text = text.replace("<%>", "")
    text = text.replace("<->", "")
    text = text.replace("<$>", "")
    text = text.replace("<#>", "")
    text = text.replace("<_>", "")
    text = text.replace("<space>", "")
    text = text.replace("`", "")
    text = text.replace("&", "")
    text = text.replace(",", "")
    if re.search("[a-zA-Z]", text):
        text = text.upper()
    text = text.replace("Ａ", "A")
    text = text.replace("ａ", "A")
    text = text.replace("ｂ", "B")
    text = text.replace("ｃ", "C")
    text = text.replace("ｋ", "K")
    text = text.replace("ｔ", "T")
    text = text.replace("，", "")
    text = text.replace("丶", "")
    text = text.replace("。", "")
    text = text.replace("、", "")
    text = text.replace("？", "")
    return text

def encode_audio_to_base64(audio_array: np.ndarray, sr: int) -> str:
    """Encode audio array to base64 wav data."""
    with io.BytesIO() as bio:
        sf.write(bio, audio_array, sr, format='wav')
        bio.seek(0)
        audio_bytes = bio.read()
    return base64.b64encode(audio_bytes).decode("utf-8")

def get_asr_query(audio_base64: str):
    """Construct the ASR query in the format expected by the model."""
    question = "请将这段中文语音转换为纯文本，去掉标点符号。"
    prompt = {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": f"{question}",
            },
            {
                "type": "audio_url",
                "audio_url": {"url": f"data:audio/wav;base64,{audio_base64}"},
            },
        ],
    }
    return prompt

def get_system_prompt():
    return {
        "role": "system",
        "content": "You are a speech recognition model.",
    }

def run_benchmark(args):
    # Load dataset
    print(f"Loading dataset {args.huggingface_dataset}...")
    dataset = datasets.load_dataset(
        args.huggingface_dataset,
        args.subset_name,
        split=args.split_name,
        trust_remote_code=True,
    )
    dataset = dataset.select(range(800))

    # Initialize Normalizer
    zh_tn_model = ZhNormalizer(
        cache_dir="./cache",
        remove_erhua=False,
        remove_interjections=False,
        remove_puncts=True,
        overwrite_cache=False,
    )

    def normalize_text(text):
        text = unicodedata.normalize("NFKC", text)
        text = normalize_text_alimeeting(text)
        return zh_tn_model.normalize(text)

    results = []
    target_sr = 16000
    
    # Model parameters for generation
    model_name = "Qwen/Qwen2.5-Omni-7B" # Or make this an arg
    model_name = "/workspace_yuekai/Qwen2.5-Omni-3B"
    # Use tqdm for progress bar
    print("Starting inference...")
    start_time = time.time()
    
    for item in tqdm(dataset):
        # Extract ID
        utt_id = item.get("id") or item.get("segment_id") or str(item.get("key", "unknown"))
        
        # Extract Reference Text
        ref_text = item.get(args.ref_column, "")
        if not ref_text:
             if "text" in item: ref_text = item["text"]
             elif "sentence" in item: ref_text = item["sentence"]
        
        # Extract Audio
        audio_info = item["audio"]
        audio_array = audio_info["array"]
        sr = audio_info["sampling_rate"]

        # Resample if needed
        if sr != target_sr:
            raise ValueError(f"Sampling rate mismatch: {sr} != {target_sr}")
            # Using torchaudio for resampling as in infer_batch.py
            # audio_tensor = torch.from_numpy(audio_array).float()
            # resampler = torchaudio.transforms.Resample(sr, target_sr)
            # audio_tensor = resampler(audio_tensor)
            # audio_array = audio_tensor.numpy()
            # sr = target_sr
        
        # Encode to Base64
        audio_base64 = encode_audio_to_base64(audio_array, sr)

        # Construct Query
        prompt = get_asr_query(audio_base64)
        system_prompt = get_system_prompt()

        try:
            chat_completion = client.chat.completions.create(
                messages=[system_prompt, prompt],
                model=model_name,
                temperature=0.0, # Deterministic
                top_p=1.0,
                modalities=["text"],
                max_tokens=2048,
                extra_body={"repetition_penalty": 1.1}, # Add other params if needed
            )
            
            hyp_text = chat_completion.choices[0].message.content
            print(hyp_text)
            # breakpoint()
        except Exception as e:
            logger.error(f"Error processing {utt_id}: {e}")
            hyp_text = ""

        # Normalize
        hyp_norm = normalize_text(hyp_text).upper()
        ref_norm = normalize_text(ref_text).upper()
        
        results.append((utt_id, ref_norm, hyp_norm))
        # Optional: Print ongoing results?
        # print(f"{utt_id} | REF: {ref_norm} | HYP: {hyp_norm}")

    end_time = time.time()
    print(f"Inference time: {end_time - start_time} seconds")

    output_path = os.path.join(args.log_dir, args.output_file)
    stats_path = os.path.join(args.log_dir, args.stats_file)
    time_path = os.path.join(args.log_dir, "time.txt")
    os.makedirs(args.log_dir, exist_ok=True)

    print(f"Saving transcripts to {output_path}...")
    store_transcripts(output_path, results)

    print(f"Saving error stats to {stats_path}...")
    with open(stats_path, "w") as f:
        write_error_stats(f, args.huggingface_dataset, results)
    
    print(f"Saving time to {time_path}...")
    with open(time_path, "w") as f:
        f.write(f"Inference time: {end_time - start_time} seconds")
    
    print("Done.")

def parse_args():
    parser = argparse.ArgumentParser(description="ASR Benchmark using OpenAI-compatible API")
    parser.add_argument(
        "--log_dir",
        type=str,
        default="./results_aishell",
        help="Directory to save the results and stats"
    )
    # parser.add_argument(
    #     "--huggingface_dataset",  
    #     type=str,
    #     default="yuekai/speechio",
    #     help="Dataset name"
    # )
    # parser.add_argument(
    #     "--subset_name", 
    #     type=str, 
    #     default="SPEECHIO_ASR_ZH00007", 
    #     help="Dataset subset name"
    # )
    parser.add_argument(
        "--huggingface_dataset",  
        type=str,
        default="yuekai/aishell",
        help="Dataset name"
    )
    parser.add_argument(
        "--subset_name", 
        type=str, 
        default="test", 
        help="Dataset subset name"
    )
    parser.add_argument(
        "--split_name", 
        type=str, 
        default="test", 
        help="Dataset split name"
    )
    parser.add_argument(
        "--ref_column",
        type=str,
        default="text",
        help="Column name for reference text in dataset"
    )
    parser.add_argument(
        "--output_file", 
        type=str, 
        default="hypos.txt", 
        help="Output file for transcripts"
    )
    parser.add_argument(
        "--stats_file", 
        type=str, 
        default="wer.txt", 
        help="Output file for error statistics"
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_benchmark(args)
