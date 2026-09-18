import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import argparse
import time
import traceback
import json
import re
from AgentDropout.agents import AgentRegistry
from AgentDropout.agents.supervisor_reasoning_pick_metric import Supervisor
from AgentDropout.agents.final_decision import FinalRefer
from autogen_agentchat.teams import SelectorGroupChat
from AgentDropout.usage import TrackedOpenAIChatCompletionClient, set_usage_phase
from AgentDropout.teams import create_team
from AgentDropout.run_support import UsageRun, attach_usage, mark_sample_failed, sample_id
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.ui import Console
import asyncio
from typing import List, Tuple, Dict
import numpy as np 
from tqdm import tqdm


# ==============================================================================
def log_message(msg: str, log_file: str = None):
    print(msg)
    if log_file:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(msg + "\n")


# ==============================================================================
try:
    from math_verify import parse, verify
    HAS_MATH_VERIFY = True
except ImportError:
    HAS_MATH_VERIFY = False
    print("[WARNING] 'math_verify' library not found. Falling back to string comparison.")

def extract_boxed(text):
    if not text: return ""
    stack = []
    boxed_contents = []
    i = 0
    start_idx = -1
    while i < len(text):
        if text[i : i + 7] == "\\boxed{" and (i == 0 or text[i - 1] != "\\"):
            if not stack:
                start_idx = i + 7
            stack.append("{")
            i += 7
        elif text[i] == "{" and (i == 0 or text[i - 1] != "\\"):
            stack.append("{")
            i += 1
        elif text[i] == "}" and (i == 0 or text[i - 1] != "\\"):
            if stack:
                stack.pop()
                if not stack and start_idx != -1:
                    boxed_contents.append(text[start_idx:i])
                    start_idx = -1
            i += 1
        else:
            i += 1
    if boxed_contents:
        return boxed_contents[-1]
    pattern = r"\\boxed{((?:[^{}]|{(?:[^{}]|{[^{}]*})*})*?)}"
    matches = list(re.finditer(pattern, text))
    if matches:
        return matches[-1].group(1)
    return ""

def format_for_math_verify(answer):
    if not answer: return "$.$"
    answer = str(answer).strip()
    if answer.startswith("$"): answer = answer[1:]
    if answer.endswith("$"): answer = answer[:-1]
    answer = answer.strip()
    if not answer: return "$.$"
    return f"${answer}$"

def string_compare_answers(extracted, gold):
    def normalize(text):
        if not text: return ""
        text = str(text)
        text = re.sub(r"\s+", "", text)
        text = text.replace("\\frac", "")
        text = text.replace("\\cdot", "*")
        text = text.replace("\\times", "*")
        text = re.sub(r"\\[a-zA-Z]+", "", text)
        return text
    normalized_extracted = normalize(extracted)
    normalized_gold = normalize(gold)
    return (
        normalized_extracted == normalized_gold
        or normalized_gold in normalized_extracted
        or normalized_extracted in normalized_gold
    )

def check_correctness_olym(extracted_answer, gold_answer):
    if not extracted_answer: return False
    if HAS_MATH_VERIFY:
        try:
            formatted_gold = format_for_math_verify(gold_answer)
            formatted_extracted = format_for_math_verify(extracted_answer)
            gold_parsed = parse(formatted_gold)
            extracted_parsed = parse(formatted_extracted)
            if verify(gold_parsed, extracted_parsed):
                return True
        except Exception: pass 
    try:
        if string_compare_answers(extracted_answer, gold_answer):
            return True
    except Exception: pass
    return False


# ==============================================================================
def load_global_resources(metric_file, cache_file):
    if args.baseline_only or args.use_simple_audit:
        return [], np.array([])
    print(f"Loading Global Resources...")
    print(f" - Metrics: {metric_file}")
    print(f" - Cache:   {cache_file}")
    
    with open(metric_file, "r", encoding='utf-8') as f:
        metrics = json.load(f)
        
    emb_map = {}
    if os.path.exists(cache_file):
        with open(cache_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        emb_map[record["name"]] = record["vector"]
                    except: pass
                
    vectors_list = []
    for m in metrics:
        name = m['name']
        if name in emb_map:
            vectors_list.append(np.array(emb_map[name], dtype=np.float32))
        else:
            if len(emb_map) > 0:
                vectors_list.append(np.zeros(len(next(iter(emb_map.values()))), dtype=np.float32))
            else:
                vectors_list.append(np.array([]))
            
    embeddings = np.stack(vectors_list) if vectors_list else np.array([])
    print(f"Global Resources Loaded. Embedding Shape: {embeddings.shape}")
    
    return metrics, embeddings


# ==============================================================================
def init_team(preloaded_metrics, preloaded_embeddings) -> Tuple[SelectorGroupChat, FinalRefer, Dict[str, str], Supervisor]:
    
    
    supervisor = Supervisor(
        model=args.supervisor_model,
        api_key=args.supervisor_key, 
        base_url=args.supervisor_url,
        domain="olymMATH",
        pass_rate=args.pass_rate,
        prune_flag=True, 
        
        embedding_api_key=args.embedding_key,
        embedding_model=args.embedding_model,
        embedding_api_base=args.embedding_url,
        
        preloaded_metrics=preloaded_metrics,
        preloaded_embeddings=preloaded_embeddings,
        
        direct_k=args.direct_k,
        use_simple_audit=int(args.use_simple_audit),
        random_k=args.random_k,
        random_k_min=args.random_k_min,
        random_k_max=args.random_k_max,
        retrieval_mode=args.retrieval_mode,
        retrieve_p=args.retrieve_p,
        select_q=args.select_q,
        exact_select_q=args.exact_select_q,
        batch_audit_metrics=args.batch_audit_metrics,
)

    agent_resgistry = AgentRegistry()
    participants = [
        agent_resgistry.get(
            agent_name="MathSolver_olymMATH", 
            name=f"Participant_{i + 1}",
            domain="olymMATH",                
            model=args.reasoning_model,
            api_key=args.reasoning_key,
            base_url=args.reasoning_url,
            supervisor=supervisor,
            reflection_time=args.retries_times,
        ) for i in range(5)
    ]

    role_map = {agent.name: agent.role for agent in participants}
    for agent in participants:
        agent.role_map = role_map
        if not hasattr(agent, 'description'):
             agent.description = f"An AI agent with the role of {agent.role}."
        
    selector_prompt = """Select an agent to perform task.
    {roles}
    Current conversation context:
    {history}
    Read the above conversation, then select an agent from {participants} to perform the next task.
    Make sure the planner agent has assigned tasks before other agents start working.
    Only select one agent.
    """

    model_client = TrackedOpenAIChatCompletionClient(
        model=args.selector_model, 
        api_key=args.selector_key, 
        base_url=args.selector_url
    ) if args.framework == "dynamic" else None
    
    text_mention_termination = TextMentionTermination("TERMINATE")
    max_messages_termination = MaxMessageTermination(max_messages=args.max_turns)
    termination = text_mention_termination | max_messages_termination

    team = create_team(
        framework=args.framework,
        fixed_rounds=args.fixed_rounds,
        participants=participants,
        model_client=model_client,
        termination_condition=termination,
        selector_prompt=selector_prompt,
        allow_repeated_speaker=True,  
    )

    decision_maker = AgentRegistry.get(
        agent_name="FinalRefer",
        name="DecisionMaker",
        domain="olymMATH", 
        model=args.reasoning_model,
        api_key=args.reasoning_key,
        base_url=args.reasoning_url
    )
    
    return team, decision_maker, role_map, supervisor


# ==============================================================================
async def reasoning(question, team: SelectorGroupChat, decision_maker: FinalRefer, role_map: Dict[str, str], supervisor: Supervisor):
    
    is_baseline_mode = getattr(args, 'baseline_only', False)

    if is_baseline_mode:
        print(f"\n>>> [Mode] Baseline Only (No Audit / No Pruning)")
        await team.reset()
        supervisor.reset()
        supervisor.prune_flag = False 
        await Console(team.run_stream(task=question))
        history_messages = (team.final_messages if args.framework == "fixed"
                            else supervisor.get_messages_above_threshold())

    else:
        print(f"\n>>> [Phase 1]  (Task: {question[:30]}...)")
        
        await team.reset()
        supervisor.reset()
        supervisor.prune_flag = True 
        await Console(team.run_stream(task=question))
        
        history_messages = (team.final_messages if args.framework == "fixed"
                            else supervisor.get_messages_above_threshold())
        retained_count = len(history_messages)
        print(f"\n[Check] save: {retained_count}")
        
    
        if args.framework == "dynamic" and retained_count <= 1:
            print(f"\n⚠️  (Fallback Triggered)！")
            print(">>> [Phase 2]  Vanilla AutoGen...")
            
            await team.reset()
            supervisor.reset()
            set_usage_phase("fallback")
            original_prune_flag = supervisor.prune_flag
            supervisor.prune_flag = False 
            
            await Console(team.run_stream(task=question))
            
            history_messages = (team.final_messages if args.framework == "fixed"
                            else supervisor.get_messages_above_threshold())
            print(f"[Fallback Result] save: {len(history_messages)}")
            supervisor.prune_flag = original_prune_flag
    
    print("\n" + "="*50)
    print("--- [DEBUG] Final Decision  ---")
    
    if not history_messages:
        print("  >> !")
    else:
        for i, msg in enumerate(history_messages):
            if msg.source != 'user':
                print(f"  -  {i+1} |from: {msg.source}")

    raw_answer = await decision_maker.run_decision(history_messages=history_messages, role_map=role_map, task=question)
    raw_content = raw_answer.content.strip()
    
    print("\n[DEBUG] 2. Final Decision :")
    print(raw_content[:200] + "...") 

    print("="*50 + "\n")
    
    ret_scores = supervisor.get_scores(role_map)
    reflection_records = getattr(supervisor, 'reflection_records', [])
    
    return raw_content, ret_scores, reflection_records


# ==============================================================================
def write_to_file(out_file, data_id, data):
    data = attach_usage(data)
    exist_data = {}
    if os.path.exists(out_file):
        try:
            with open(out_file, 'r', encoding='utf-8') as f:
                exist_data = json.load(f)
        except json.JSONDecodeError:
            pass
    
    exist_data[str(data_id)] = data
  
    try:
        sorted_keys = sorted(exist_data.keys(), key=lambda x: int(x))
    except:
        sorted_keys = sorted(exist_data.keys())
        
    sorted_data = {key: exist_data[key] for key in sorted_keys}
    
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(sorted_data, f, indent=4, ensure_ascii=False)
    

async def run_sample(data, out_file, team, decision_maker, role_map, supervisor, log_file_path):
    question = data.get('question', data.get('problem', ''))
    instance_id = str(data.get('unique_id', data.get('id', 'unknown'))) 
    subject = data.get('subject', '')
    ground_truth = str(data.get('answer', data.get('final_answer', '')))
    
    try:
        current_time = time.strftime('%Y-%m-%d %H:%M:%S')
        log_message(f"\n{'='*40}", log_file_path)
        log_message(f"=== TASK ID: {instance_id} ===", log_file_path)
        log_message(f"{'='*40}", log_file_path)
        log_message(f"--- [ {current_time} ] Start Processing ---", log_file_path)

        raw_content, scores, reflection_records = await reasoning(question, team, decision_maker, role_map, supervisor)
        
  
        hypothesis = extract_boxed(raw_content)
        is_correct = check_correctness_olym(hypothesis, ground_truth)

        log_message(f"over: {instance_id} | Correct: {is_correct} (GT: {ground_truth} vs Pred: {hypothesis})", log_file_path)
        
        write_to_file(
            out_file, 
            instance_id, 
            {
                'id': instance_id, 
                'unique_id': instance_id,
                'subject': subject,
                'answer': ground_truth,
                'hypothesis': hypothesis, 
                'question': question,
                'raw_response': raw_content, 
                'is_correct': is_correct,   
                'scores': scores,            
                'reflection_records': reflection_records 
            }
        )
    except Exception as e:
        mark_sample_failed(e)
        log_message(f"!!!!!! [CRITICAL ERROR] Task {instance_id}: {e} !!!!!!", log_file_path)
        traceback.print_exc()


async def main():
    if not os.path.exists(args.in_file):
        raise FileNotFoundError(f"Input file not found: {args.in_file}")

    print(f"[INFO] Loading data from {args.in_file}...")
    try:
        input_data = []
        with open(args.in_file, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if content.startswith("["):
                input_data = json.loads(content)
            else:
                input_data = [json.loads(line) for line in content.splitlines() if line.strip()]

        if args.limit is not None and args.limit > 0:
            input_data = input_data[:args.limit]
            print(f"[INFO] Limit applied: {len(input_data)} tasks.")
            
    except Exception as e:
        raise RuntimeError(f"Data load failed: {args.in_file}") from e
        
    global_metrics, global_embeddings = load_global_resources(
        args.metric_pool_file, 
        args.embedding_cache_file
    )
    
    if args.log_file:
        FINAL_LOG_FILE = args.log_file
    else:
        base_dir = os.path.dirname(args.out_file)
        base_name = os.path.basename(args.out_file).replace(".json", "_full.log")
        FINAL_LOG_FILE = os.path.join(base_dir, base_name)
    
    os.makedirs(os.path.dirname(FINAL_LOG_FILE), exist_ok=True)
    
    with open(FINAL_LOG_FILE, "w", encoding="utf-8") as f:
        f.write(f"=== OlymMATH Run Logs ===\n")
        f.write(f"Total: {len(input_data)}\n\n")


    
    start_time = time.time()

    usage_run = UsageRun(args.out_file, framework=args.framework,
                         configuration={"fixed_rounds": args.fixed_rounds, "baseline_only": args.baseline_only})
    for instance in tqdm(input_data, desc="Processing"):
        with usage_run.sample(sample_id(instance)):
            team, decision_maker, role_map, supervisor = init_team(
                global_metrics,
                global_embeddings
            )
            await run_sample(instance, args.out_file, team, decision_maker, role_map, supervisor, FINAL_LOG_FILE)

    usage_run.raise_if_failed()
    total_time = time.time() - start_time
    print(f"\n🎉: {total_time:.2f}s")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_file', type=str, required=True)
    parser.add_argument('--out_file', type=str, required=True)
    
    parser.add_argument('--selector_url', type=str)
    parser.add_argument('--selector_model', type=str)
    parser.add_argument('--selector_key', type=str)
    
    
    parser.add_argument('--reasoning_url', type=str)
    parser.add_argument('--reasoning_model', type=str)
    parser.add_argument("--reasoning_key", type=str, default="EMPTY")
    parser.add_argument('--supervisor_url', type=str) 
    parser.add_argument('--supervisor_model', type=str)
    parser.add_argument("--supervisor_key", type=str, default="EMPTY")
    
    parser.add_argument("--embedding_url", type=str, required=True)
    parser.add_argument("--embedding_model", type=str, required=True)
    parser.add_argument("--embedding_key", type=str, default="EMPTY")
    parser.add_argument("--metric_pool_file", type=str, required=True)
    parser.add_argument("--embedding_cache_file", type=str, required=True)
    
    parser.add_argument("--metrics_retrieve_k", type=int, default=20)
    parser.add_argument("--pass_rate", type=float, default=0.8)
    parser.add_argument('--max_turns', type=int, default=10) 
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--log_file', type=str)
    
    parser.add_argument("--baseline_only", action="store_true")
    
    parser.add_argument("--use_simple_audit", action="store_true", help="Enable Simple Audit")
    
    parser.add_argument("--direct_k", type=int, default=5)
    
    parser.add_argument("--random_k", type=int, default=0)
    parser.add_argument("--random_k_min", type=int, default=0)
    parser.add_argument("--random_k_max", type=int, default=0)
    parser.add_argument("--retrieval_mode", choices=["direct", "rerank", "random"], default="direct")
    parser.add_argument("--retrieve_p", type=int, default=20)
    parser.add_argument("--select_q", type=int, default=None)
    parser.add_argument("--exact_select_q", action="store_true")
    parser.add_argument("--batch_audit_metrics", action="store_true")
    parser.add_argument('--retries_times', type=int, default=3)

    parser.add_argument("--framework", choices=["dynamic", "fixed"], default="dynamic")
    parser.add_argument("--fixed_rounds", type=int, default=None)
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.out_file), exist_ok=True)
    
    asyncio.run(main())
