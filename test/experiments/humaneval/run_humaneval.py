import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import argparse
import time
import traceback
import json
from masrubric.agents import AgentRegistry
from masrubric.agents.supervisor_reasoning_pick_metric import Supervisor
from masrubric.agents.final_decision import FinalWriteCodeMBPP
from autogen_agentchat.teams import SelectorGroupChat
from masrubric.usage import TrackedOpenAIChatCompletionClient, set_usage_phase
from masrubric.teams import create_team
from masrubric.run_support import UsageRun, attach_usage, mark_sample_failed, sample_id
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.ui import Console
import asyncio
from typing import List, Tuple, Dict
import re
import numpy as np
from tqdm import tqdm

from masrubric.tools.coding.python_executor import HumanEvalExecutor


# ==============================================================================
def log_message(msg: str, log_file: str = None):
    print(msg)
    if log_file:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(msg + "\n")


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

def prepare_humaneval_data(item: dict) -> dict | None:
    if 'prompt' not in item or 'test' not in item:
        return None
    task_id = item.get('task_id', item.get('name', 'Unknown'))
    return {
        "id": task_id,
        "prompt": item['prompt'],
        "test": item['test'],
        "entry_point": item.get('entry_point', ''),
        "canonical_solution": item.get('canonical_solution', ''),
        "original_data": item
    }

def extract_code_from_response(response):
    if isinstance(response, str):
        code = response.lstrip("```python\n").rstrip("\n```")
        if "```" in code:
            code = code.split("```")[0]
        return code.strip()
    return ""


# ==============================================================================
def init_team(preloaded_metrics, preloaded_embeddings) -> Tuple[SelectorGroupChat, FinalWriteCodeMBPP, Dict[str, str], Supervisor]:
    

    supervisor = Supervisor(
        model=args.supervisor_model,
        api_key=args.supervisor_key,
        base_url=args.supervisor_url,
        metrics_retrieve_k=args.metrics_retrieve_k,
        pass_rate=args.pass_rate,
        prune_flag=True,
        metric_pool_file=args.metric_pool_file, 
        embedding_cache_file=args.embedding_cache_file,
        embedding_api_key=args.embedding_key,
        embedding_model = args.embedding_model,
        embedding_api_base = args.embedding_url,
        domain="code", 
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
            agent_name="CodeWriting_humaneval", 
            name=f"Participant_{i + 1}",
            domain="humaneval",                 
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

    selector_prompt = """You are the team's scrum master. Select the next agent to advance the task.
    ... (Same Prompt) ...
    Select the single most logical agent from {participants} to speak next. Only return the agent's name.
    """

    from openai import Timeout
    model_client = TrackedOpenAIChatCompletionClient(
        model=args.selector_model, 
        api_key=args.selector_key, 
        base_url=args.selector_url,
        http_client_args={"timeout": Timeout(120.0, connect=10.0)},
        max_retries=3,
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
        agent_name="FinalWriteCodeMBPP", 
        name="DecisionMaker",
        domain="humaneval", 
        model=args.reasoning_model,
        api_key=args.reasoning_key,
        base_url=args.reasoning_url,
    )

    return team, decision_maker, role_map, supervisor


# ==============================================================================
async def reasoning(question, team: SelectorGroupChat, decision_maker: FinalWriteCodeMBPP, role_map: Dict[str, str], supervisor: Supervisor):
    
    is_baseline_mode = getattr(args, 'baseline_only', False)

    if is_baseline_mode:
        print(f"\n>>> [Mode] Baseline Only")
        await team.reset()
        supervisor.reset()
        supervisor.prune_flag = False 
        await Console(team.run_stream(task=question))
        history_messages = (team.final_messages if args.framework == "fixed"
                            else supervisor.get_messages_above_threshold())
    else:
        print(f"\n>>> [Phase 1] Adversarial Audit Mode")
        await team.reset()
        supervisor.reset()
        supervisor.prune_flag = True 
        await Console(team.run_stream(task=question))
        
        history_messages = (team.final_messages if args.framework == "fixed"
                            else supervisor.get_messages_above_threshold())
        if args.framework == "dynamic" and len(history_messages) <= 1:
            print(f"\n⚠️ Fallback Triggered!")
            await team.reset()
            supervisor.reset()
            set_usage_phase("fallback")
            original_prune_flag = supervisor.prune_flag
            supervisor.prune_flag = False 
            await Console(team.run_stream(task=question))
            history_messages = (team.final_messages if args.framework == "fixed"
                            else supervisor.get_messages_above_threshold())
            supervisor.prune_flag = original_prune_flag

    print("\n" + "=" * 50)
    print("--- [DEBUG] Final Decision ---")
    
    raw_answer = await decision_maker.run_decision(history_messages=history_messages, role_map=role_map, task=question)
    raw_content = raw_answer.content.strip()
    final_answer = extract_code_from_response(raw_content)
    
    print(f"\n[DEBUG] Extracted Code Length: {len(final_answer)}")
    print("=" * 50 + "\n")

    ret_scores = supervisor.get_scores(role_map)
    reflection_records = getattr(supervisor, 'reflection_records', [])
    
    return final_answer, ret_scores, reflection_records


# ==============================================================================
def write_to_file(out_file, data_id, data):
    data = attach_usage(data)
    exist_data = {}
    if os.path.exists(out_file):
        try:
            with open(out_file, "r") as f:
                exist_data = json.load(f)
        except: pass

    exist_data[str(data_id)] = data
    try:
        sorted_keys = sorted(exist_data.keys(), key=lambda x: int(x.split('/')[-1]) if '/' in x else x)
    except:
        sorted_keys = sorted(exist_data.keys())
        
    sorted_data = {key: exist_data[key] for key in sorted_keys}

    with open(out_file, "w") as f:
        json.dump(sorted_data, f, indent=4, ensure_ascii=False)

async def run_sample(data, out_file, team, decision_maker, role_map, supervisor, log_file_path):
    instance_id = data.get("id", "unknown")
    entry_point = data.get("entry_point", None) 
    
    try:
        current_time = time.strftime('%Y-%m-%d %H:%M:%S')
        log_message(f"\n{'='*40}", log_file_path)
        log_message(f"=== TASK ID: {instance_id} ===", log_file_path)
        log_message(f"{'='*40}", log_file_path)
        log_message(f"--- [ {current_time} ] Start Processing ---", log_file_path)

        task = data["prompt"] 
        test_code = data.get("test", "")
        
        answer, scores, reflection_records = await reasoning(
            task, team, decision_maker, role_map, supervisor
        )

        is_solved = False
        execution_result = ""
        error = ""

        if answer and test_code:
            try:
                executor = HumanEvalExecutor() 
                is_solved, execution_result, error_tuple = executor.execute(
                    answer, 
                    [test_code], 
                    entry_point=entry_point, 
                    timeout=15 
                )
                error = str(error_tuple)
            except Exception as e:
                mark_sample_failed(e)
                is_solved = False
                execution_result = "Executor crashed."
                error = str(e)

        log_message(f"Finished: {instance_id} | Solved: {is_solved} | {execution_result}", log_file_path)

        write_to_file(
            out_file,
            instance_id,
            {
                "answer": data.get("canonical_solution", ""),
                "hypothesis": answer,
                "solved": is_solved,
                "execution_result": execution_result,
                "error": error,
                "question": task,
                "scores": scores,
                "reflection_records": reflection_records
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
        with open(args.in_file, "r", encoding='utf-8') as f:
            input_lines = f.readlines()
        raw_data = [json.loads(line) for line in input_lines if line.strip()]
        
        adapted_data = []
        for item in raw_data:
            res = prepare_humaneval_data(item) 
            if res: adapted_data.append(res)
        
        if args.limit is not None and args.limit > 0:
            print(f"[INFO] 🔧 Limit applied: {args.limit} tasks.")
            adapted_data = adapted_data[:args.limit]
        
    except Exception as e:
        raise RuntimeError(f"Data loading failed: {args.in_file}") from e

    global_metrics, global_embeddings = load_global_resources(
        args.metric_pool_file, 
        args.embedding_cache_file
    )

    if args.log_file:
        FINAL_LOG_FILE = args.log_file
    else:
        base_dir = os.path.dirname(args.out_file)
        base_name = os.path.basename(args.out_file).replace(".json", "_detailed.log")
        FINAL_LOG_FILE = os.path.join(base_dir, base_name)

    os.makedirs(os.path.dirname(FINAL_LOG_FILE), exist_ok=True)
    
    with open(FINAL_LOG_FILE, "w", encoding="utf-8") as f:
        f.write(f"=== HumanEval Run Logs ===\n")
        f.write(f"Total: {len(adapted_data)}\n\n")

    print(f"🚀 Processing {len(adapted_data)} tasks...")
    print(f"📝 Logs: {FINAL_LOG_FILE}")
    
    start_time = time.time()
    
    usage_run = UsageRun(args.out_file, framework=args.framework,
                         configuration={"fixed_rounds": args.fixed_rounds, "baseline_only": args.baseline_only})
    for instance in tqdm(adapted_data, desc="Processing"):
        with usage_run.sample(sample_id(instance)):
            team, decision_maker, role_map, supervisor = init_team(global_metrics, global_embeddings)
            await run_sample(instance, args.out_file, team, decision_maker, role_map, supervisor, FINAL_LOG_FILE)
    
    usage_run.raise_if_failed()
    total_time = time.time() - start_time
    print(f"\n🎉 Done in {total_time:.2f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_file", type=str)
    parser.add_argument("--out_file", type=str)
    parser.add_argument("--log_file", type=str)
    
    parser.add_argument('--selector_url', type=str)
    parser.add_argument('--selector_model', type=str)
    parser.add_argument('--selector_key', type=str)
    
    parser.add_argument("--reasoning_url", type=str)
    parser.add_argument("--reasoning_model", type=str)
    parser.add_argument("--reasoning_key", type=str, default="EMPTY")
    parser.add_argument("--supervisor_url", type=str)
    parser.add_argument("--supervisor_model", type=str)
    parser.add_argument("--supervisor_key", type=str, default="EMPTY")
    
    parser.add_argument("--embedding_url", type=str, required=True)
    parser.add_argument("--embedding_model", type=str, required=True)
    parser.add_argument("--embedding_key", type=str, default="EMPTY")
    parser.add_argument("--metric_pool_file", type=str, required=True)
    parser.add_argument("--embedding_cache_file", type=str, required=True)
    
    parser.add_argument("--max_turns", type=int)
    parser.add_argument("--metrics_retrieve_k", type=int, default=20)
    parser.add_argument("--pass_rate", type=float, default=0.8)
    parser.add_argument("--retries_times", type=int, default=3)
    
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--baseline_only", action="store_true")
    
    parser.add_argument("--use_simple_audit", action="store_true")
    parser.add_argument("--direct_k", type=int, default=5)
    parser.add_argument("--random_k", type=int, default=0)
    parser.add_argument("--random_k_min", type=int, default=0)
    parser.add_argument("--random_k_max", type=int, default=0)
    parser.add_argument("--retrieval_mode", choices=["direct", "rerank", "random"], default="direct")
    parser.add_argument("--retrieve_p", type=int, default=20)
    parser.add_argument("--select_q", type=int, default=None)
    parser.add_argument("--exact_select_q", action="store_true")
    parser.add_argument("--batch_audit_metrics", action="store_true")
    parser.add_argument("--framework", choices=["dynamic", "fixed"], default="dynamic")
    parser.add_argument("--fixed_rounds", type=int, default=None)
    args = parser.parse_args()

    asyncio.run(main())