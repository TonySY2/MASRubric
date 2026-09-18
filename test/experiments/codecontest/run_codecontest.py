import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import argparse
import time
import traceback
import json
from AgentDropout.agents import AgentRegistry
from AgentDropout.agents.supervisor_reasoning_pick_metric import Supervisor
from AgentDropout.agents.final_decision import FinalWriteCodeMBPP
from autogen_agentchat.teams import SelectorGroupChat
from AgentDropout.usage import TrackedOpenAIChatCompletionClient, set_usage_phase
from AgentDropout.teams import create_team
from AgentDropout.run_support import UsageRun, attach_usage, mark_sample_failed, sample_id
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.ui import Console
import asyncio
from typing import List, Tuple, Dict
import re
import subprocess
import tempfile
import numpy as np
from tqdm import tqdm


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

def prepare_codecontest_data(item: dict, idx: int) -> dict:
    if 'description' in item: 
        desc = item['description']
    elif 'problem' in item:   
        desc = item['problem']
    else:
        return None

    task_id = item.get('name', item.get('id', str(idx)))
    
    tests = {}
    if 'public_tests' in item and item['public_tests']:
        tests = item['public_tests']
    elif 'tests' in item and item['tests']:
        tests = item['tests']
    
    if isinstance(tests, str):
        try:
            tests = json.loads(tests)
        except:
            tests = {'inputs': [], 'outputs': []}

    return {
        "id": str(task_id),
        "prompt": desc, 
        "tests": tests,
        "original_data": item
    }

def extract_code_from_response(response):
    if not isinstance(response, str): return ""
    content = response.strip()
    if "```python" in content:
        return content.split("```python")[1].split("```")[0].strip()
    elif "```" in content:
        return content.split("```")[1].split("```")[0].strip()
    if "def " in content or "import " in content or "print(" in content:
        return content
    return ""


# ==============================================================================
class CodeContestExecutor:
    def __init__(self, timeout: int = 2): 
        self.timeout = timeout

    def normalize(self, output: str) -> str:
        if not output: return ""
        return "\n".join([line.rstrip() for line in output.strip().splitlines()]).strip()

    def execute(self, code: str, tests: Dict[str, List[str]]) -> Tuple[bool, str, str]:
        if not code:
            return False, "No code generated", ""
            
        inputs = tests.get('inputs', [])
        outputs = tests.get('outputs', [])
        
        if not inputs or not outputs:
            return False, "No test cases found", ""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as tmp_file:
            tmp_file.write(code)
            tmp_file_path = tmp_file.name

        passed_count = 0
        total_count = len(inputs)
        first_error = ""

        try:
            for i in range(total_count):
                in_str = inputs[i]
                expected_out = outputs[i]
                
                try:
                    process = subprocess.run(
                        [sys.executable, tmp_file_path],
                        input=in_str.encode('utf-8'),
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        timeout=self.timeout
                    )
                    
                    if process.returncode != 0:
                        err = process.stderr.decode('utf-8', errors='ignore')
                        if not first_error: first_error = f"Runtime Error on case {i}: {err.strip()}"
                        continue

                    actual_out = process.stdout.decode('utf-8', errors='ignore')
                    
                    if self.normalize(actual_out) == self.normalize(expected_out):
                        passed_count += 1
                    else:
                        if not first_error: 
                            short_act = actual_out[:100] + "..." if len(actual_out) > 100 else actual_out
                            first_error = f"Wrong Answer on case {i}. Got: {short_act}"
                
                except subprocess.TimeoutExpired:
                    if not first_error: first_error = f"TLE on case {i}"
                except Exception as e:
                    return False, "System Error", str(e)

            is_solved = (passed_count == total_count)
            result_str = f"Passed {passed_count}/{total_count} cases"
            return is_solved, result_str, first_error

        finally:
            if os.path.exists(tmp_file_path):
                os.remove(tmp_file_path)

# ==============================================================================
def init_team(preloaded_metrics, preloaded_embeddings) -> Tuple[SelectorGroupChat, FinalWriteCodeMBPP, Dict[str, str], Supervisor]:
    
    supervisor = Supervisor(
        model=args.supervisor_model,
        api_key=args.supervisor_key,
        base_url=args.supervisor_url,
        domain="code",  # Code Domain
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
            agent_name="CodeWriting_codecontest", 
            name=f"Participant_{i + 1}",
            domain="codecontest",                 
            model=args.reasoning_model,
            api_key=args.reasoning_key,
            base_url=args.reasoning_url,
            supervisor=supervisor,
            reflection_time=args.retries_times,
        )
        for i in range(5)
    ]

    role_map = {agent.name: agent.role for agent in participants}
    for agent in participants:
        agent.role_map = role_map

    selector_prompt = """You are the team's scrum master. Select the next agent to advance the task.
    ... (Prompt remains same) ...
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
        domain="codecontest", 
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
    
    raw_answer = await decision_maker.run_decision(
        history_messages=history_messages, role_map=role_map, task=question
    )
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
    
        sorted_keys = sorted(exist_data.keys(), key=lambda x: int(x.split('_')[0]) if '_' in x and x.split('_')[0].isdigit() else x)
    except:
        sorted_keys = sorted(exist_data.keys())
        
    sorted_data = {key: exist_data[key] for key in sorted_keys}

    with open(out_file, "w") as f:
        json.dump(sorted_data, f, indent=4, ensure_ascii=False)

async def run_sample(data, out_file, team, decision_maker, role_map, supervisor, log_file_path):
    instance_id = data.get("id", "unknown")
    
    try:
        current_time = time.strftime('%Y-%m-%d %H:%M:%S')
        log_message(f"\n{'='*40}", log_file_path)
        log_message(f"=== TASK ID: {instance_id} ===", log_file_path)
        log_message(f"{'='*40}", log_file_path)
        log_message(f"--- [ {current_time} ] Start Processing ---", log_file_path)

        task = data["prompt"] 
        tests = data["tests"]
        
        answer, scores, reflection_records = await reasoning(
            task, team, decision_maker, role_map, supervisor
        )

        is_solved = False
        execution_result = ""
        error = ""

        if answer and tests:
            try:
                executor = CodeContestExecutor(timeout=2)
                is_solved, execution_result, error = executor.execute(answer, tests)
            except Exception as e:
                mark_sample_failed(e)
                is_solved = False
                execution_result = "Executor Error"
                error = str(e)

        log_message(f"Finished: {instance_id} | Solved: {is_solved} | {execution_result}", log_file_path)

        write_to_file(
            out_file,
            instance_id,
            {
                "id": instance_id,
                "hypothesis": answer,
                "is_solved": is_solved,
                "execution_result": execution_result,
                "error": error,
                "prompt": task,
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
        adapted_data = []
        with open(args.in_file, "r", encoding='utf-8') as f:
            lines = f.readlines()
            for idx, line in enumerate(lines):
                if not line.strip(): continue
                item = json.loads(line)
                res = prepare_codecontest_data(item, idx)
                if res: adapted_data.append(res)
        
        if args.limit is not None and args.limit > 0:
            print(f"[INFO] 🔧 Limit applied: Processing first {args.limit} tasks.")
            adapted_data = adapted_data[:args.limit]
        
        print(f"[INFO] Loaded {len(adapted_data)} tasks.")

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
        f.write(f"=== CodeContest Run Logs ===\n")
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
    parser.add_argument("--in_file", type=str, required=True)
    parser.add_argument("--out_file", type=str, required=True)
    
    
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