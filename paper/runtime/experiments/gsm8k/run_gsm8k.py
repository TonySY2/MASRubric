import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import argparse
import time
import traceback
import json
import re
from masrubric.agents import AgentRegistry
from masrubric.agents.supervisor_reasoning_pick_metric import Supervisor
from masrubric.agents.final_decision import FinalRefer
from autogen_agentchat.teams import SelectorGroupChat
from masrubric.usage_tracking import TrackedOpenAIChatCompletionClient
from masrubric.agents.agent_mode_helpers import (
    attach_agent_mode_state,
    configure_agent_mode_team,
    get_agent_mode_history,
    normalize_agent_mode,
    reset_agent_mode_state,
)
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.ui import Console
import asyncio
from typing import List, Tuple, Dict
import random
import builtins
import contextvars
import io
from tqdm.asyncio import tqdm_asyncio 
from openai import Timeout
import numpy as np 

# [核心依赖]
try:
    from masrubric.agents.math_grader import MathGrader
except ImportError:
    print("[FATAL] 找不到 masrubric.agents.math_grader，请确认文件位置。")
    sys.exit(1)

# ==========================================
# [优雅方案] 内存日志缓冲系统 (带开关)
# ==========================================
_current_log_buffer = contextvars.ContextVar('current_log_buffer', default=None)
_global_log_store = {}
_original_print = builtins.print

# [新增] 全局文件写入锁，防止并发写入导致数据丢失
_file_write_lock = asyncio.Lock()

def scoped_print(*args, **kwargs):
    buffer = _current_log_buffer.get()
    if buffer:
        kwargs['file'] = buffer
        _original_print(*args, **kwargs)
    else:
        _original_print(*args, **kwargs)
        
        
def gsm8k_extract_answer(text: str) -> str:
    """[增强版] 专用于 GSM8K 的答案提取"""
    if not text: return ""
    clean_text = text.replace("$", "").replace("%", "")
    
    # 1. 优先匹配 "The answer is X"
    pattern_explicit = r"[Tt]he answer is\s*(-?[\d,]+(?:\.\d+)?)"
    match_explicit = re.search(pattern_explicit, clean_text)
    if match_explicit:
        return match_explicit.group(1).replace(",", "")

    # 2. 匹配 \boxed{X} (兼容 MATH 风格输出)
    pattern_boxed = r"\\boxed\{([^}]+)\}"
    match_boxed = re.search(pattern_boxed, text)
    if match_boxed:
        return match_boxed.group(1).strip()

    # 3. 贪婪匹配最后一个数字
    matches = re.findall(r'-?\d+(?:,\d+)*(?:\.\d+)?', clean_text)
    if matches:
        return matches[-1].replace(",", "")
    
    return ""

def gsm8k_is_correct(pred: str, gt: str) -> bool:
    if not pred or not gt: return False
    try:
        return abs(float(pred) - float(gt)) < 1e-6
    except ValueError:
        return pred.strip() == gt.strip()

# ==========================================
# [GSM8K 专用数据适配] (增强版)
# ==========================================
def extract_gsm8k_gold(text: str) -> str:
    """从 GSM8K 的 answer 字段中提取 #### 后的最终数值"""
    if "####" in text:
        return text.split("####")[1].strip()
    return text.strip()

def prepare_gsm8k_data(item: dict) -> dict | None:
    """
    适配 GSM8K 数据格式 (增强健壮性)
    兼容多种字段名: 
    - Question: question, problem, task
    - Answer: answer, solution
    """
    
    # 1. 尝试提取 Question
    question = item.get('question') or item.get('problem') or item.get('task')
    
    # 2. 尝试提取 Answer
    raw_answer = item.get('answer') or item.get('solution') or item.get('ground_truth')
    
    # 如果两者任一缺失，则视为无效数据
    if not question or not raw_answer:
        # 可选：打印警告日志
        # print(f"[WARNING] Skipping item due to missing fields. ID: {item.get('id', 'unknown')}")
        return None
    
    # 3. 提取 Gold Value (纯数值)
    gold_answer = extract_gsm8k_gold(raw_answer)
    
    # 4. 兼容 ID 字段
    task_id = item.get('id', item.get('question_id', item.get('unique_id', 'unknown')))
    
    return {
        "id": task_id,
        "question": question,
        "gold_answer": gold_answer,       # 仅包含最终数值 (e.g. "18")，用于比对
        "canonical_solution": raw_answer, # 完整推理过程 (e.g. "...#### 18")，用于记录
        "original_data": item
    }

# ==========================================
# [全局资源加载]
# ==========================================
def load_global_resources(metric_file, cache_file):
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
    else:
        print("[Warning] Cache file not found.")
                
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


def init_team(preloaded_metrics, preloaded_embeddings) -> Tuple[SelectorGroupChat, FinalRefer, Dict[str, str], Supervisor]:
    
    use_llm = not args.force_direct_search
    
    # GSM8K 也是数学任务，Domain 设为 math
    supervisor = Supervisor(
        model=args.supervisor_model,
        api_key=os.environ.get("MASRUBRIC_SUPERVISOR_API_KEY", "EMPTY"), 
        base_url=args.supervisor_url,
        domain="math", 
        metrics_retrieve_k=args.metrics_retrieve_k,
        pass_rate=args.pass_rate,
        prune_flag=True, 
        metric_pool_file=args.metric_pool_file, 
        embedding_cache_file=args.embedding_cache_file,
        
        embedding_api_key=os.environ.get("MASRUBRIC_EMBEDDING_API_KEY", "EMPTY"),
        embedding_model=args.embedding_model,
        embedding_api_base=args.embedding_url,
        
        preloaded_metrics=preloaded_metrics,
        preloaded_embeddings=preloaded_embeddings,
        
        use_llm_rerank=use_llm, 
        
        # [新增参数对齐]
        max_metrics_count=args.max_metrics_count,
        lock_metrics_after_first_round=args.lock_metrics_after_first_round,
        use_simple_audit=args.use_simple_audit,
        force_direct_search=args.force_direct_search,
        direct_k=args.direct_k,
        retrieve_p=args.retrieve_p,
        select_q=args.select_q,
        
        # [新增] 传递随机参数
        random_k=args.random_k,
    )

    agent_resgistry = AgentRegistry()
    participants = [
        agent_resgistry.get(
            agent_name="MathSolver_gsm8k",  
            name=f"Participant_{i + 1}",
            domain="gsm8k",                    
            model=args.reasoning_model,
            api_key=os.environ.get("MASRUBRIC_REASONING_API_KEY", "EMPTY"),
            base_url=args.reasoning_url,
            supervisor=supervisor,
            reflection_time=args.retries_times,
        ) for i in range(5)
    ]

    shared_score_board, predictor = configure_agent_mode_team(
        participants,
        agent_mode=args.agent_mode,
        prm_url=args.prm_url,
        prm_n_samples=args.prm_n_samples,
        model=args.reasoning_model,
        api_key=os.environ.get("MASRUBRIC_REASONING_API_KEY", "EMPTY"),
        base_url=args.reasoning_url,
    )

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
        api_key=args.selector_api_key,
        base_url=args.selector_url,
        http_client_args={"timeout": Timeout(120.0, connect=10.0)},
        max_retries=5,
        usage_stage="selector",
        usage_source="SelectorGroupChat",
    )
    
    text_mention_termination = TextMentionTermination("TERMINATE")
    max_messages_termination = MaxMessageTermination(max_messages=args.max_turns)
    termination = text_mention_termination | max_messages_termination

    team = SelectorGroupChat(
        participants=participants,
        model_client=model_client,
        termination_condition=termination,
        selector_prompt=selector_prompt,
        allow_repeated_speaker=True,  
    )
    attach_agent_mode_state(team, shared_score_board, predictor)

    decision_maker = AgentRegistry.get(
        agent_name="FinalRefer",
        name="DecisionMaker",
        domain="gsm8k",
        model=args.reasoning_model,
        api_key=os.environ.get("MASRUBRIC_REASONING_API_KEY", "EMPTY"),
        base_url=args.reasoning_url
    )
    
    return team, decision_maker, role_map, supervisor


async def reasoning(question, team: SelectorGroupChat, decision_maker: FinalRefer, role_map: Dict[str, str], supervisor: Supervisor):
    
    is_baseline_mode = getattr(args, 'baseline_only', False)

    agent_mode = normalize_agent_mode(getattr(args, 'agent_mode', 'supervisor'))

    if agent_mode != 'supervisor':
        print(f"\n>>> [Mode] Agent comparison mode: {agent_mode}")
        await team.reset()
        reset_agent_mode_state(team)
        supervisor.reset()
        supervisor.prune_flag = False
        await Console(team.run_stream(task=question))
        history_messages = get_agent_mode_history(team)
        print(f"[Agent Mode Result] collected messages: {len(history_messages)}")

    elif is_baseline_mode:
        print(f"\n>>> [Mode] Baseline Only (No Audit / No Pruning)")
        await team.reset()
        supervisor.reset()
        supervisor.prune_flag = False 
        
        await Console(team.run_stream(task=question))
        history_messages = supervisor.get_messages_above_threshold()

    else:
        print(f"\n>>> [Phase 1] 启动 Adversarial Audit 模式 (Task: {question[:30]}...)")
        
        await team.reset()
        supervisor.reset()
        supervisor.prune_flag = True 
        
        await Console(team.run_stream(task=question))
        
        history_messages = supervisor.get_messages_above_threshold()
        retained_count = len(history_messages)
        print(f"\n[Check] 本轮保留消息数: {retained_count}")
        
        # 保底机制
        if retained_count <= 1: 
            print(f"\n⚠️ 触发保底机制 (Fallback Triggered)！")
            print(">>> [Phase 2] 放弃本轮操作，重新执行无剪枝模式...")
            
            await team.reset()
            supervisor.reset()
            original_prune_flag = supervisor.prune_flag
            supervisor.prune_flag = False 
            
            await Console(team.run_stream(task=question))
            history_messages = supervisor.get_messages_above_threshold()
            print(f"[Fallback Result] 保底运行结束。")
            supervisor.prune_flag = original_prune_flag
    
    print("\n" + "="*50)
    print("--- [DEBUG] Final Decision 阶段开始 ---")
    
    if not history_messages:
        print("  >> 警告: 历史消息为空!")
    else:
        for i, msg in enumerate(history_messages):
            if msg.source != 'user':
                print(f"  - 消息 {i+1} | 来自: {msg.source}")

    raw_answer = await decision_maker.run_decision(history_messages=history_messages, role_map=role_map, task=question)
    raw_content = raw_answer.content.strip()
    
    print("\n[DEBUG] 原始输出预览:")
    print(raw_content[:200] + "...") 

    preview_answer = MathGrader.extract_answer(raw_content)
    print(f"解析预览 (hypothesis): \"{preview_answer}\"")
    print("="*50 + "\n")
    
    ret_scores = supervisor.get_scores(role_map)
    reflection_records = getattr(supervisor, 'reflection_records', [])
    
    return raw_content, ret_scores, reflection_records

# [修复] 改为 async 函数，并加锁
async def write_to_file(out_file, data_id, data):
    async with _file_write_lock:
        exist_data = {}
        if os.path.exists(out_file):
            try:
                with open(out_file, 'r', encoding='utf-8') as f:
                    exist_data = json.load(f)
            except json.JSONDecodeError:
                pass
        
        exist_data[str(data_id)] = data
        sorted_keys = sorted(exist_data.keys(), key=lambda x: int(x) if x.isdigit() else x)
        sorted_data = {key: exist_data[key] for key in sorted_keys}
        
        with open(out_file, 'w', encoding='utf-8') as f:
            json.dump(sorted_data, f, indent=4, ensure_ascii=False)
    

async def run_sample(data, out_file, team, decision_maker, role_map, supervisor):
    instance_id = str(data.get('id', 'unknown')) 
    question = data['question']
    gold_answer = data['gold_answer'] # 纯净的数值
    canonical_solution = data['canonical_solution'] # 完整过程
    
    # [核心修改] 根据开关决定是否启用日志缓冲
    if not args.disable_log_buffer:
        log_capture = io.StringIO()
        token = _current_log_buffer.set(log_capture)
    
    try:
        current_time = time.strftime('%Y-%m-%d %H:%M:%S')
        print(f"--- [ {current_time} ] ---")
        print(f"开始处理: {instance_id}")

        raw_content, scores, reflection_records = await reasoning(question, team, decision_maker, role_map, supervisor)
        
        # [修改] 使用增强版提取和判分
        hypothesis = gsm8k_extract_answer(raw_content)
        is_correct = gsm8k_is_correct(hypothesis, gold_answer)
        
        # [可选] 保留 MathGrader 作为参考/日志，但不作为最终判定
        # mg_hyp = MathGrader.extract_answer(raw_content) 
        
        print(f"完成处理: {instance_id} | Correct: {is_correct} (GT: {gold_answer} | Pred: {hypothesis})")
        
        await write_to_file(
            out_file, 
            instance_id, 
            {
                'id': instance_id, 
                'answer': canonical_solution, # 写入完整答案供参考
                'gold_value': gold_answer,    # 写入纯净值
                'hypothesis': hypothesis, 
                'question': question,
                'raw_response': raw_content, 
                'is_correct': is_correct,   
                'scores': scores,
                'reflection_records': reflection_records
            }
        )
    except Exception as e:
        _original_print(f"!!!!!! [CRITICAL ERROR] Task {instance_id}: {e} !!!!!!")
        traceback.print_exc()
        
    finally:
        if not args.disable_log_buffer:
            _global_log_store[instance_id] = log_capture.getvalue()
            log_capture.close()
            _current_log_buffer.reset(token)


async def main():
    if not os.path.exists(args.in_file):
        print(f"[ERROR] Input file not found: {args.in_file}")
        return

    # 1. 加载数据
    with open(args.in_file, 'r', encoding='utf-8') as f:
        input_lines = f.readlines()
        
    raw_data = [json.loads(line) for line in input_lines if line.strip()]
    
    # 2. 适配数据
    adapted_data = []
    for item in raw_data:
        res = prepare_gsm8k_data(item)
        if res:
            adapted_data.append(res)
            
    if args.limit is not None and args.limit > 0:
        adapted_data = adapted_data[:args.limit]
        print(f"[INFO] 截取前 {len(adapted_data)} 个任务")

    existing_ids: set[str] = set()
    if os.path.exists(args.out_file):
        try:
            with open(args.out_file, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
            if isinstance(existing_data, dict):
                existing_ids = {str(k) for k in existing_data.keys()}
            elif isinstance(existing_data, list):
                for row in existing_data:
                    if isinstance(row, dict) and "id" in row:
                        existing_ids.add(str(row["id"]))
            if existing_ids:
                before_count = len(adapted_data)
                adapted_data = [
                    item for item in adapted_data
                    if str(item.get('id', 'unknown')) not in existing_ids
                ]
                print(
                    f"[INFO] Resume mode: detected {len(existing_ids)} existing results, "
                    f"remaining {len(adapted_data)}/{before_count} tasks."
                )
        except Exception as e:
            print(f"[WARNING] Failed to load existing results for resume: {e}")
        
    # 3. 全局资源加载
    global_metrics, global_embeddings = load_global_resources(
        args.metric_pool_file, 
        args.embedding_cache_file
    )
    
    # 4. 并发设置
    CONCURRENCY_LIMIT = args.concurrency_limit 
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    
    if args.log_file:
        FINAL_LOG_FILE = args.log_file
    else:
        base_dir = os.path.dirname(args.out_file)
        base_name = os.path.basename(args.out_file).replace(".json", "_full.log")
        FINAL_LOG_FILE = os.path.join(base_dir, base_name)
    
    os.makedirs(os.path.dirname(FINAL_LOG_FILE), exist_ok=True)

    async def worker(instance):
        async with semaphore:
            team, decision_maker, role_map, supervisor = init_team(
                global_metrics, 
                global_embeddings
            )
            await run_sample(instance, args.out_file, team, decision_maker, role_map, supervisor)

    print(f"🚀 开始并发处理 {len(adapted_data)} 个任务 (并发度: {CONCURRENCY_LIMIT})")
    
    start_time = time.time()
    
    tasks = [worker(instance) for instance in adapted_data]
    await tqdm_asyncio.gather(*tasks, desc="GSM8K Tasks")
    
    total_time = time.time() - start_time
    print(f"\n🎉 任务完成! 总耗时: {total_time:.2f}s")
    
    # 4. 写入日志
    if not args.disable_log_buffer:
        print(f"📝 详细日志将写入: {FINAL_LOG_FILE}")
        def sort_key(k):
            try: return int(k)
            except: return str(k)
        
        print(f"💾 正在写入日志到文件...")
        sorted_ids = sorted(_global_log_store.keys(), key=sort_key)
        
        with open(FINAL_LOG_FILE, "w", encoding="utf-8") as f:
            f.write(f"=== GSM8K Run Logs ===\n")
            f.write(f"Total Tasks: {len(adapted_data)}\n")
            f.write(f"Total Time: {total_time:.2f}s\n\n")
            
            for task_id in sorted_ids:
                log_content = _global_log_store[task_id]
                f.write(f"\n{'='*40}\n")
                f.write(f"=== TASK ID: {task_id} ===\n")
                f.write(f"{'='*40}\n")
                f.write(log_content)
                f.write("\n")
                
        print(f"✅ 日志写入完毕: {FINAL_LOG_FILE}")
    else:
        print("[INFO] 日志缓冲已禁用，请查看控制台输出。")


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_file', type=str, required=True)
    parser.add_argument('--out_file', type=str, required=True)
    parser.add_argument('--reasoning_url', type=str)
    parser.add_argument('--reasoning_model', type=str)
    parser.add_argument('--supervisor_url', type=str) 
    parser.add_argument('--supervisor_model', type=str)
    parser.add_argument('--selector_model', type=str, default="Qwen/Qwen3.5-9B")
    parser.add_argument('--selector_url', type=str, default=None)
    parser.add_argument('--selector_api_key', type=str, default="EMPTY")
    
    # RAG 参数
    parser.add_argument("--embedding_url", type=str, required=True)
    parser.add_argument("--embedding_model", type=str, required=True)
    parser.add_argument("--metric_pool_file", type=str, required=True)
    parser.add_argument("--embedding_cache_file", type=str, required=True)
    parser.add_argument("--metrics_retrieve_k", type=int, default=20)
    parser.add_argument("--pass_rate", type=float, default=0.8)
    
    parser.add_argument('--max_turns', type=int, default=5)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--log_file', type=str)
    
    # [新增] 对齐 math500 的控制参数
    parser.add_argument("--baseline_only", action="store_true", help="Run only in baseline mode (no pruning/audit).")
    parser.add_argument("--max_metrics_count", type=int, default=5)
    parser.add_argument("--lock_metrics_after_first_round", action="store_true")
    # [修改] 简单审计模式：0=关闭, 1=V1(通用), 2=V2(优化版)
    parser.add_argument("--use_simple_audit", type=int, default=0, help="0: Disable, 1: Simple V1, 2: Simple V2")
    parser.add_argument("--disable_log_buffer", action="store_true")
    parser.add_argument("--concurrency_limit", type=int, default=100)
    
    parser.add_argument("--force_direct_search", action="store_true")
    parser.add_argument("--retrieve_p", type=int, default=20)
    parser.add_argument("--select_q", type=int, default=5)
    parser.add_argument("--direct_k", type=int, default=5)
    
    # [新增] 随机对照参数
    parser.add_argument("--random_k", type=int, default=0, help="Randomly select k metrics (Baseline Mode)")
    parser.add_argument('--retries_times', type=int, default=3)
    parser.add_argument('--agent_mode', default='supervisor', choices=['supervisor', 'prm', 'self_refine', 'self-refine', 'multi_tag', 'multi-tag'])
    parser.add_argument('--prm_url', type=str, default=None)
    parser.add_argument('--prm_n_samples', type=int, default=3)
    
    args = parser.parse_args()
    if not args.selector_url:
        args.selector_url = args.reasoning_url
    
    # [核心修改] 只有在未禁用缓冲时才劫持 print
    if not args.disable_log_buffer:
        builtins.print = scoped_print
    
    os.makedirs(os.path.dirname(args.out_file), exist_ok=True)
    
    asyncio.run(main())
