import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
import traceback
import json
from openai import DefaultAsyncHttpxClient, Timeout
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
from autogen_agentchat.agents import BaseChatAgent
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.ui import Console
from project_datasets.aqua_dataset import aqua_get_predict, aqua_data_process
import asyncio
from typing import List, Tuple, Dict
import numpy as np 

import builtins
import contextvars
import io
from tqdm.asyncio import tqdm_asyncio 

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
                    except:
                        pass
    else:
        print(f"[Warning] Cache file not found: {cache_file}")
                
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

    supervisor = Supervisor(
        model=args.supervisor_model,
        api_key=os.environ.get("MASRUBRIC_SUPERVISOR_API_KEY", "EMPTY"),
        base_url=args.supervisor_url,
        metrics_retrieve_k=args.metrics_retrieve_k,
        pass_rate=args.pass_rate,
        prune_flag=True,
        metric_pool_file=args.metric_pool_file, 
        embedding_cache_file=args.embedding_cache_file,
        embedding_api_key = "EMPTY",
        embedding_model = args.embedding_model,
        embedding_api_base = args.embedding_url,
        domain="math", 
        preloaded_metrics=preloaded_metrics,
        preloaded_embeddings=preloaded_embeddings,
        use_llm_rerank=use_llm,
        max_metrics_count=args.max_metrics_count,
        lock_metrics_after_first_round=args.lock_metrics_after_first_round,
        use_simple_audit=args.use_simple_audit,
        #召回调参
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
            agent_name="MathSolver_aqua",
            name=f"Participant_{i + 1}",
            domain="aqua",
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
        http_client=DefaultAsyncHttpxClient(
            trust_env=False,
            timeout=Timeout(120.0, connect=10.0),
        ),
        timeout=Timeout(120.0, connect=10.0),
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
        domain="aqua",
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
        print("\n>>> [Phase 1] 启动 Adversarial Audit 模式...")
        await team.reset()
        supervisor.reset()
        supervisor.prune_flag = True 

        await Console(team.run_stream(task=question))
        
        retained_messages = supervisor.get_messages_above_threshold()
        retained_count = len(retained_messages)
        print(f"\n[Check] 本轮保留消息数: {retained_count}")

        if retained_count <= 1:
            print(f"\n⚠️ 触发保底机制 (Fallback Triggered)！因为保留消息数 ({retained_count}) 过少。")
            print(">>> [Phase 2] 放弃本轮操作，正在重新执行 Vanilla AutoGen (无剪枝模式)...")
            await team.reset()
            supervisor.reset()
            original_prune_flag = supervisor.prune_flag
            supervisor.prune_flag = False 
            await Console(team.run_stream(task=question))
            retained_messages = supervisor.get_messages_above_threshold()
            print(f"[Fallback Result] 保底运行结束。当前保留消息数: {len(retained_messages)}")
            supervisor.prune_flag = original_prune_flag
        
        # [修复] 统一变量名
        history_messages = retained_messages

    print("\n" + "="*50)
    print("--- [DEBUG] Final Decision 阶段开始 ---")
    
    if not history_messages:
        print("  >> 警告: 历史消息为空!")
    else:
        for i, msg in enumerate(history_messages):
            if msg.source != 'user':
                print(f"  - 消息 {i+1} | 来自: {msg.source}")
    print("-" * 20)

    raw_answer = await decision_maker.run_decision(history_messages=history_messages, role_map=role_map, task=question)
    raw_content = raw_answer.content.strip()
    
    print("\n[DEBUG] 2. Final Decision 的原始输出 (raw_content):")
    print("  " + "-"*40)
    print(raw_content)
    print("  " + "-"*40)

    final_answer = aqua_get_predict(raw_content)
    print(f"  >> 解析后 (hypothesis): \"{final_answer}\"")
    
    print("\n--- [DEBUG] Final Decision 阶段结束 ---")
    print("="*50 + "\n")
    
    ret_scores = supervisor.get_scores(role_map)
    reflection_records = getattr(supervisor, 'reflection_records', [])

    return final_answer, ret_scores, reflection_records

# [修复] 改为 async 函数，并加锁
async def write_to_file(out_file, data_id, data, reflection_records):
    async with _file_write_lock:
        exist_data = {}
        if os.path.exists(out_file):
            try:
                with open(out_file, 'r') as f:
                    exist_data = json.load(f)
            except json.JSONDecodeError:
                pass
                
        exist_data[str(data_id)] = data
        sorted_keys = sorted(exist_data.keys(), key=lambda x: int(x) if x.isdigit() else x)
        sorted_data = {key: exist_data[key] for key in sorted_keys}
        
        with open(out_file, 'w') as f:
            json.dump(sorted_data, f, indent=4, ensure_ascii=False)
        
        base_dir, base_name = os.path.split(out_file)
        relection_records_file = os.path.join(base_dir, f"reflection_records_{base_name}")

        rec_data = {}
        if os.path.exists(relection_records_file):
            try:
                with open(relection_records_file, 'r') as f:
                    rec_data = json.load(f)
            except json.JSONDecodeError:
                pass
        rec_data[str(data_id)] = reflection_records
        with open(relection_records_file, 'w') as f:
            json.dump(rec_data, f, indent=4, ensure_ascii=False)
    

async def run_sample(data, out_file, team, decision_maker, role_map, supervisor):
    instance_id = data.get('id', 'unknown')
    
    # [核心修改] 根据开关决定是否启用日志缓冲
    if not args.disable_log_buffer:
        log_capture = io.StringIO()
        token = _current_log_buffer.set(log_capture)
    
    try:
        current_time = time.strftime('%Y-%m-%d %H:%M:%S')
        print(f"--- [ {current_time} ] ---")
        print(f"开始处理: {instance_id}")
        
        start_time = time.time()
        
        answer, scores, reflection_records = await reasoning(data['task'], team, decision_maker, role_map, supervisor)
        
        end_time = time.time()
        print(f"完成处理: {instance_id} (耗时: {end_time - start_time:.2f} 秒)")
        
        # [修改] 使用 await 调用异步写入函数
        await write_to_file(out_file, instance_id, {'answer': data['answer'], 'hypothesis': answer, 'question': data['task'], 'scores': scores}, reflection_records)
        
    except Exception as e:
        print(f"处理任务 {instance_id} 发生错误: {e}")
        traceback.print_exc() 
        
    finally:
        # [核心修改] 仅在启用缓冲时进行保存和重置
        if not args.disable_log_buffer:
            _global_log_store[instance_id] = log_capture.getvalue()
            log_capture.close()
            _current_log_buffer.reset(token)


async def main():
    with open(args.in_file, 'r') as f:
        input_lines = f.readlines()
        input_data = [json.loads(line) for line in input_lines]
    
    input_data = aqua_data_process(input_data)
    
    if args.limit is not None and args.limit > 0:
        print(f"\n[INFO] 🔧 测试模式启用：仅处理前 {args.limit} 个任务 (原总数: {len(input_data)})")
        input_data = input_data[:args.limit]
    
    global_metrics, global_embeddings = load_global_resources(
        args.metric_pool_file, 
        args.embedding_cache_file
    )
    
    # [修改] 使用参数控制并发
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
            team, decision_maker, role_map, supervisor = init_team(global_metrics, global_embeddings)
            await run_sample(instance, args.out_file, team, decision_maker, role_map, supervisor)

    tasks = [worker(instance) for instance in input_data]
    
    print(f"🚀 开始并发处理 {len(input_data)} 个任务 (并发度: {CONCURRENCY_LIMIT})")
    
    start_time = time.time()
    await tqdm_asyncio.gather(*tasks, desc="Processing Tasks")
    total_time = time.time() - start_time
    
    print(f"\n🎉 所有任务处理完毕，耗时: {total_time:.2f} 秒")
    
    if not args.disable_log_buffer:
        print(f"📝 详细日志将写入: {FINAL_LOG_FILE}")
        def sort_key(k):
            try: return int(k)
            except: return str(k)
            
        print(f"💾 正在将内存日志写入文件...")
        sorted_ids = sorted(_global_log_store.keys(), key=sort_key)
        
        with open(FINAL_LOG_FILE, "w", encoding="utf-8") as f:
            f.write(f"=== 并发运行日志汇总 ===\n")
            f.write(f"总任务数: {len(input_data)}\n")
            f.write(f"总耗时: {total_time:.2f}s\n")
            f.write("="*50 + "\n\n")
            
            for task_id in sorted_ids:
                log_content = _global_log_store[task_id]
                f.write(f"\n{'='*40}\n")
                f.write(f"=== TASK ID: {task_id} ===\n")
                f.write(f"{'='*40}\n")
                f.write(log_content)
                f.write("\n")
                
        print(f"✅ 完整日志已生成: {FINAL_LOG_FILE}")
    else:
        print("[INFO] 日志缓冲已禁用，请查看控制台输出。")


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_file', type=str)
    parser.add_argument('--out_file', type=str)
    parser.add_argument('--reasoning_url', type=str)
    parser.add_argument('--reasoning_model', type=str)
    parser.add_argument('--supervisor_url', type=str)
    parser.add_argument('--supervisor_model', type=str)
    parser.add_argument('--selector_model', type=str, default="Qwen/Qwen3.5-9B")
    parser.add_argument('--selector_url', type=str, default=None)
    parser.add_argument('--selector_api_key', type=str, default="EMPTY")
    
    parser.add_argument('--embedding_url', type=str, required=True)
    parser.add_argument('--embedding_model', type=str, required=True)
    parser.add_argument("--metric_pool_file", type=str, required=True)
    parser.add_argument("--embedding_cache_file", type=str, required=True)
    
    parser.add_argument('--max_turns', type=int)
    parser.add_argument('--metrics_retrieve_k', type=int)
    parser.add_argument('--pass_rate', type=float, default=0.8)
    parser.add_argument('--retries_times', type=int, default=3)
    parser.add_argument('--agent_mode', default='supervisor', choices=['supervisor', 'prm', 'self_refine', 'self-refine', 'multi_tag', 'multi-tag'])
    parser.add_argument('--prm_url', type=str, default=None)
    parser.add_argument('--prm_n_samples', type=int, default=3)
    parser.add_argument('--log_file', type=str)
    
    
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--baseline_only", action="store_true")
    parser.add_argument("--max_metrics_count", type=int, default=5)
    parser.add_argument("--lock_metrics_after_first_round", action="store_true")
    # [修改] 简单审计模式：0=关闭, 1=V1(通用), 2=V2(优化版)
    parser.add_argument("--use_simple_audit", type=int, default=0, help="0: Disable, 1: Simple V1, 2: Simple V2")
    
    # [新增] 禁用日志缓冲开关
    parser.add_argument("--disable_log_buffer", action="store_true", help="Disable in-memory log buffering (print to stdout directly).")
    # [新增] 并发控制参数
    parser.add_argument("--concurrency_limit", type=int, default=100, help="Max concurrent tasks.")

    parser.add_argument("--force_direct_search", action="store_true")
    parser.add_argument("--retrieve_p", type=int, default=20, help="Top-P candidates for reranking.")
    parser.add_argument("--select_q", type=int, default=5, help="Top-Q metrics selected by LLM.")
    parser.add_argument("--direct_k", type=int, default=5, help="Top-K metrics for direct search mode.")
    
    
    # [新增] 随机对照参数
    parser.add_argument("--random_k", type=int, default=0, help="Randomly select k metrics (Baseline Mode)")
    
    
    args = parser.parse_args()
    if not args.selector_url:
        args.selector_url = args.reasoning_url
    
    # [核心修改] 只有在未禁用缓冲时才劫持 print
    if not args.disable_log_buffer:
        builtins.print = scoped_print
    
    asyncio.run(main())
