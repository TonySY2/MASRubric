import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import argparse
import time
import traceback
import json
import asyncio
import random
from typing import List, Tuple, Dict, Any
from tqdm import tqdm

from masrubric.agents import AgentRegistry
from masrubric.agents.supervisor import Supervisor
from masrubric.agents.final_decision import FinalRefer
from autogen_agentchat.teams import SelectorGroupChat
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.ui import Console
from project_datasets.aqua_dataset import aqua_get_predict
from autogen_core.models import ModelFamily
from openai import AsyncOpenAI

# ==============================================================================

# ==============================================================================
def log_message(msg: str, log_file: str = None):
    print(msg)
    if log_file:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

def init_team() -> Tuple[SelectorGroupChat, FinalRefer, Dict[str, str], Supervisor]:
    
 
    supervisor = Supervisor(
        model=args.supervisor_model,
        api_key=args.supervisor_key,
        base_url=args.supervisor_url,
        embedding_model_name=args.embedding_model,
        embedding_base_url=args.embedding_url,
        embedding_api_key=args.embedding_key,
        supervisor_mode='collect', 
        sample_times=3,
        elite_strictness_coeff=1.0,
        warmup_rounds=2
    )

    agent_resgistry = AgentRegistry()
    participants = [
        agent_resgistry.get(
            agent_name="MathSolver_aqua",
            name=f"Participant_{i + 1}",
            domain="aqua",
            model=args.reasoning_model,
            api_key=args.reasoning_key,
            base_url=args.reasoning_url,
            supervisor=supervisor,
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

    model_client = OpenAIChatCompletionClient(
        model=args.selector_model, 
        api_key=args.selector_key, 
        base_url=args.selector_url
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

    decision_maker = AgentRegistry.get(
        agent_name="FinalRefer",
        name="DecisionMaker",
        domain="aqua",
        model=args.reasoning_model,
        api_key=args.reasoning_key,
        base_url=args.reasoning_url
    )
    
    return team, decision_maker, role_map, supervisor


async def reasoning(
    question, 
    team: SelectorGroupChat, 
    decision_maker: FinalRefer, 
    role_map: Dict[str, str], 
    supervisor: Supervisor,
    structured_context: str 
):
    
    await team.reset()
    supervisor.reset()
    
    supervisor.current_full_task_json = structured_context
    
    await Console(team.run_stream(task=question))
    
    history_messages = supervisor.get_messages_above_threshold()
    
    print("\n" + "="*50)
    print("--- [DEBUG] Final Decision ---")
    raw_answer = await decision_maker.run_decision(history_messages=history_messages, role_map=role_map, task=question)
    raw_content = raw_answer.content.strip()
    
    final_answer = aqua_get_predict(raw_content)
    
    print(f"before: {raw_content[:100]}...")
    print(f"after: {final_answer}")
    print("="*50 + "\n")
    
    ret_trajectories = supervisor.get_collected_trajectories()
    
    return final_answer, ret_trajectories


def write_to_file(out_file, data_id, data):

    exist_data = {}
    if os.path.exists(out_file):
        with open(out_file, 'r', encoding='utf-8') as f:
            try:
                exist_data = json.load(f)
            except json.JSONDecodeError:
                pass
    
    exist_data[str(data_id)] = data
    
    def smart_key(k):
        try: return int(k)
        except: return k
        
    sorted_keys = sorted(exist_data.keys(), key=smart_key)
    sorted_data = {key: exist_data[key] for key in sorted_keys}
    
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(sorted_data, f, indent=4, ensure_ascii=False)
    

async def run_sample(data, out_file, team, decision_maker, role_map, supervisor, log_file_path):
    instance_id = str(data.get('id', data.get('idx', 'unknown')))
    
    current_time = time.strftime('%Y-%m-%d %H:%M:%S')
    log_message(f"\n{'='*40}", log_file_path)
    log_message(f"=== TASK ID: {instance_id} ===", log_file_path)
    log_message(f"{'='*40}", log_file_path)
    log_message(f"--- [ {current_time} ] Start Processing ---", log_file_path)
    
    start_time = time.time()
    
    try:
  
        teacher_b_context = json.dumps(data, ensure_ascii=False)
        question_text = data.get('question', '')
        options = data.get('options', [])
        options_str = "\n".join(options)
        agent_task_prompt = f"{question_text}\nAnswer Choices:\n{options_str}"


        hypothesis, trajectories = await reasoning(
            agent_task_prompt, 
            team, 
            decision_maker, 
            role_map, 
            supervisor,
            teacher_b_context
        )
        
   
        ground_truth = str(data.get('correct', '')).strip().upper()
        prediction = str(hypothesis).strip().upper()
        is_solved = (prediction == ground_truth)
        
        execution_result = f"Prediction: {prediction}, Ground Truth: {ground_truth}"
        log_message(f"\n>>> Task Result: {is_solved} (GT: {ground_truth} | Pred: {prediction})", log_file_path)

    
        updated_trajectories = trajectories
        if not is_solved:
            log_message(f">>> [Audit] ❌ Task Failed. Triggering Teacher B Review...", log_file_path)
            updated_trajectories = await supervisor.review_failed_trajectories()
        else:
            log_message(f">>> [Audit] ✅ Task Solved. Skipping Review.", log_file_path)

        duration = time.time() - start_time
        log_message(f"over: {instance_id} (lasting: {duration:.2f} s)", log_file_path)
        
   
        write_to_file(
            out_file, 
            instance_id, 
            {
                'id': instance_id,
                'question': question_text,
                'options': options,
                'answer': data.get('rationale', ''), 
                'correct_option': ground_truth,
                'hypothesis': hypothesis, 
                'is_correct': is_solved,
                'execution_result': execution_result,
                'trajectories': updated_trajectories
            }
        )
    except Exception as e:
        log_message(f"[CRITICAL ERROR] Task {instance_id}: {e}", log_file_path)
        traceback.print_exc()
        write_to_file(
            out_file,
            instance_id,
            {'error': str(e), 'is_correct': False}
        )


async def main():
    input_file_path = args.in_file
    print(f"[DEBUG] Loading AQuA data from: {input_file_path}")
    
    if not os.path.exists(input_file_path):
        print("[ERROR] File not found.")
        return


    with open(input_file_path, "r", encoding='utf-8') as f:
        input_lines = f.readlines()
    
    raw_input_data = []
    for idx, line in enumerate(input_lines):
        line = line.strip()
        if not line: continue
        try:
            item = json.loads(line)
            if 'id' not in item and 'idx' not in item:
                item['id'] = str(idx)
            raw_input_data.append(item)
        except json.JSONDecodeError:
            pass
            
    input_data = raw_input_data
    
    if args.limit is not None and args.limit > 0:
        if args.limit < len(input_data):
            print(f"[INFO] Sampling {args.limit} tasks.")
            input_data = input_data[:args.limit] 
            
    print(f"[INFO] Total tasks to process: {len(input_data)}")


    if args.log_file:
        FINAL_LOG_FILE = args.log_file
    else:
        base_dir = os.path.dirname(args.out_file)
        base_name = os.path.basename(args.out_file).replace(".json", "_full.log")
        FINAL_LOG_FILE = os.path.join(base_dir, base_name)
    os.makedirs(os.path.dirname(FINAL_LOG_FILE), exist_ok=True)


    with open(FINAL_LOG_FILE, "w", encoding="utf-8") as f:
        f.write(f"=== AQuA Run Logs ===\n")
        f.write(f"Total Tasks: {len(input_data)}\n\n")

    print(f"🚀 (Single Thread)...")
    start_time = time.time()
    

    for instance in tqdm(input_data, desc="AQuA Tasks"):
        team, decision_maker, role_map, supervisor = init_team()
        await run_sample(instance, args.out_file, team, decision_maker, role_map, supervisor, FINAL_LOG_FILE)
    
    total_time = time.time() - start_time
    print(f"\n🎉 All tasks finished in {total_time:.2f}s")
    print("✅ Done.")


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_file', type=str, required=True)
    parser.add_argument('--out_file', type=str, required=True)
    
    parser.add_argument('--selector_url', type=str)
    parser.add_argument('--selector_model', type=str)
    parser.add_argument('--selector_key', type=str)
    
    parser.add_argument('--reasoning_url', type=str, required=True)
    parser.add_argument('--reasoning_model', type=str, required=True)
    parser.add_argument('--reasoning_key', type=str, default="EMPTY")
    
    parser.add_argument('--supervisor_url', type=str)
    parser.add_argument('--supervisor_model', type=str)
    parser.add_argument('--supervisor_key', type=str, default="EMPTY")
    
    parser.add_argument('--embedding_url', type=str, required=True)
    parser.add_argument('--embedding_model', type=str, required=True)
    parser.add_argument("--embedding_key", type=str, default="EMPTY")
    parser.add_argument('--max_turns', type=int, default=6)
    parser.add_argument('--log_file', type=str, help="Path for log file")
    

    parser.add_argument('--limit', type=int, default=None)
    
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.out_file), exist_ok=True)
    
    asyncio.run(main())
