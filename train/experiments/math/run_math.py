import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import argparse
import time
import traceback
import json
import re
from masrubric.agents import AgentRegistry
from masrubric.agents.supervisor import Supervisor
from masrubric.agents.final_decision import FinalRefer
from autogen_agentchat.teams import SelectorGroupChat
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.ui import Console
import asyncio
from typing import List, Tuple, Dict
from openai import AsyncOpenAI 
from tqdm import tqdm

try:
    from masrubric.agents.math_grader import MathGrader
except ImportError:
    print("[FATAL] ")
    sys.exit(1)




# ==============================================================================
def log_message(msg: str, log_file: str = None):
    print(msg)
    if log_file:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(msg + "\n")


# ==============================================================================

async def quick_check_by_llm(client: AsyncOpenAI, model_name: str, question, ground_truth, hypothesis):
    check_prompt = f"""
You are a Math Equivalence Checker.
Problem: {question}
Ground Truth: {ground_truth}
Agent Prediction: {hypothesis}

Task: Check if the Agent Prediction is mathematically equivalent to the Ground Truth.
- Ignore formatting differences (e.g., "14,641" vs "14641").
- Ignore order of terms (e.g., "x+1" vs "1+x").
- Ignore unit styles if values are correct.

Output ONLY "YES" if they are equivalent, or "NO" if they are different.
"""
    try:
        response = await client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": check_prompt}],
            temperature=0.0,
            max_tokens=10
        )
        content = response.choices[0].message.content.strip().upper()
        return "YES" in content
    except Exception as e:
        print(f"[QuickCheck Error] {e}, assuming False.")
        return False


def extract_boxed_answer(text: str) -> str:
    if not text: return ""
    match = re.search(r'\\boxed{([^{}]*)}', text)
    if match:
        return match.group(1)
    return ""

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
        warmup_rounds=2,            
        elite_strictness_coeff=1.0  
    )

    agent_resgistry = AgentRegistry()
    participants = [
        agent_resgistry.get(
            agent_name="MathSolver_math500",  
            name=f"Participant_{i + 1}",
            domain="math500",                    
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
        domain="math500",
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
    
    print("\n" + "="*50)
    print("--- [DEBUG] Final Decision  ---")
    
    history_messages = supervisor.get_messages_above_threshold()
    
    raw_answer = await decision_maker.run_decision(history_messages=history_messages, role_map=role_map, task=question)
    raw_content = raw_answer.content.strip()
    
    print("\n[DEBUG] Final Decision :")
    print(raw_content)
    print("="*50 + "\n")
    
    ret_trajectories = supervisor.get_collected_trajectories()
    
    return raw_content, ret_trajectories

def write_to_file(out_file, data_id, data):
    exist_data = {}
    if os.path.exists(out_file):
        with open(out_file, 'r', encoding='utf-8') as f:
            try:
                exist_data = json.load(f)
            except json.JSONDecodeError:
                pass
    
    exist_data[str(data_id)] = data
    sorted_keys = sorted(exist_data.keys(), key=lambda x: int(x) if x.isdigit() else x)
    sorted_data = {key: exist_data[key] for key in sorted_keys}
    
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(sorted_data, f, indent=4, ensure_ascii=False)
    

async def run_sample(data, out_file, team, decision_maker, role_map, supervisor, log_file_path):
    question = data.get('problem', data.get('question', ''))
    instance_id = str(data.get('id', 'unknown')) 
    unique_id = data.get('unique_id', 'unknown') 
    ground_truth = data.get('solution', data.get('answer', ''))
    
    teacher_b_context = json.dumps(data, ensure_ascii=False)
    
    check_client = AsyncOpenAI(api_key=args.reasoning_key, base_url=args.reasoning_url)
    
    try:
        current_time = time.strftime('%Y-%m-%d %H:%M:%S')
        log_message(f"\n{'='*40}", log_file_path)
        log_message(f"=== TASK ID: {instance_id} ===", log_file_path)
        log_message(f"{'='*40}", log_file_path)
        log_message(f"--- [ {current_time} ] Start Processing ---", log_file_path)


        raw_content, trajectories = await reasoning(
            question, team, decision_maker, role_map, supervisor, teacher_b_context
        )
        
        hypothesis = MathGrader.extract_answer(raw_content)
        is_correct = MathGrader.check_correctness(raw_content, ground_truth)
        

        final_verdict = is_correct
        if not is_correct:
            log_message(f">>> [Grader] Rule-based check failed. Requesting LLM Quick Check...", log_file_path)
            llm_is_correct = await quick_check_by_llm(
                check_client, 
                args.reasoning_model, 
                question, 
                ground_truth, 
                hypothesis
            )
            
            if llm_is_correct:
                log_message(f">>> [QuickCheck] LLM says: EQUIVALENT! Overriding Grader result.", log_file_path)
                is_correct = True
                final_verdict = True
            else:
                log_message(f">>> [QuickCheck] LLM says: DIFFERENT. Confirming failure.", log_file_path)
                final_verdict = False
        
        log_message(f"Task Result: {final_verdict} (GT: {ground_truth} | Pred: {hypothesis})", log_file_path)

        updated_trajectories = trajectories
   
        if not final_verdict:
            log_message(f">>> [Audit] ❌ Task Failed. Triggering Teacher B Review (Generating Metrics)...", log_file_path)
            updated_trajectories = await supervisor.review_failed_trajectories()
        else:
            log_message(f">>> [Audit] ✅ Task Solved. Skipping Review.", log_file_path)
            
        write_to_file(
            out_file, 
            instance_id, 
            {
                'unique_id': unique_id,
                'id': instance_id, 
                'answer': ground_truth,
                'hypothesis': hypothesis, 
                'question': question,
                'raw_response': raw_content,
                'is_correct': is_correct,
                'scores': {}, 
                'trajectories': updated_trajectories 
            }
        )
    except Exception as e:
        log_message(f"!!!!!! [CRITICAL ERROR] Task {instance_id}: {e} !!!!!!", log_file_path)
        traceback.print_exc()


async def main():
    if not os.path.exists(args.in_file):
        print(f"[ERROR] Input file not found: {args.in_file}")
        return

    with open(args.in_file, 'r', encoding='utf-8') as f:
        input_lines = f.readlines()
    
    raw_data = []
    for line in input_lines:
        line = line.strip()
        if line:
            try:
                raw_data.append(json.loads(line))
            except json.JSONDecodeError:
                pass

    def get_id(item):
        try: return int(item.get('id', float('inf')))
        except: return float('inf')
            
    input_data = sorted(raw_data, key=get_id)
    
    if args.limit is not None and args.limit > 0:
        print(f"[INFO] Limit applied: Processing the first {args.limit} tasks.")
        input_data = input_data[:args.limit]
    
    print(f"[INFO] waiting: {len(input_data)}")
    
    if args.log_file:
        FINAL_LOG_FILE = args.log_file
    else:
        base_dir = os.path.dirname(args.out_file)
        base_name = os.path.basename(args.out_file).replace(".json", "_full.log")
        FINAL_LOG_FILE = os.path.join(base_dir, base_name)
    
    os.makedirs(os.path.dirname(FINAL_LOG_FILE), exist_ok=True)
    

    with open(FINAL_LOG_FILE, "w", encoding="utf-8") as f:
        f.write(f"=== Math500 Training Run Logs ===\n")
        f.write(f"Total Tasks: {len(input_data)}\n\n")


    
    start_time = time.time()
    

    for instance in tqdm(input_data, desc="Training Tasks"):
        team, decision_maker, role_map, supervisor = init_team()
        await run_sample(instance, args.out_file, team, decision_maker, role_map, supervisor, FINAL_LOG_FILE)
    
    total_time = time.time() - start_time
    print(f"\n🎉 : {total_time:.2f}s")
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_file', type=str)
    parser.add_argument('--out_file', type=str)
    
    parser.add_argument('--selector_url', type=str)
    parser.add_argument('--selector_model', type=str)
    parser.add_argument('--selector_key', type=str)
    
    
    parser.add_argument('--reasoning_url', type=str)
    parser.add_argument('--reasoning_model', type=str)
    parser.add_argument('--reasoning_key', type=str, default="EMPTY")
    

    parser.add_argument('--supervisor_url', type=str)
    parser.add_argument('--supervisor_model', type=str)
    parser.add_argument('--supervisor_key', type=str, default="EMPTY")
    
    parser.add_argument('--embedding_url', type=str, required=True)
    parser.add_argument('--embedding_model', type=str, required=True)
    parser.add_argument("--embedding_key", type=str, default="EMPTY")
    
    parser.add_argument('--max_turns', type=int)
    
    parser.add_argument('--limit', type=int, default=None, help="Process only the first N tasks.")
    parser.add_argument('--log_file', type=str, help="Path for the final aggregated log file")
    
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.out_file), exist_ok=True)
    
    asyncio.run(main())
