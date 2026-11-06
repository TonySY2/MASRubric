#!/usr/bin/env python3
"""Single-agent sampling and answer voting under an explicit total-token cap.

Requests contain only the problem and optional multiple-choice options. The
budget includes prompt and completion tokens for every generation. A tokenizer
endpoint and complete usage records are required; unavailable usage stops a run.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
DATASETS = {'math500': 'math500/test_ids.jsonl', 'aqua': 'aqua/aqua_id.jsonl'}
SYSTEM = 'Solve the problem step by step. Put your final answer in \\boxed{...}.'


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, obj):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temp.replace(path)


def messages(row, benchmark):
    if benchmark == 'math500':
        question = row['problem']
    elif benchmark == 'aqua':
        question = row['question'] + '\n' + '\n'.join(row['options'])
        question += '\nReturn the selected option letter A, B, C, D or E in the box.'
    else:
        raise ValueError('Unsupported benchmark')
    return [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': question}]


def boxed_answer(text):
    answers = []
    for match in re.finditer(r'\\boxed\s*\{', text):
        depth = 1
        for end in range(match.end(), len(text)):
            depth += (text[end] == '{') - (text[end] == '}')
            if depth == 0:
                answers.append(text[match.end():end].strip())
                break
    return answers[-1] if answers else None


def vote_key(answer):
    if answer is None:
        return None
    value = answer.strip().replace(r'\left', '').replace(r'\right', '').replace('$', '')
    value = re.sub(r'\s+', '', value).replace(r'\dfrac', r'\frac').replace(r'\tfrac', r'\frac')
    fraction = re.fullmatch(r'\\frac\{(-?\d+)\}\{(-?\d+)\}', value)
    if fraction:
        value = fraction[1] + '/' + fraction[2]
    try:
        return str(Fraction(value))
    except (ValueError, ZeroDivisionError):
        return value or None


def select_vote(candidates):
    """Gold-free normalized exact voting; ties choose the earliest sample."""
    keys = [vote_key(c.get('answer')) for c in candidates]
    counts = Counter(k for k in keys if k is not None)
    if not counts:
        return {'index': None, 'answer': None, 'votes': 0, 'parseable': 0}
    best = max(counts.values())
    index = next(i for i, key in enumerate(keys) if key is not None and counts[key] == best)
    return {'index': index, 'answer': candidates[index]['answer'], 'votes': best,
            'parseable': sum(k is not None for k in keys)}


def generation_limit(budget, spent, prompt_tokens, maximum, context_length):
    return max(0, min(maximum, budget - spent - prompt_tokens, context_length - prompt_tokens))


def valid_usage(value):
    names = ('prompt_tokens', 'completion_tokens', 'total_tokens')
    if not isinstance(value, dict) or any(type(value.get(k)) is not int or value[k] < 0 for k in names):
        raise ValueError('Missing or invalid token usage; budget accounting cannot continue')
    if value['prompt_tokens'] + value['completion_tokens'] != value['total_tokens']:
        raise ValueError('Inconsistent total token usage')
    return {k: value[k] for k in names}


class Client:
    def __init__(self, base, model, key, timeout):
        self.base, self.model, self.key, self.timeout = base.rstrip('/'), model, key, timeout

    def request(self, url, payload=None):
        req = urllib.request.Request(url, data=None if payload is None else json.dumps(payload).encode(),
            headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + self.key})
        with urllib.request.urlopen(req, timeout=self.timeout) as response:
            return json.load(response)

    def check_model(self):
        names = [x['id'] for x in self.request(self.base + '/models')['data']]
        if self.model not in names:
            raise ValueError('Requested model is not advertised by the service')

    def count_prompt(self, prompt):
        root = self.base[:-3] if self.base.endswith('/v1') else self.base
        result = self.request(root + '/tokenize', {'model': self.model, 'messages': prompt,
            'add_generation_prompt': True, 'chat_template_kwargs': {'enable_thinking': False}})
        count = result.get('count')
        if type(count) is not int or count < 1:
            raise ValueError('Tokenizer endpoint did not return a positive count')
        return count

    def generate(self, prompt, maximum, seed):
        return self.request(self.base + '/chat/completions', {'model': self.model, 'messages': prompt,
            'temperature': 0.7, 'max_tokens': maximum, 'seed': seed,
            'chat_template_kwargs': {'enable_thinking': False}})


def sample_problem(row, benchmark, budget, args, client):
    prompt = messages(row, benchmark)
    prompt_tokens = client.count_prompt(prompt)
    candidates, spent = [], 0
    stop = 'sample_limit'
    for index in range(args.max_samples):
        maximum = generation_limit(budget, spent, prompt_tokens, args.max_output_tokens, args.context_length)
        if maximum == 0:
            stop = 'budget_or_context_exhausted'
            break
        seed = args.seed + index
        try:
            response = client.generate(prompt, maximum, seed)
            choice = response['choices'][0]
            raw = choice['message'].get('content') or ''
        except Exception as exc:
            return {'id': str(row['id']), 'status': 'failed', 'error_type': type(exc).__name__,
                    'budget': budget, 'accounted_tokens': spent, 'usage_complete': False,
                    'failed_sample_index': index, 'candidates': candidates,
                    'vote': select_vote(candidates)}
        candidate = {'index': index, 'seed': seed, 'raw_response': raw, 'answer': boxed_answer(raw),
                     'finish_reason': choice.get('finish_reason'), 'model': response.get('model'),
                     'max_tokens': maximum, 'tokenizer_prompt_tokens': prompt_tokens,
                     'usage': response.get('usage')}
        candidates.append(candidate)
        if benchmark == 'aqua' and candidate['answer'] is not None:
            candidate['answer'] = candidate['answer'].strip().upper()
        try:
            usage = valid_usage(response.get('usage'))
            if usage['prompt_tokens'] != prompt_tokens:
                raise ValueError('Tokenizer count differs from generation prompt usage')
            if usage['completion_tokens'] > maximum:
                raise ValueError('Service exceeded requested completion limit')
            spent += usage['total_tokens']
            if spent > budget:
                raise ValueError('Service usage exceeded the total-token cap')
        except ValueError as exc:
            return {'id': str(row['id']), 'status': 'accounting_failed', 'error': str(exc),
                    'budget': budget, 'accounted_tokens': spent, 'usage_complete': False,
                    'candidates': candidates, 'vote': select_vote(candidates)}
    return {'id': str(row['id']), 'status': 'complete', 'budget': budget, 'total_tokens': spent,
            'unused_budget': budget - spent, 'stop_reason': stop, 'candidates': candidates,
            'vote': select_vote(candidates)}


def budgets_for(ids, args):
    if args.budget_file:
        source = read_json(args.budget_file)
        if source.get('scope') != 'total_prompt_plus_completion' or not source.get('source_run'):
            raise ValueError('Budget file requires scope=total_prompt_plus_completion and source_run')
        raw = source['budgets']
        if any(identity not in raw for identity in ids):
            raise ValueError('Budget file omits selected problem IDs')
        values = {identity: raw[identity] for identity in ids}
    else:
        values = {identity: args.total_token_budget for identity in ids}
    if any(type(value) is not int or value < 1 for value in values.values()):
        raise ValueError('Each total-token budget must be a positive integer')
    return values


def run(args):
    dataset = args.dataset or ROOT / 'paper/assets/datasets' / DATASETS[args.benchmark]
    rows = [json.loads(line) for line in dataset.read_text(encoding='utf-8').splitlines() if line.strip()]
    indexed = {str(row['id']): row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError('Dataset has duplicate IDs')
    ids = args.ids.split(',') if args.ids else list(indexed)
    if len(set(ids)) != len(ids) or any(i not in indexed for i in ids):
        raise ValueError('Use unique existing IDs')
    budgets = budgets_for(ids, args)
    plan = {'benchmark': args.benchmark, 'dataset': str(dataset), 'sample_ids': ids,
            'full_benchmark_total': len(rows), 'subset': len(ids) != len(rows), 'budgets': budgets,
            'budget_scope': 'all_generation_prompt_plus_completion_tokens',
            'budget_source': str(args.budget_file) if args.budget_file else 'explicit_uniform_cap',
            'method': 'single_agent_normalized_exact_answer_vote', 'system_prompt': SYSTEM,
            'temperature': 0.7, 'seed': args.seed, 'max_samples': args.max_samples,
            'chat_template_kwargs': {'enable_thinking': False},
            'max_output_tokens': args.max_output_tokens, 'context_length': args.context_length,
            'tie_break': 'earliest_sample', 'tokenizer_required': True}
    if not args.execute:
        print(json.dumps(plan, indent=2)); return 0
    if args.output_dir.exists():
        raise FileExistsError('Use a fresh output directory')
    base, model = os.environ['REASONING_URL'], os.environ['REASONING_MODEL']
    client = Client(base, model, os.environ.get('REASONING_KEY', 'EMPTY'), args.timeout)
    args.output_dir.mkdir(parents=True)
    plan.update(model=model, started_at=datetime.now(timezone.utc).isoformat())
    write_json(args.output_dir / 'manifest.json', plan)
    try:
        client.check_model()
    except Exception as exc:
        receipt = {'expected': len(ids), 'recorded': 0, 'completed': 0, 'complete': False,
                   'status': 'preflight_failed', 'error_type': type(exc).__name__, 'missing_ids': ids}
        write_json(args.output_dir / 'receipt.json', receipt)
        print(json.dumps(receipt)); return 1
    results = {}
    for identity in ids:
        start = time.monotonic()
        try:
            result = sample_problem(indexed[identity], args.benchmark, budgets[identity], args, client)
        except Exception as exc:
            result = {'id': identity, 'status': 'failed', 'error_type': type(exc).__name__}
        result['elapsed_seconds'] = time.monotonic() - start
        results[identity] = result
        write_json(args.output_dir / 'result.json', results)
        if result['status'] != 'complete':
            break
    receipt = {'expected': len(ids), 'recorded': len(results),
               'completed': sum(r['status'] == 'complete' for r in results.values()),
               'missing_ids': [i for i in ids if i not in results]}
    receipt['complete'] = receipt['completed'] == receipt['expected']
    write_json(args.output_dir / 'receipt.json', receipt)
    print(json.dumps(receipt)); return 0 if receipt['complete'] else 1


def summarize(args):
    if args.output.exists():
        raise FileExistsError('Use a fresh summary output path')
    manifest = read_json(args.run_dir / 'manifest.json')
    results = read_json(args.run_dir / 'result.json')
    rows = {str(x['id']): x for x in (json.loads(line) for line in
            Path(manifest['dataset']).read_text(encoding='utf-8').splitlines() if line.strip())}
    spec = importlib.util.spec_from_file_location('sampling_math_grader',
              ROOT / 'paper/runtime/masrubric/agents/math_grader.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    per_problem, correct, unknown = [], 0, 0
    for identity in manifest['sample_ids']:
        row, result = rows[identity], results.get(identity, {})
        if result.get('status') != 'complete':
            unknown += 1; per_problem.append({'id': identity, 'status': 'unknown'}); continue
        def grade(candidate):
            answer = candidate.get('answer')
            if answer is None:
                return False
            if manifest['benchmark'] == 'aqua':
                return answer.strip().upper() == row['correct'].strip().upper()
            return bool(module.MathGrader.check_correctness(candidate['raw_response'], row.get('solution', row['answer'])))
        labels = [grade(c) for c in result['candidates']]
        index = result['vote']['index']
        verdict = bool(index is not None and labels[index])
        correct += verdict
        n, c = len(labels), sum(labels)
        pass_k = {str(k): 1 - (math.comb(n-c, k) / math.comb(n, k) if n-c >= k else 0)
                  for k in args.k if n >= k}
        per_problem.append({'id': identity, 'vote_correct': verdict, 'generated': n, 'correct_samples': c,
                            'total_tokens': result['total_tokens'], 'generated_set_subset_success': pass_k,
                            'subset_success_unavailable': [k for k in args.k if n < k]})
    total = len(manifest['sample_ids'])
    report = {'benchmark': manifest['benchmark'], 'subset': manifest['subset'], 'expected': total,
              'vote_correct': correct, 'unknown': unknown, 'accuracy_fixed_denominator': correct/total,
              'per_problem': per_problem, 'generated_set_subset_success_at_k': {},
              'subset_success_interpretation': 'Probability that a uniformly selected k-subset of the generated candidates contains a correct answer. Adaptive budget stopping and a truncated final candidate do not provide an iid pass@k estimate.'}
    for k in args.k:
        values = [r.get('generated_set_subset_success', {}).get(str(k)) for r in per_problem]
        report['generated_set_subset_success_at_k'][str(k)] = {'eligible': sum(v is not None for v in values), 'expected': total,
            'mean': sum(values)/total if all(v is not None for v in values) else None}
    write_json(args.output, report)
    print(json.dumps({k:v for k,v in report.items() if k!='per_problem'})); return 0


def positive(value):
    value = int(value)
    if value < 1: raise argparse.ArgumentTypeError('Must be positive')
    return value


def main():
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='command',required=True)
    r=sub.add_parser('run'); r.add_argument('--benchmark',choices=DATASETS,required=True)
    r.add_argument('--dataset',type=Path); r.add_argument('--ids',help='Comma-separated exact IDs; omitted means full dataset')
    group=r.add_mutually_exclusive_group(required=True)
    group.add_argument('--total-token-budget',type=positive); group.add_argument('--budget-file',type=Path)
    r.add_argument('--max-samples',type=positive,default=16); r.add_argument('--max-output-tokens',type=positive,default=4096)
    r.add_argument('--context-length',type=positive,default=32768); r.add_argument('--seed',type=int,default=20261106)
    r.add_argument('--timeout',type=positive,default=600); r.add_argument('--output-dir',type=Path,required=True)
    r.add_argument('--execute',action='store_true')
    s=sub.add_parser('summarize');s.add_argument('--run-dir',type=Path,required=True)
    s.add_argument('--output',type=Path,required=True);s.add_argument('--k',type=positive,nargs='+',default=[1,2,3])
    a=p.parse_args();return run(a) if a.command=='run' else summarize(a)


if __name__=='__main__':
    try: raise SystemExit(main())
    except (OSError,ValueError,KeyError) as exc:
        print(type(exc).__name__ + ': ' + str(exc), file=sys.stderr);raise SystemExit(2)
