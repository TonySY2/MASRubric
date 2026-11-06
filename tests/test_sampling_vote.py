import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

PATH = Path(__file__).resolve().parents[1] / 'scripts/rebuttal/sampling_vote.py'
SPEC = importlib.util.spec_from_file_location('sampling_vote', PATH)
vote = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(vote)


class Client:
    def __init__(self, outputs, prompt=10):
        self.outputs, self.prompt, self.calls = iter(outputs), prompt, []
    def count_prompt(self, prompt):
        return self.prompt
    def generate(self, prompt, maximum, seed):
        self.calls.append({'maximum': maximum, 'seed': seed, 'prompt': prompt})
        output = next(self.outputs)
        if isinstance(output, Exception): raise output
        return output


def response(text=r'\boxed{2}', completion=6, prompt=10, usage=True):
    obj={'choices':[{'message':{'content':text},'finish_reason':'stop'}], 'model':'fixture'}
    if usage: obj['usage']={'prompt_tokens':prompt,'completion_tokens':completion,'total_tokens':prompt+completion}
    return obj


class SamplingVoteTests(unittest.TestCase):
    def setUp(self):
        self.row={'id':'x','problem':'What is one plus one?', 'answer':'SECRET_GOLD', 'solution':'SECRET_SOLUTION'}
        self.args=SimpleNamespace(max_samples=4,max_output_tokens=8,context_length=100,seed=10)

    def test_gold_and_rationale_are_never_in_request(self):
        row=dict(self.row,question='Choose an option',options=['A) 1','B) 2'],correct='SECRET_CHOICE',rationale='SECRET_RATIONALE')
        for benchmark in ('math500','aqua'):
            self.assertNotIn('SECRET', str(vote.messages(row,benchmark)))

    def test_cap_reserves_prompt_tokens_for_every_call(self):
        client=Client([response(),response(completion=4)])
        result=vote.sample_problem(self.row,'math500',30,self.args,client)
        self.assertEqual([x['maximum'] for x in client.calls],[8,4])
        self.assertEqual(result['total_tokens'],30)
        self.assertEqual(result['stop_reason'],'budget_or_context_exhausted')

    def test_missing_usage_preserves_response_and_stops(self):
        client=Client([response(usage=False),response()])
        result=vote.sample_problem(self.row,'math500',100,self.args,client)
        self.assertEqual(result['status'],'accounting_failed')
        self.assertEqual(len(client.calls),1)
        self.assertEqual(len(result['candidates']),1)
        self.assertFalse(result['usage_complete'])

    def test_prompt_count_mismatch_stops(self):
        result=vote.sample_problem(self.row,'math500',100,self.args,Client([response(prompt=11)]))
        self.assertEqual(result['status'],'accounting_failed')
        self.assertIn('Tokenizer',result['error'])

    def test_failed_later_call_preserves_previous_usage(self):
        client=Client([response(),TimeoutError()])
        result=vote.sample_problem(self.row,'math500',100,self.args,client)
        self.assertEqual(result['status'],'failed')
        self.assertEqual(result['accounted_tokens'],16)
        self.assertEqual(len(result['candidates']),1)

    def test_service_overrun_is_failed_not_hidden(self):
        result=vote.sample_problem(self.row,'math500',15,self.args,Client([response(completion=8)]))
        self.assertEqual(result['status'],'accounting_failed')
        self.assertIn('exceeded',result['error'])

    def test_no_request_if_even_prompt_does_not_fit(self):
        client=Client([])
        result=vote.sample_problem(self.row,'math500',10,self.args,client)
        self.assertEqual(client.calls,[])
        self.assertEqual(result['vote']['index'],None)

    def test_vote_groups_simple_fractions_and_never_uses_correctness(self):
        candidates=[{'answer':r'\frac{1}{2}','is_correct':False},{'answer':'0.5','is_correct':False},
                    {'answer':'9','is_correct':True}]
        selected=vote.select_vote(candidates)
        self.assertEqual(selected['index'],0)
        self.assertEqual(selected['votes'],2)

    def test_tie_uses_earliest_parseable_and_nested_box(self):
        text=r'old \boxed{1} then \boxed{\frac{1}{2}}'
        self.assertEqual(vote.boxed_answer(text),r'\frac{1}{2}')
        self.assertEqual(vote.select_vote([{'answer':None},{'answer':'3'},{'answer':'2'}])['index'],1)

    def test_bad_budget_mapping_does_not_silently_fallback(self):
        self.assertRaises(ValueError, vote.budgets_for, ['x'], SimpleNamespace(budget_file=None,total_token_budget=0))

    def test_aqua_case_is_normalized_before_voting(self):
        row={'id':'a','question':'Choose','options':['A) 0','B) 1']}
        self.args.max_samples=2
        result=vote.sample_problem(row,'aqua',40,self.args,Client([response(r'\boxed{b}'),response(r'\boxed{B}')]))
        self.assertEqual(result['vote']['votes'],2)
        self.assertEqual(result['vote']['answer'],'B')


if __name__=='__main__':unittest.main()
