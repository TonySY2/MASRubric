import copy
import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import importlib.util

REPO_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location('rubric_reviews', REPO_ROOT / 'scripts' / 'rebuttal' / 'merge_rubric_reviews.py')
m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m)

ROOT = REPO_ROOT / 'docs' / 'rebuttal' / 'annotation'
KEY = json.loads((ROOT / 'rubric_review_reference.json').read_text(encoding='utf-8'))
TEMPLATE = ROOT / 'MASRubric_准则独立盲审模板.xlsx'


def completed(name):
    return {'source':'synthetic fixture','reviewer':name,'date':'2026-11-06','declared_complete':True,
            'complete':True,'completion_issues':[],'blank_labels':[],'missing_reasons':[],
            'rows':{r['id']:{'labels':['是']*4,'reason':''} for r in KEY['rows']}}


class ReviewTests(unittest.TestCase):
    def test_delivered_template_is_blank_and_exact(self):
        review=m.parse_review(TEMPLATE,KEY)
        self.assertFalse(review['complete'])
        self.assertEqual(len(review['blank_labels']),800)
        self.assertEqual(len(review['rows']),200)
        self.assertFalse(any(r['reason'] for r in review['rows'].values()))

    def test_explicit_completion_required(self):
        a=completed('A'); b=completed('B'); b['complete']=False; b['declared_complete']=False
        report,_=m.summarize(a,b,KEY)
        self.assertFalse(report['status']['finalized'])
        self.assertIsNone(report['domains']['数学']['overall_reliability']['rate'])
        self.assertIsNone(report['domains']['代码']['dimensions']['语义正确']['agreement']['rate'])

    def test_unknown_is_not_no_and_is_excluded_from_binary_denominator(self):
        a=completed('A'); b=completed('B'); item=KEY['rows'][0]; i=item['id']; domain=item['domain']
        a['rows'][i]['labels'][0]='不确定'; b['rows'][i]['labels'][0]='不确定'
        a['rows'][i]['reason']=b['rows'][i]['reason']='synthetic uncertainty'
        report,_=m.summarize(a,b,KEY)
        self.assertTrue(report['status']['finalized'])
        dim=report['domains'][domain]['dimensions']['语义正确']
        self.assertEqual(dim['final']['uncertain'],1)
        self.assertEqual(dim['final']['no'],0)
        self.assertEqual(dim['final']['determinate_denominator'],99)
        self.assertEqual(dim['agreement']['rate'],1)
        self.assertEqual(report['domains'][domain]['overall_reliability']['all_four_determinate_denominator'],99)

    def test_dispute_needs_explicit_adjudication(self):
        a=completed('A'); b=completed('B'); i=KEY['rows'][0]['id']; b['rows'][i]['labels'][0]='否'
        b['rows'][i]['reason']='synthetic negative'
        report,disputes=m.summarize(a,b,KEY)
        self.assertFalse(report['status']['finalized'])
        self.assertEqual(report['status']['disagreements'],1)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'adjudication.csv'; row=disputes[0]
            row.update({'裁决标签':'不确定','裁决理由':'synthetic adjudication','裁决者':'C','裁决日期':'2026-11-06','裁决完成':'是'})
            p.write_text(m.csv_text([row],list(row)),encoding='utf-8-sig')
            report,_=m.summarize(a,b,KEY,p)
            self.assertTrue(report['status']['finalized'])
            self.assertEqual(report['final_labels'][i][0],'不确定')
            self.assertEqual(report['adjudication_records'][0]['裁决者'],'C')
            row['裁决完成']=''
            p.write_text(m.csv_text([row],list(row)),encoding='utf-8-sig')
            with self.assertRaisesRegex(ValueError,'不完整'):m.summarize(a,b,KEY,p)

    def test_same_person_cannot_form_independent_pair(self):
        report,_=m.summarize(completed('Same'),completed('same'),KEY)
        self.assertFalse(report['status']['both_reviews_complete'])

    def test_duplicate_missing_and_modified_source_rejected(self):
        original=m.read_xlsx(TEMPLATE)
        for operation in ['duplicate','missing','source']:
            data=copy.deepcopy(original); sheet=data['盲审']
            if operation=='duplicate': sheet['A3']=sheet['A2']
            elif operation=='missing':
                for c in 'ABCDEFGHIJK':sheet.pop(c+'2',None)
            else: sheet['D2']='modified source'
            with self.subTest(operation=operation),patch.object(m,'read_xlsx',return_value=data),self.assertRaises(ValueError):
                m.parse_review('fixture',KEY)

    def test_yes_confirmation_does_not_override_blank_labels(self):
        data=m.read_xlsx(TEMPLATE)
        data['协议'].update({'B3':'A','B4':'2026-11-06','B5':'是'})
        with patch.object(m,'read_xlsx',return_value=data):review=m.parse_review('fixture',KEY)
        self.assertTrue(review['declared_complete'])
        self.assertFalse(review['complete'])


if __name__=='__main__':unittest.main()
