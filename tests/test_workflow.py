import copy
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, Mock
from najah.store import Store
from najah.schema import ORDINAL, BINARY, MEASURES, column, validate_prediction
from najah.metrics import evaluate, ordinal_stats, binary_stats
from najah.judges import run_batch, provider_request

GATES=json.loads(Path('config/gates.json').read_text())

def episode(i,pid=None):
    return {'episode_id':f'E{i:03}','participant_id':pid or f'P{i:03}','module':'cv_building','language':'fr',
            'sampling_weight':1,'treatment':'secret-arm','turns':[{'turn_id':'001','role':'user','text':'Bonjour'}]}

def ratings_csv(eps):
    rows=[]
    for e in eps:
        for rid in ['r1','r2']:
            row={'episode_id':e['episode_id'],'rater_id':rid,'rater_name':rid,'rater_email':'private@example.com',
                 'review_layer':'primary','annotation_status':'complete','rubric_version':'v1'}
            row.update({column(m):'3' if m in ORDINAL else 'no' if m in BINARY else 'in_progress_no_output' for m in MEASURES})
            rows.append(row)
    s=io.StringIO();w=csv.DictWriter(s,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows);return s.getvalue()

def judgment(e):
    return {m:{'value':'3' if m in ORDINAL else 'no' if m in BINARY else 'in_progress_no_output',
               'evidence_turns':['001'],'reason':'Observed record.'} for m in MEASURES}

class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.s=Store(str(Path(self.tmp.name)/'study.sqlite'))
    def tearDown(self): self.s.db.close();self.tmp.cleanup()

    def test_split_never_leaks_participant_and_cannot_reshuffle(self):
        eps=[episode(i,f'P{i//3}') for i in range(60)];self.s.import_episodes(eps)
        mapping=self.s.create_split(7)
        for pid in {e['participant_id'] for e in eps}: self.assertEqual(len({mapping[e['episode_id']] for e in eps if e['participant_id']==pid}),1)
        with self.assertRaises(ValueError):self.s.create_split(8)
        with self.assertRaises(ValueError):self.s.import_episodes([episode(999)])

    def test_conflicting_imports_rejected_and_contacts_removed(self):
        eps=[episode(1)];self.s.import_ratings(ratings_csv(eps));self.s.import_ratings(ratings_csv(eps))
        self.assertEqual(len(self.s.ratings('E001')),2)
        self.assertNotIn('rater_email',self.s.ratings('E001')[0])
        with self.assertRaises(ValueError):self.s.import_ratings(ratings_csv(eps).replace('3,3,3','2,3,3'))

    def test_reference_requires_independent_evidence_and_logs_changes(self):
        self.s.import_episodes([episode(1)]);self.s.import_ratings(ratings_csv([episode(1)]))
        self.assertIsNone(self.s.reference('E001','safety'))
        with self.assertRaises(ValueError):self.s.save_reference('E001','safety','3','r1',['001'],'Reviewed')
        with self.assertRaises(ValueError):self.s.save_reference('E001','safety','3','reviewer',[],'Reviewed')
        self.s.save_reference('E001','safety','unresolved','reviewer',[],'Insufficient evidence')
        self.assertEqual(self.s.reference('E001','safety')['value'],'unresolved')
        self.s.save_reference('E001','safety','3','reviewer',['001'],'Reviewed original')
        self.assertEqual(self.s.db.execute("SELECT count(*) FROM audit WHERE event='adjudicate'").fetchone()[0],2)

    def test_full_mocked_workflow_and_frozen_test_immutability(self):
        eps=[episode(i) for i in range(12)];self.s.import_episodes(eps);self.s.import_ratings(ratings_csv(eps));mapping=self.s.create_split()
        for e in eps:
            for m in MEASURES:
                self.s.save_reference(e['episode_id'],m,judgment(e)[m]['value'],'expert',['001'],'Reviewed original')
        a=self.s.add_candidate('openai','pinned-openai-model','rubric',{'max_output_tokens':1000})
        b=self.s.add_candidate('anthropic','pinned-anthropic-model','rubric',{'max_output_tokens':1000})
        with self.assertRaises(ValueError):run_batch(self.s,a,'test')
        with patch('najah.judges.provider_request',side_effect=lambda c,e:{'judgments':judgment(e)}):
            run_batch(self.s,a,'validation',100);run_batch(self.s,b,'validation',100)
            frozen=self.s.freeze(a,b,GATES,'Selected using validation; test not used for prompt development.')
            run_batch(self.s,a,'test',100)
            self.assertEqual(run_batch(self.s,a,'test',100),0)
        with self.assertRaises(ValueError):self.s.save_reference('E000','safety','2','expert',['001'],'Changed')
        with self.assertRaises(ValueError):self.s.add_candidate('openai','other','rubric',{})
        with self.assertRaises(ValueError):run_batch(self.s,a,'train')
        testid=next(k for k,v in mapping.items() if v=='test')
        with self.assertRaises(ValueError):self.s.save_prediction(a,testid,{})
        self.assertEqual(len(self.s.report_records(a,'test','safety')),sum(v=='test' for v in mapping.values()))
        self.assertEqual(frozen['hash'],self.s.setting('freeze')['hash'])

    def test_schema_rejects_fabricated_evidence_and_missing_measures(self):
        e=episode(1);obj=judgment(e);validate_prediction(obj,e)
        obj['safety']['evidence_turns']=['999']
        with self.assertRaises(ValueError):validate_prediction(obj,e)
        with self.assertRaises(ValueError):validate_prediction({},e)

    def test_provider_request_blinded_and_response_validated(self):
        e=episode(1);e['human_label']='secret-human-label';e['rater_id']='secret-rater'
        e['turns'][0]['text_en']='translation-for-reviewers-only'
        c={'provider':'openai','model':'snapshot','prompt':'rubric','config':{'max_output_tokens':1000}}
        resp=Mock(ok=True,status_code=200);resp.json.return_value={'status':'completed','model':'snapshot','id':'resp_test','output':[{'content':[{'type':'output_text','text':json.dumps(judgment(e))}]}]}
        with patch.dict('os.environ',{'OPENAI_API_KEY':'fake'}),patch('najah.judges.requests.post',return_value=resp) as post:
            result=provider_request(c,e);body=post.call_args.kwargs['json'];wire=json.dumps(body)
            for secret in ['secret-arm','secret-human-label','secret-rater','P001','translation-for-reviewers-only','text_en']:self.assertNotIn(secret,wire)
            self.assertFalse(body['store']);self.assertEqual(result['returned_model'],'snapshot')

    def test_anthropic_request_and_truncation(self):
        e=episode(1);c={'provider':'anthropic','model':'snapshot','prompt':'rubric','config':{}}
        resp=Mock(ok=True,status_code=200);resp.json.return_value={'stop_reason':'end_turn','model':'snapshot','content':[{'type':'text','text':json.dumps(judgment(e))}]}
        with patch.dict('os.environ',{'ANTHROPIC_API_KEY':'fake'}),patch('najah.judges.requests.post',return_value=resp) as post:
            provider_request(c,e);self.assertEqual(post.call_args.kwargs['headers']['anthropic-version'],'2023-06-01')
            resp.json.return_value['stop_reason']='max_tokens'
            with self.assertRaises(ValueError):provider_request(c,e)

class MetricTests(unittest.TestCase):
    def test_ordinal_known_values(self):
        out=ordinal_stats([1,2,3],[1,2,3],[1,1,1]);self.assertEqual(out['kappa'],1);self.assertEqual(out['mae'],0)
        self.assertIsNone(ordinal_stats([3,3],[3,3],[1,1])['kappa'])

    def test_no_positive_cases_cannot_pass(self):
        rs=[{'human':'no','prediction':'no','participant_id':str(i),'weight':1} for i in range(100)]
        r=evaluate('unsafeAdvice',rs,GATES['unsafeAdvice'],bootstrap=100)
        self.assertEqual(r['status'],'insufficient_evidence');self.assertIsNone(r['recall'])

    def test_abstentions_count_as_errors_and_missing_prediction_coverage(self):
        out=binary_stats(['yes','no'],['cannot_determine','missing'],[1,1]);self.assertEqual(out['recall'],0);self.assertEqual(out['specificity'],0)
        rs=[{'human':'yes' if i<30 else 'no','prediction':'missing','participant_id':str(i),'weight':1} for i in range(80)]
        self.assertEqual(evaluate('fabrication',rs,GATES['fabrication'],bootstrap=100)['prediction_coverage'],0)

    def test_ordinal_na_abstention_does_not_hide_failures(self):
        rs=[{'human':str(1+i%3),'prediction':'na','participant_id':str(i),'weight':1} for i in range(80)]
        out=evaluate('routing',rs,GATES['routing'],bootstrap=100);self.assertEqual(out['prediction_coverage'],0);self.assertEqual(out['status'],'insufficient_evidence')

    def test_perfect_rare_detection_not_zero_width_ci(self):
        rs=[{'human':'yes' if i<30 else 'no','prediction':'yes' if i<30 else 'no','participant_id':str(i),'weight':1} for i in range(80)]
        out=evaluate('fabrication',rs,GATES['fabrication'],bootstrap=100)
        self.assertLess(out['ci']['recall'][0],1);self.assertGreater(out['ci']['recall'][0],.8);self.assertEqual(out['status'],'pass')

    def test_missing_outcome_class_withholds_pass(self):
        rs=[{'human':'in_progress_no_output','prediction':'in_progress_no_output','participant_id':str(i),'weight':1} for i in range(80)]
        self.assertEqual(evaluate('task_outcome',rs,GATES['task_outcome'],bootstrap=100)['status'],'insufficient_evidence')

    def test_weighted_metric_not_unweighted(self):
        out=ordinal_stats([1,3],[3,3],[9,1]);self.assertAlmostEqual(out['mae'],1.8)

if __name__=='__main__': unittest.main()
