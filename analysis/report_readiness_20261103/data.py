"""Load and validate saved inference records for offline analysis."""
from collections import Counter
from pathlib import Path
from zipfile import ZipFile
import hashlib
import json
import csv
import numpy as np
from engine import question_groups

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / 'records' / 'six-model-analysis-20261024-145231.zip'
DIGEST = '70ea712be64e8625bb0638635e09036c091e9122ff56cf98e350dbe69749bfe4'
SIGNALS = {'cheap_consistency':'consistency', 'cheap_self_conf':'self_confidence', 'cheap_logprob_conf':'token_confidence'}
RATE = 3.29


class Evidence:
    def __init__(self):
        self.hash = hashlib.sha256(ARCHIVE.read_bytes()).hexdigest()
        assert self.hash == DIGEST, 'Archive changed'
        z = ZipFile(ARCHIVE)
        read = lambda n: json.loads(z.read(n).decode('utf-8-sig'))
        combined = 'full-run-six-models-qwen27-gqa'
        self.manifest = read(combined+'/experiment_manifest.json')
        self.split = read(combined+'/analysis/data_split.json')
        self.models = self.manifest['models']; self.ids = self.manifest['trace_ids']
        self.thresholds = np.array(self.manifest['threshold_grid'], float)
        self.gold = {}
        for name in z.namelist():
            if name.startswith('Who&When/') and name.endswith('.json'):
                d = read(name); source = name.split('/')[1]
                tid = source+'__'+name.rsplit('/',1)[1]
                self.gold[tid] = {'trace_id':tid, 'source':source,
                    'question_id':d.get('question_ID'), 'question':d.get('question',''),
                    'answer':d.get('ground_truth',''), 'agent':d['mistake_agent'],
                    'step':str(d['mistake_step']), 'label_reason':d.get('mistake_reason',''),
                    'messages':d.get('messages') or d.get('history',[])}
        assert len(self.gold) == len(set(self.ids)) == 184
        self.index = {tid:i for i,tid in enumerate(self.ids)}
        self.vi = np.array([self.index[t] for t in self.split['val_trace_ids']])
        self.ti = np.array([self.index[t] for t in self.split['test_trace_ids']])
        assert not set(self.vi)&set(self.ti) and len(self.vi)==110 and len(self.ti)==74
        self.groups = question_groups([self.gold[t] for t in self.ids])
        self.source = np.array([int(self.gold[t]['source']=='Hand-Crafted') for t in self.ids])
        self.arrays = {}; self.records = {}; self.raw = {}; self.checks = Counter()
        for model in self.models:
            folder = 'qwen27-bf16-gqa' if model=='qwen-27b' else 'full-run-all-5-models'
            records=[]; self.raw[model]={}
            for tid in self.ids:
                raw = [read(f'{folder}/raw/{model}/{Path(tid).stem}_run{i}.json') for i in range(3)]
                assert [(r['trace_id'],r['model_alias'],r['run_index']) for r in raw] == [(tid,model,i) for i in range(3)]
                self.checks['raw_identity']+=3
                self.raw[model][tid]=raw
                votes=[(r['predicted_agent'],str(r['predicted_step'])) if r['parse_success'] else (None,None) for r in raw]
                pred,count=Counter(votes).most_common(1)[0]
                agg=read(f'{combined}/aggregated/{model}/{tid}')
                assert (agg['majority_agent'],agg['majority_step'])==pred
                assert agg['n_runs']==3 and agg['consistency_confidence']==round(count/3,4)
                elapsed=round(sum(r['elapsed_seconds'] for r in raw),3)
                assert elapsed==agg['total_elapsed']
                if model=='gpt-5.6-sol':
                    cost=0.
                    for r in raw:
                        p,o=r['prompt_tokens'],r['completion_tokens'];cached=r.get('cached_prompt_tokens',0) or 0;written=r.get('cache_write_prompt_tokens',0) or 0
                        im,om=(2,1.5) if p>272000 else (1,1)
                        dollars=((p-cached-written+1.25*written)*4*im+cached*.4*im+o*20*om)/1e6
                        assert abs(dollars-r['usage_cost_usd'])<1e-8
                        cost+=dollars
                else:cost=elapsed*RATE/3600
                values=lambda key:[r[key] for r in raw if r.get(key) is not None]
                selfconf=values('self_reported_confidence'); tokenconf=values('logprob_confidence')
                selfscore=round(float(np.mean(selfconf))/100,4) if selfconf else np.nan
                tokenscore=round(float(np.mean(tokenconf)),6) if tokenconf else np.nan
                g=self.gold[tid]; agent=pred[0]==g['agent'];step=pred[1]==g['step']
                record={'model':model,'trace_id':tid,'source':g['source'],
                    'question_group':self.groups[self.index[tid]],'original_split':'validation' if self.index[tid] in set(self.vi) else 'test',
                    'predicted_agent':pred[0],'predicted_step':pred[1],'agent_correct':agent,'step_correct':step,
                    'y':int(agent and step),'valid':pred!=(None,None),'consistency':round(count/3,4),
                    'self_confidence':selfscore,'token_confidence':tokenscore,'cost':cost,'seconds':elapsed,
                    'tokens':sum(r['total_tokens'] for r in raw),'three_way_tie':len(set(votes))==3,
                    'invalid_responses':sum(not r['parse_success'] for r in raw),
                    'capped_responses':sum(r['completion_tokens']>=self.manifest['model_configs'][model]['max_new_tokens'] for r in raw)}
                records.append(record); self.checks['aggregates']+=1
            self.records[model]=records
            self.arrays[model]={key:np.array([r[key] for r in records]) for key in ['y','valid','agent_correct','step_correct','cost','seconds','tokens']}
            self.arrays[model]['scores']={s:np.array([r[c] for r in records],float) for s,c in SIGNALS.items()}
        self.cost_report_path=ROOT/'outputs/cost_review_20261101/sourced_cost_reanalysis.json'
        self.cost_report_hash=hashlib.sha256(self.cost_report_path.read_bytes()).hexdigest()
        self.cost_report=json.loads(self.cost_report_path.read_text(encoding='utf-8'))
        assert self.cost_report['archive_sha256']==self.hash
        self.primary=[r for r in self.cost_report['validation_reselected_policies'] if r['h100_usd_per_hour']==RATE]
        assert len(self.primary)==8
        self.pairs=[(r['pair'],r['cheap'],r['strong']) for r in self.primary]
        previous=ROOT/'reference_checks'
        with (previous/'per_trace_predictions.csv').open(encoding='utf-8-sig') as f:
            old=list(csv.DictReader(f))
        assert len(old)==1104
        lookup={(r['model'],r['trace_id']):r for r in old}
        for model in self.models:
            for r in self.records[model]:
                a=lookup[model,r['trace_id']]
                assert (r['predicted_agent'],r['predicted_step']) == (a['predicted_agent'] or None,a['predicted_step'] or None)
                assert r['y']==int(a['joint_correct']=='True')
                for new,prior in [('consistency','consistency'),('self_confidence','self_confidence'),('token_confidence','token_confidence'),('seconds','elapsed_seconds')]:
                    prior_value=float(a[prior]) if a[prior] else np.nan
                    assert np.isclose(r[new],prior_value,rtol=0,atol=1e-9,equal_nan=True)
                self.checks['prior_prediction_records']+=1
        with (previous/'test_routing.csv').open(encoding='utf-8-sig') as f:
            self.previous_routes={(r['pair'],r['trace_id']):r for r in csv.DictReader(f)}
        assert len(self.previous_routes)==592
        z.close()

    def context(self, tid):
        g=self.gold[tid];at=int(g['step']);key='role' if g['source']=='Hand-Crafted' else 'name'
        def canonical(a):
            return 'Orchestrator' if isinstance(a,str) and a.startswith('Orchestrator (') else a
        return {'trace_id':tid,'source':g['source'],'question':g['question'],
                'correct_task_answer':g['answer'],'labelled_agent':g['agent'],'labelled_step':g['step'],
                'label_reason':g['label_reason'],
                'label_step_speaker':canonical(g['messages'][at].get(key)),
                'nearby_messages':[{'step':j,'speaker':canonical(g['messages'][j].get(key)),
                                    'content':g['messages'][j].get('content','')} for j in range(max(0,at-1),min(len(g['messages']),at+2))]}
