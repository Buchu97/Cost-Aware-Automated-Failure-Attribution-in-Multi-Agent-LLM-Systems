"""Evaluate the cascades and exploratory controls using saved responses."""
from pathlib import Path
from collections import Counter,defaultdict
import csv,json,hashlib,sys
import numpy as np
from data import Evidence,ROOT,ARCHIVE,RATE,SIGNALS
from engine import (fit_iso,apply_iso,clean_raw,evaluate,select_policy,budget_probability,
                    random_expected_correct,calibration_metrics,bootstrap_control,group_folds,crossfit_iso)

OUT=ROOT/'outputs/report_readiness_20261103'
SEED=20261103;MC=50000;BOOT=10000;REPEATS=10;FOLDS=5


def plain(x):
    if isinstance(x,np.ndarray):return [plain(v) for v in x.tolist()]
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,dict):return {str(k):plain(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [plain(v) for v in x]
    if isinstance(x,float) and not np.isfinite(x):return None
    return x


def write_json(name,obj):
    (OUT/name).write_text(json.dumps(plain(obj),indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')


def write_csv(name,rows):
    if not rows:return
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,keys);w.writeheader();w.writerows([{k:plain(v) for k,v in r.items()} for r in rows])


def pct_range(x):return [float(v) for v in np.quantile(x,[.025,.975])]


def fixed_analysis(d):
    fits={};mapped={};cal=[]
    for cheap in ['llama-8b','qwen-7b']:
        a=d.arrays[cheap];fits[cheap]={};mapped[cheap]={}
        for signal,x in a['scores'].items():
            fits[cheap][signal]=fit_iso(x[d.vi],a['y'][d.vi])
            mapped[cheap][signal]=apply_iso(x,fits[cheap][signal])
            for label,ix in [('validation_fit_sample',d.vi),('test',d.ti)]:
                for method,p in [('raw',x),('isotonic',mapped[cheap][signal]),('constant_validation_rate',np.full(len(x),a['y'][d.vi].mean()))]:
                    metric=calibration_metrics(a['y'][ix],p[ix])
                    cal.append({'model':cheap,'signal':signal,'split':label,'method':method,**{k:v for k,v in metric.items() if k!='bins'}})
    primary=[];ablations=[];random=[];intervals=[];disagreements=[];routes=[]
    for pairno,(pair,cheap,strong) in enumerate(d.pairs):
        a,b=d.arrays[cheap],d.arrays[strong]
        variants={'isotonic_all_signals':mapped[cheap],
                  'isotonic_token_only':{'cheap_logprob_conf':mapped[cheap]['cheap_logprob_conf']},
                  'raw_token_only':{'cheap_logprob_conf':clean_raw(a['scores']['cheap_logprob_conf'])},
                  'raw_all_signals':{s:clean_raw(x) for s,x in a['scores'].items()}}
        for method,scores in variants.items():
            p=select_policy({s:x[d.vi] for s,x in scores.items()},a['y'][d.vi],b['y'][d.vi],a['cost'][d.vi],b['cost'][d.vi],d.thresholds)
            keep=scores[p['signal']]>=p['threshold']
            val=evaluate(a['y'][d.vi],b['y'][d.vi],a['cost'][d.vi],b['cost'][d.vi],keep[d.vi])
            test=evaluate(a['y'][d.ti],b['y'][d.ti],a['cost'][d.ti],b['cost'][d.ti],keep[d.ti])
            row={'pair':pair,'cheap':cheap,'strong':strong,'method':method,'signal':p['signal'],'threshold':p['threshold'],**{'validation_'+k:v for k,v in val.items()},**{'test_'+k:v for k,v in test.items()}}
            ablations.append(row)
            if method!='isotonic_all_signals':continue
            expected=next(r for r in d.primary if r['pair']==pair)
            assert p['signal']==expected['signal'] and p['threshold']==expected['threshold'],pair
            for key,reference in [('correct','test_cascade_correct'),('strong_correct','test_strong_correct'),('cost','test_cascade_cost_usd'),('strong_cost','test_strong_cost_usd'),('escalated','test_escalated')]:
                assert np.isclose(test[key],expected[reference],rtol=0,atol=1e-8),(pair,key,test[key],expected[reference])
            for key,reference in [('correct','validation_cascade_correct'),('strong_correct','validation_strong_correct'),('cost','validation_cascade_cost_usd'),('strong_cost','validation_strong_cost_usd'),('escalated','validation_escalated')]:
                assert np.isclose(val[key],expected[reference],rtol=0,atol=1e-8),(pair,key,val[key],expected[reference])
            for i in d.ti:
                old=d.previous_routes[pair,d.ids[i]]
                assert (not bool(keep[i]))==(old['escalated']=='True')
                assert np.isclose(scores[p['signal']][i],float(old['calibrated_confidence']),rtol=0,atol=1e-9)
                assert int(a['y'][i] if keep[i] else b['y'][i])==int(old['cascade_correct'])
                d.checks['prior_test_routes']+=1
            for label,ii in [('validation',d.vi),('test',d.ti)]:
                kk=keep[ii];yc0=a['y'][ii];ys0=b['y'][ii]
                gains=int(np.sum(kk&(yc0==1)&(ys0==0)));losses=int(np.sum(kk&(yc0==0)&(ys0==1)))
                assert gains-losses==int(np.where(kk,yc0,ys0).sum()-ys0.sum())
                disagreements.append({'pair':pair,'split':label,'kept_cheap':int(kk.sum()),
                    'cheap_correct_on_kept':int(yc0[kk].sum()),'strong_correct_on_kept':int(ys0[kk].sum()),
                    'gained_correct_cases':gains,'lost_correct_cases':losses,'net_correct_difference':gains-losses})
                for i in ii:
                    record=d.records[cheap if keep[i] else strong][i]
                    routes.append({'pair':pair,'split':label,'trace_id':d.ids[i],'question_group':d.groups[i],
                        'calibrated_confidence':float(scores[p['signal']][i]),'cutoff':p['threshold'],'escalated':not bool(keep[i]),
                        'cheap_correct':int(a['y'][i]),'strong_correct':int(b['y'][i]),'cascade_correct':record['y'],
                        'predicted_agent':record['predicted_agent'],'predicted_step':record['predicted_step'],
                        'cost':float(a['cost'][i]+(0 if keep[i] else b['cost'][i]))})
            primary.append(row)
            ix=d.ti;yc,ys,cc,sc=a['y'][ix],b['y'][ix],a['cost'][ix],b['cost'][ix];kt=keep[ix]
            boot=bootstrap_control(yc,ys,cc,sc,kt,d.groups[ix],BOOT,SEED+pairno)
            ci=pct_range(boot['gap']*100)
            intervals.append({'pair':pair,'cascade_minus_strong_accuracy_pp':100*(test['correct']-test['strong_correct'])/len(ix),
                'accuracy_difference_ci_low_pp':pct_range(boot['accuracy_difference']*100)[0],
                'accuracy_difference_ci_high_pp':pct_range(boot['accuracy_difference']*100)[1],
                'saving_percent':test['saving_percent'],'saving_ci_low':pct_range(boot['saving_percent'])[0],
                'saving_ci_high':pct_range(boot['saving_percent'])[1]})
            probabilities=[('test_expected_cost_matched',budget_probability(cc,sc,kt)),
                ('validation_budget_probability_frozen',budget_probability(a['cost'][d.vi],b['cost'][d.vi],keep[d.vi]))]
            for ctrl,pstrong in probabilities:
                expected_cost=float(cc.sum()+pstrong*sc.sum());expected_correct=random_expected_correct(yc,ys,pstrong)
                draws=np.random.default_rng(SEED+100+pairno).random((MC,len(ix)))<pstrong
                sampled_correct=yc.sum()+draws@(ys-yc);sampled_cost=cc.sum()+draws@sc
                if ctrl=='test_expected_cost_matched':assert abs(expected_cost-test['cost'])<1e-9
                random.append({'pair':pair,'control':ctrl,'strong_probability':pstrong,
                    'expected_escalated_cases':pstrong*len(ix),'expected_cost':expected_cost,
                    'confidence_policy_cost':test['cost'],'expected_random_correct':expected_correct,
                    'confidence_policy_correct':test['correct'],
                    'confidence_advantage_pp':100*(test['correct']-expected_correct)/len(ix),
                    'bootstrap_advantage_ci_low_pp':ci[0] if ctrl=='test_expected_cost_matched' else None,
                    'bootstrap_advantage_ci_high_pp':ci[1] if ctrl=='test_expected_cost_matched' else None,
                    'random_assignment_correct_low':pct_range(sampled_correct)[0],
                    'random_assignment_correct_high':pct_range(sampled_correct)[1],
                    'random_assignment_cost_low':pct_range(sampled_cost)[0],
                    'random_assignment_cost_high':pct_range(sampled_cost)[1],
                    'fraction_random_assignments_matching_or_beating_observed':float(np.mean(sampled_correct>=test['correct'])),
                    'monte_carlo_draws':MC})
        print('[fixed]',pair,'primary reproduced',flush=True)
    write_csv('primary_results.csv',primary);write_csv('fixed_split_ablations.csv',ablations)
    write_csv('calibration_diagnostics.csv',cal);write_csv('random_routing_controls.csv',random)
    write_csv('primary_fixed_policy_uncertainty.csv',intervals)
    write_csv('routing_disagreements.csv',disagreements)
    write_csv('primary_routes.csv',routes)
    return primary,ablations,random,cal


def grouped_analysis(d):
    policies=[];predictions=[];members=[];inner_members=[]
    methods=['isotonic_in_sample','isotonic_crossfit','raw_confidence']
    for repeat in range(REPEATS):
        outer=group_folds(d.groups,d.source,FOLDS,SEED+repeat)
        for i,tid in enumerate(d.ids):members.append({'repeat':repeat,'trace_id':tid,'question_group':d.groups[i],'outer_fold':int(outer[i]),'source':int(d.source[i])})
        for fold in range(FOLDS):
            train=np.flatnonzero(outer!=fold);test=np.flatnonzero(outer==fold)
            assert set(d.groups[train]).isdisjoint(d.groups[test])
            inner=group_folds(d.groups[train],d.source[train],3,SEED+10000+repeat*100+fold)
            for i,j in enumerate(train):inner_members.append({'repeat':repeat,'outer_fold':fold,'trace_id':d.ids[j],'question_group':d.groups[j],'inner_fold':int(inner[i])})
            for innerfold in range(3):assert set(d.groups[train][inner==innerfold]).isdisjoint(d.groups[train][inner!=innerfold])
            scores_train={};scores_test={}
            for cheap in ['llama-8b','qwen-7b']:
                a=d.arrays[cheap];insample={};oof={};held={}
                for signal,x in a['scores'].items():
                    fit=fit_iso(x[train],a['y'][train]);insample[signal]=apply_iso(x[train],fit)
                    oof[signal]=crossfit_iso(x[train],a['y'][train],inner)
                    held[signal]=apply_iso(x[test],fit)
                scores_train[cheap]={'isotonic_in_sample':insample,'isotonic_crossfit':oof,'raw_confidence':{s:clean_raw(x[train]) for s,x in a['scores'].items()}}
                scores_test[cheap]={'isotonic_in_sample':held,'isotonic_crossfit':held,'raw_confidence':{s:clean_raw(x[test]) for s,x in a['scores'].items()}}
            for pair,cheap,strong in d.pairs:
                a,b=d.arrays[cheap],d.arrays[strong]
                for method in methods:
                    p=select_policy(scores_train[cheap][method],a['y'][train],b['y'][train],a['cost'][train],b['cost'][train],d.thresholds)
                    keep=scores_test[cheap][method][p['signal']]>=p['threshold']
                    y=np.where(keep,a['y'][test],b['y'][test]);cost=a['cost'][test]+np.where(keep,0,b['cost'][test])
                    policies.append({'repeat':repeat,'outer_fold':fold,'pair':pair,'method':method,
                        'outer_training_n':len(train),'outer_test_n':len(test),'inner_calibration_folds':3 if method=='isotonic_crossfit' else None,**p})
                    for j,i in enumerate(test):predictions.append({'repeat':repeat,'outer_fold':fold,'pair':pair,'method':method,
                        'trace_id':d.ids[i],'question_group':d.groups[i],'joint_correct':int(y[j]),'strong_correct':int(b['y'][i]),
                        'cost':float(cost[j]),'strong_cost':float(b['cost'][i]),'escalated':not bool(keep[j])})
        print('[grouped] completed repetition',repeat+1,'/',REPEATS,flush=True)
    grouped=defaultdict(list)
    for r in predictions:grouped[(r['repeat'],r['pair'],r['method'])].append(r)
    repeats=[]
    for (repeat,pair,method),rr in grouped.items():
        assert len(rr)==len({r['trace_id'] for r in rr})==184
        correct=sum(r['joint_correct'] for r in rr);strong=sum(r['strong_correct'] for r in rr)
        cost=sum(r['cost'] for r in rr);strongcost=sum(r['strong_cost'] for r in rr)
        repeats.append({'repeat':repeat,'pair':pair,'method':method,'n':184,'correct':correct,'strong_correct':strong,
            'accuracy':correct/184,'retention':correct/strong if strong else None,'meets_target':correct>=.95*strong,
            'cost':cost,'strong_cost':strongcost,'saving_percent':100*(1-cost/strongcost),
            'escalated':sum(r['escalated'] for r in rr)})
    summary=[]
    for pair,cheap,strong in d.pairs:
        for method in methods:
            rr=[r for r in repeats if r['pair']==pair and r['method']==method]
            row={'pair':pair,'method':method,'repetitions':REPEATS,'distinct_traces':184,
                 'distinct_question_groups':len(set(d.groups)),'target_met_repetitions':sum(r['meets_target'] for r in rr),
                 'target_and_saving_repetitions':sum(r['meets_target'] and r['saving_percent']>0 for r in rr)}
            for key in ['correct','accuracy','retention','saving_percent','escalated']:
                vals=np.array([r[key] for r in rr],float)
                row.update({key+'_mean':float(vals.mean()),key+'_min':float(vals.min()),key+'_max':float(vals.max())})
            summary.append(row)
    write_csv('grouped_outer_folds.csv',members);write_csv('grouped_inner_folds.csv',inner_members)
    write_csv('grouped_cv_policies.csv',policies);write_csv('grouped_cv_predictions.csv',predictions)
    write_csv('grouped_cv_repeats.csv',repeats);write_csv('grouped_cv_summary.csv',summary)
    assert len(predictions)==REPEATS*len(d.pairs)*len(methods)*184
    return summary


def errors(d,mapped):
    counts=[];cases=[];validity=[];selected=set()
    for model in d.models:
        rr=[d.records[model][i] for i in d.ti]
        def cat(r):
            if not r['valid']:return 'invalid_final_attribution'
            if r['agent_correct'] and r['step_correct']:return 'both_correct'
            if r['agent_correct']:return 'right_agent_wrong_step'
            if r['step_correct']:return 'wrong_agent_right_step'
            return 'both_wrong'
        counter=Counter(cat(r) for r in rr)
        for category in ['both_correct','right_agent_wrong_step','wrong_agent_right_step','both_wrong','invalid_final_attribution']:
            counts.append({'model':model,'category':category,'count':counter[category],'n':74})
        for category in ['right_agent_wrong_step','wrong_agent_right_step','both_wrong']:
            candidates=[r for r in rr if cat(r)==category]
            if candidates:selected.add(candidates[0]['trace_id'])
        raw=[r for tid in d.ids for r in d.raw[model][tid]]
        cap=d.manifest['model_configs'][model]['max_new_tokens']
        capped=[r for r in raw if r['completion_tokens']>=cap]
        validity.append({'model':model,'responses':len(raw),'invalid_responses':sum(not r['parse_success'] for r in raw),
            'three_way_ties_all':sum(r['three_way_tie'] for r in d.records[model]),
            'three_way_ties_test':sum(r['three_way_tie'] for r in rr),'output_limit':cap,'responses_at_limit':len(capped),
            'at_limit_valid_attribution':sum(r['parse_success'] for r in capped),
            'stop_reason_recorded':any('finish_reason' in r or 'stop_reason' in r for r in raw)})
        selected.update(r['trace_id'] for r in capped if not r['parse_success'])
    mismatches=[]
    for tid in d.ids:
        context=d.context(tid)
        if context['label_step_speaker']!=context['labelled_agent']:
            mismatches.append(context);selected.add(tid)
    for i in d.ti:
        if np.isclose(mapped['llama-8b']['cheap_logprob_conf'][i],1.):selected.add(d.ids[i])
    for tid in sorted(selected):
        case=d.context(tid);case['original_split']='test' if d.index[tid] in set(d.ti) else 'validation'
        case['primary_calibrated_token_scores']={m:float(mapped[m]['cheap_logprob_conf'][d.index[tid]]) for m in ['llama-8b','qwen-7b']}
        case['predictions']=[{'model':m,**{k:d.records[m][d.index[tid]][k] for k in ['predicted_agent','predicted_step','agent_correct','step_correct','valid','three_way_tie','capped_responses']},
            'selected_response_text':d.raw[m][tid][next((j for j,r in enumerate(d.raw[m][tid]) if ((r['predicted_agent'],str(r['predicted_step'])) if r['parse_success'] else (None,None))==(d.records[m][d.index[tid]]['predicted_agent'],d.records[m][d.index[tid]]['predicted_step'])),0)]['generated_text']} for m in d.models]
        cases.append(case)
    write_csv('error_categories.csv',counts);write_csv('response_validity.csv',validity)
    write_json('annotation_speaker_review.json',mismatches);write_json('case_review_evidence.json',cases)
    return counts,validity,mismatches


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    d=Evidence();print('[load]',d.checks,'question groups',len(set(d.groups)),flush=True)
    write_csv('trace_groups.csv',[{'trace_id':t,'question_group':d.groups[i],'question_id':d.gold[t]['question_id'],
        'source':d.gold[t]['source'],'original_split':'validation' if i in set(d.vi) else 'test'} for i,t in enumerate(d.ids)])
    write_csv('model_trace_records.csv',[r for m in d.models for r in d.records[m]])
    primary,ablations,random,cal=fixed_analysis(d)
    write_csv('standalone_baselines.csv',[{'model':m,'n':74,'correct':int(d.arrays[m]['y'][d.ti].sum()),
        'accuracy':float(d.arrays[m]['y'][d.ti].mean()),'cost':float(d.arrays[m]['cost'][d.ti].sum())} for m in d.models])
    if '--fixed-only' in sys.argv:
        print('[fixed-only] completed; grouped analysis not requested in this invocation');return
    cv=grouped_analysis(d)
    mapped={m:{s:apply_iso(x,fit_iso(x[d.vi],d.arrays[m]['y'][d.vi])) for s,x in d.arrays[m]['scores'].items()} for m in ['llama-8b','qwen-7b']}
    counts,validity,mismatches=errors(d,mapped)
    assert hashlib.sha256(ARCHIVE.read_bytes()).hexdigest()==d.hash
    assert hashlib.sha256(d.cost_report_path.read_bytes()).hexdigest()==d.cost_report_hash
    files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}
    write_json('analysis_manifest.json',{'analysis_date':'2026-11-03','status':'computed_pending_qualitative_review',
        'archive_sha256':d.hash,'original_cost_report_sha256':d.cost_report_hash,'protocol_sha256':hashlib.sha256((OUT/'analysis_protocol.md').read_bytes()).hexdigest(),
        'script_sha256':files,'numpy_version':np.__version__,'seed':SEED,'random_assignments_per_control':MC,'cluster_bootstrap_draws':BOOT,
        'outer_repetitions':REPEATS,'outer_folds':FOLDS,'inner_calibration_folds':3,'distinct_traces':184,'canonical_question_groups':len(set(d.groups)),
        'primary_policies_reproduced':8,'raw_records_checked':d.checks['raw_identity'],'aggregate_records_checked':d.checks['aggregates'],
        'prior_prediction_records_reproduced':d.checks['prior_prediction_records'],'prior_test_routes_reproduced':d.checks['prior_test_routes'],
        'new_inference_calls':0,'paid_api_calls':0,'primary_inputs_unchanged':True,
        'interpretation':'Exploratory follow-up. Fixed primary test preserved. Random ranges describe routing assignment variability. Repeated folds reuse the same cases; split ranges are not confidence intervals.'})
    print('[complete] numerical analyses computed; case review and narrative remain',flush=True)


if __name__=='__main__':main()
