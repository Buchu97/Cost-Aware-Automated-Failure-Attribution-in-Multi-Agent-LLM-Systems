"""Pure offline numerical routines; no inference, configuration or network imports."""
import numpy as np


def fit_iso(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    good = np.isfinite(x) & np.isfinite(y)
    x, y = x[good], y[good]
    if not len(x):
        raise ValueError('No finite calibration examples')
    unique, inverse = np.unique(x, return_inverse=True)
    totals = np.bincount(inverse, weights=y)
    counts = np.bincount(inverse)
    blocks = []
    for i, (total, count) in enumerate(zip(totals, counts)):
        blocks.append([i, i, float(total), int(count)])
        while len(blocks) > 1 and blocks[-2][2]/blocks[-2][3] > blocks[-1][2]/blocks[-1][3]:
            b, a = blocks.pop(), blocks.pop()
            blocks.append([a[0], b[1], a[2]+b[2], a[3]+b[3]])
    fitted = np.empty(len(unique))
    for lo, hi, total, count in blocks:
        fitted[lo:hi+1] = total/count
    return {'x': unique, 'y': fitted}


def apply_iso(x, fitted):
    x = np.asarray(x, float)
    out = np.interp(x, fitted['x'], fitted['y'])
    out[~np.isfinite(x)] = -1.
    return out


def clean_raw(x):
    return np.where(np.isfinite(x), x, -1.)


def evaluate(ycheap, ystrong, ccheap, cstrong, keep):
    keep = np.asarray(keep, bool)
    y = np.where(keep, ycheap, ystrong)
    sc = int(np.sum(ystrong)); cc = int(y.sum())
    strong_cost = float(np.sum(cstrong))
    cost = float(np.sum(ccheap) + np.sum(np.asarray(cstrong)[~keep]))
    return {'n': len(y), 'correct': cc, 'accuracy': float(y.mean()),
            'strong_correct': sc, 'retention': cc/sc if sc else None,
            'meets_target': bool(cc >= .95*sc), 'cost': cost,
            'strong_cost': strong_cost,
            'saving_percent': 100*(1-cost/strong_cost) if strong_cost else None,
            'escalated': int((~keep).sum()), 'kept': int(keep.sum())}


def select_policy(scores, ycheap, ystrong, ccheap, cstrong, thresholds):
    minimum = .95*float(np.mean(ystrong))
    winners = []
    for signal, values in scores.items():
        eligible = []
        for t in thresholds:
            keep = np.asarray(values) >= t
            correct = int(np.where(keep, ycheap, ystrong).sum())
            accuracy = correct/len(keep)
            if accuracy >= minimum:
                cost = float(np.sum(ccheap)+np.sum(np.asarray(cstrong)[~keep]))
                eligible.append((round(cost, 6), -accuracy, float(t), cost, correct))
        if not eligible:
            raise ValueError('No eligible rule; check all-strong fallback')
        best = min(eligible)
        winners.append({'signal': signal, 'threshold': best[2], 'cost': best[3],
                        'correct': best[4], 'selection_n': len(ycheap),
                        'selection_accuracy': -best[1],
                        'selection_strong_correct': int(np.sum(ystrong)),
                        'minimum_required_correct': int(np.ceil(.95*np.sum(ystrong)-1e-12))})
    # Preserve original signal order when rounded minimum costs tie.
    return min(winners, key=lambda r: round(r['cost'], 6))


def budget_probability(ccheap, cstrong, keep):
    denominator = float(np.sum(cstrong))
    selected = float(np.sum(np.asarray(cstrong)[~np.asarray(keep, bool)]))
    return selected/denominator if denominator else 0.


def random_expected_correct(ycheap, ystrong, probability):
    return float(np.sum((1-probability)*np.asarray(ycheap)+probability*np.asarray(ystrong)))


def question_groups(records):
    parent = list(range(len(records)))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    seen = {}
    for i, row in enumerate(records):
        qid = str(row.get('question_id') or '').strip()
        wording = ' '.join(str(row.get('question') or '').casefold().split())
        for key in [('id', qid), ('text', wording)]:
            if not key[1]:
                continue
            if key in seen:
                a, b = root(i), root(seen[key])
                parent[max(a,b)] = min(a,b)
            else:
                seen[key] = i
    roots = [root(i) for i in range(len(records))]
    labels = {r: f'q{j:03d}' for j,r in enumerate(sorted(set(roots)))}
    return np.array([labels[r] for r in roots])


def group_folds(groups, source, nfolds, seed):
    groups, source = np.asarray(groups), np.asarray(source)
    unique = np.unique(groups)
    if len(unique) < nfolds:
        raise ValueError('Fewer question groups than requested folds')
    strata = np.unique(source); rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique)
    vectors = {g: np.array([np.sum((groups==g)&(source==s)) for s in strata], float) for g in unique}
    order = sorted(shuffled, key=lambda g: -float(vectors[g].sum()))
    totals = np.array([np.sum(source==s) for s in strata], float)
    target = totals/nfolds
    allocated = np.zeros((nfolds,len(strata))); assignments = {}
    for g in order:
        empty = np.flatnonzero(allocated.sum(axis=1)==0)
        candidates = empty if len(empty) else np.arange(nfolds)
        losses = []
        for f in candidates:
            trial = allocated.copy(); trial[f] += vectors[g]
            losses.append(float(np.sum((trial-target)**2/np.maximum(target,1.))))
        best = np.flatnonzero(np.isclose(losses, min(losses), rtol=0, atol=1e-10))
        f = int(candidates[int(rng.choice(best))])
        allocated[f] += vectors[g]; assignments[g] = f
    return np.array([assignments[g] for g in groups], int)


def crossfit_iso(x, y, folds):
    out = np.full(len(x), np.nan)
    for f in np.unique(folds):
        held = folds==f
        out[held] = apply_iso(np.asarray(x)[held], fit_iso(np.asarray(x)[~held], np.asarray(y)[~held]))
    return out


def cluster_bootstrap_mean(values, groups, draws, seed):
    values, groups = np.asarray(values, float), np.asarray(groups)
    unique, inverse = np.unique(groups, return_inverse=True)
    totals = np.bincount(inverse, weights=values)
    counts = np.bincount(inverse)
    choices = np.random.default_rng(seed).integers(0,len(unique),size=(draws,len(unique)))
    return totals[choices].sum(axis=1)/counts[choices].sum(axis=1)


def bootstrap_control(ycheap,ystrong,ccheap,cstrong,keep,groups,draws,seed):
    unique,inverse=np.unique(groups,return_inverse=True)
    total=lambda v:np.bincount(inverse,weights=np.asarray(v,float),minlength=len(unique))
    bins={'n':total(np.ones(len(groups))), 'cheap_y':total(ycheap),'strong_y':total(ystrong),
          'cascade_y':total(np.where(keep,ycheap,ystrong)), 'cheap_cost':total(ccheap),
          'strong_cost':total(cstrong),'selected_strong_cost':total(np.asarray(cstrong)*(~keep))}
    choices=np.random.default_rng(seed).integers(0,len(unique),size=(draws,len(unique)))
    b={k:v[choices].sum(axis=1) for k,v in bins.items()}
    p=b['selected_strong_cost']/b['strong_cost']
    expected=b['cheap_y']+p*(b['strong_y']-b['cheap_y'])
    return {'gap':(b['cascade_y']-expected)/b['n'],
            'accuracy_difference':(b['cascade_y']-b['strong_y'])/b['n'],
            'saving_percent':100*(1-(b['cheap_cost']+b['selected_strong_cost'])/b['strong_cost'])}


def calibration_metrics(y, p):
    y,p = np.asarray(y,float),np.asarray(p,float)
    good = np.isfinite(p)&(p>=0)&(p<=1)
    y,p = y[good],p[good]
    bins=[]
    for i in range(5):
        mask=(p>=i/5)&((p<(i+1)/5) if i<4 else (p<=1))
        if mask.any():
            bins.append({'lower':i/5,'upper':(i+1)/5,'n':int(mask.sum()),'confidence':float(p[mask].mean()),'accuracy':float(y[mask].mean())})
    positive,negative=p[y==1],p[y==0]
    auc=float(((positive[:,None]>negative).sum()+.5*(positive[:,None]==negative).sum())/(len(positive)*len(negative))) if len(positive) and len(negative) else None
    return {'n':len(y),'accuracy':float(y.mean()),'mean_confidence':float(p.mean()),
            'brier':float(np.mean((p-y)**2)),
            'ece_5bin':sum(b['n']*abs(b['confidence']-b['accuracy']) for b in bins)/len(y),
            'auc':auc,'bins':bins}
