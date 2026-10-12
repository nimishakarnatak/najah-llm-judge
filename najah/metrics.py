"""Weighted metrics and participant-cluster percentile bootstrap; no row-level pseudoreplication."""
from collections import Counter
import numpy as np
from .schema import ORDINAL, BINARY, UNKNOWN, OUTCOMES

def ordinal_stats(y, p, w):
    y, p, w = np.asarray(y, float), np.asarray(p, float), np.asarray(w, float)
    px = np.array([w[y == i].sum() / w.sum() for i in [1, 2, 3]])
    pp = np.array([w[p == i].sum() / w.sum() for i in [1, 2, 3]])
    expected = sum(px[i] * pp[j] * (i-j)**2 for i in range(3) for j in range(3))
    return {'kappa': float(1 - np.average((y-p)**2, weights=w) / expected) if expected > 0 else None,
            'mae': float(np.average(abs(y-p), weights=w)),
            'agreement': float(np.average(y == p, weights=w))}

def binary_stats(y, p, w):
    y, p, w = np.asarray(y), np.asarray(p), np.asarray(w, float)
    tp = w[(y == 'yes') & (p == 'yes')].sum()
    fn = w[(y == 'yes') & (p != 'yes')].sum()
    fp = w[(y == 'no') & (p == 'yes')].sum()
    tn = w[(y == 'no') & (p == 'no')].sum()
    # Abstention on a known negative is not a correct negative.
    neg = w[y == 'no'].sum()
    precision = float(tp/(tp+fp)) if tp+fp else None
    recall = float(tp/(tp+fn)) if tp+fn else None
    return {'precision': precision, 'recall': recall,
            'specificity': float(tn/neg) if neg else None,
            'f1': float(2*precision*recall/(precision+recall)) if precision is not None and recall is not None and precision+recall else 0.0}

def nominal_stats(y, p, w):
    # Include every prespecified substantive class, even when predictions never use it.
    y, p, w = np.asarray(y), np.asarray(p), np.asarray(w, float)
    recalls, f1s = [], []
    for c in OUTCOMES[:-1]:
        tp = w[(y == c) & (p == c)].sum(); actual = w[y == c].sum(); pred = w[p == c].sum()
        recalls.append(float(tp/actual) if actual else 0.0)
        f1s.append(float(2*tp/(actual+pred)) if actual+pred else 0.0)
    return {'macro_f1': float(np.mean(f1s)), 'minimum_recall': min(recalls),
            'agreement': float(np.average(y == p, weights=w))}

def evaluate(measure, records, gate, bootstrap=1000, seed=20261011):
    """records: human, prediction (missing becomes abstain), participant_id, weight.
    Ordinal numeric metrics exclude NA/abstention, but coverage and applicability gates retain them.
    """
    known = [r for r in records if r['human'] not in UNKNOWN]
    numeric = [r for r in known if r['human'] in ['1', '2', '3']]
    if measure in ORDINAL:
        paired = [r for r in numeric if r['prediction'] in ['1', '2', '3']]
        fn = ordinal_stats
        class_counts = Counter(r['human'] for r in numeric)
        enough_classes = sum(class_counts[c] for c in ['1', '2']) >= gate['min_low_scores']
    else:
        paired = known
        fn = binary_stats if measure in BINARY else nominal_stats
        class_counts = Counter(r['human'] for r in known)
        enough_classes = (class_counts['yes'] >= gate['min_positive'] and class_counts['no'] >= gate['min_negative']) if measure in BINARY else all(class_counts[c] >= gate['min_per_class'] for c in OUTCOMES[:-1])
    known_weight = sum(r['weight'] for r in known)
    covered = [r for r in known if r['prediction'] not in UNKNOWN and r['prediction'] != 'missing']
    # Ordinal false NA is excluded from numeric coverage, not silently treated as a successful response.
    if measure in ORDINAL:
        covered = [r for r in known if (r['human'] == 'na' and r['prediction'] == 'na') or
                   (r['human'] != 'na' and r['prediction'] in ['1', '2', '3'])]
    coverage = sum(r['weight'] for r in covered)/known_weight if known_weight else 0
    result = {'measure': measure, 'n_reference': len(known), 'n_paired': len(paired),
              'n_clusters': len({r['participant_id'] for r in paired}),
              'reference_coverage': len(known)/len(records) if records else 0,
              'prediction_coverage': coverage, 'class_counts': dict(class_counts), 'ci': {}}
    confusion={}
    for r in known:
        key=f"{r['human']} -> {r['prediction']}"
        cell=confusion.setdefault(key,{'count':0,'weighted_count':0.0})
        cell['count']+=1;cell['weighted_count']+=r['weight']
    result['confusion']=confusion
    if measure in ORDINAL:
        result['two_point_errors']=sum(abs(int(r['human'])-int(r['prediction']))==2 for r in paired)
    if not paired:
        return {**result, 'status': 'insufficient_evidence', 'reasons': ['No evaluable reference cases.']}
    def stats(rs):
        out = fn([r['human'] for r in rs], [r['prediction'] for r in rs], [r['weight'] for r in rs])
        return out
    result.update(stats(paired))
    groups = {}
    for r in paired: groups.setdefault(r['participant_id'], []).append(r)
    rng = np.random.default_rng(seed); ids = list(groups); draws = {}
    for _ in range(bootstrap):
        rs = [r for i in rng.integers(0, len(ids), len(ids)) for r in groups[ids[i]]]
        for name, value in stats(rs).items():
            if value is not None and np.isfinite(value): draws.setdefault(name, []).append(value)
    for name, vals in draws.items():
        # Undefined resamples must not disappear from the evidence requirement.
        if len(vals) >= bootstrap * .95:
            result['ci'][name] = [float(x) for x in np.quantile(vals, [.025, .975])]
    if measure in BINARY:
        # An all-correct bootstrap is degenerate. Widen proportion CIs with a
        # conservative Wilson bound using Kish effective participant counts.
        # This is a finite-sample guard, not an exact survey-design interval.
        for name in ['precision','recall','specificity']:
            denrows=[r for r in paired if (r['prediction']=='yes' if name=='precision' else r['human']=='yes' if name=='recall' else r['human']=='no')]
            totals={}
            for r in denrows: totals[r['participant_id']]=totals.get(r['participant_id'],0)+r['weight']
            z=1.96;value=result.get(name)
            if totals and value is not None and name in result['ci']:
                ww=np.array(list(totals.values()));neff=float(ww.sum()**2/(ww**2).sum())
                center=(value+z*z/(2*neff))/(1+z*z/neff)
                half=z*np.sqrt(value*(1-value)/neff+z*z/(4*neff**2))/(1+z*z/neff)
                result['ci'][name]=[min(result['ci'][name][0],float(center-half)),max(result['ci'][name][1],float(center+half))]
                result[f'{name}_effective_clusters']=neff
    if measure in ORDINAL:
        # Applicable vs NA balanced accuracy, only when both human classes exist.
        cats = [('na' if r['human'] == 'na' else 'yes',
                 'na' if r['prediction'] == 'na' else 'yes' if r['prediction'] in ['1','2','3'] else 'missing', r['weight']) for r in known]
        rates=[]
        for c in ['yes','na']:
            den=sum(w for y,p,w in cats if y==c)
            if den: rates.append(sum(w for y,p,w in cats if y==c and p==c)/den)
        result['applicability_balanced_accuracy'] = float(np.mean(rates)) if len(rates)==2 else None
        result['n_na'] = sum(r['human']=='na' for r in known)
    reasons=[]
    insufficient = len(paired)<gate['min_n'] or result['n_clusters']<gate['min_clusters'] or not enough_classes
    if insufficient: reasons.append('Too few cases, participant clusters, or required reference classes.')
    if result['reference_coverage']<gate['reference_coverage']: reasons.append('Reference coverage below gate.')
    if coverage<gate['prediction_coverage']: reasons.append('Prediction coverage below gate; missing/abstaining judgments count against coverage.')
    for name, bounds in gate['metrics'].items():
        value=result.get(name); ci=result['ci'].get(name)
        if value is None or ci is None:
            insufficient=True; reasons.append(f'{name}: statistic or stable confidence interval unavailable.'); continue
        if 'min' in bounds and (value<bounds['min'] or ci[0]<bounds['lower']): reasons.append(f'{name} below point-estimate or lower-confidence-bound gate.')
        if 'max' in bounds and (value>bounds['max'] or ci[1]>bounds['upper']): reasons.append(f'{name} above point-estimate or upper-confidence-bound gate.')
    if measure in ORDINAL:
        if result['n_na']<gate['min_na']:
            insufficient=True; reasons.append('Insufficient NA cases to validate applicability decisions.')
        elif result['applicability_balanced_accuracy'] is None or result['applicability_balanced_accuracy']<gate['applicability_accuracy']:
            reasons.append('Applicability balanced accuracy below gate.')
    result['status']='insufficient_evidence' if insufficient else 'fail' if reasons else 'pass'
    result['reasons']=reasons
    return result
