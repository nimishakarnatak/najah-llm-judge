"""Single-study SQLite store. Freeze is irreversible in the UI; begin a new study for retesting."""
import csv
import io
import json
import math
import random
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from .schema import MEASURES, ALLOWED, values, digest

def now(): return datetime.now(timezone.utc).isoformat()

def engine_hash():
    root=Path(__file__).parent
    return digest({name:(root/name).read_text(encoding='utf-8') for name in ['schema.py','metrics.py','judges.py','store.py']})

class Store:
    def __init__(self, path='data/study.sqlite'):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, timeout=30)
        self.db.row_factory = sqlite3.Row
        self.db.executescript('''
        PRAGMA journal_mode=WAL;
        CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY,value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS episodes(id TEXT PRIMARY KEY,payload TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS ratings(id TEXT PRIMARY KEY,episode TEXT NOT NULL,payload TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS refs(episode TEXT,measure TEXT,value TEXT,reviewer TEXT,evidence TEXT,reason TEXT,source TEXT,updated TEXT,PRIMARY KEY(episode,measure));
        CREATE TABLE IF NOT EXISTS candidates(id TEXT PRIMARY KEY,payload TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS predictions(candidate TEXT,episode TEXT,payload TEXT,status TEXT,error TEXT,updated TEXT,PRIMARY KEY(candidate,episode));
        CREATE TABLE IF NOT EXISTS audit(id INTEGER PRIMARY KEY AUTOINCREMENT,time TEXT,event TEXT,payload TEXT);
        ''')
        self.db.commit()

    def log(self, event, obj):
        self.db.execute('INSERT INTO audit(time,event,payload) VALUES(?,?,?)', (now(),event,json.dumps(obj,ensure_ascii=False)))

    def setting(self, key, default=None):
        r=self.db.execute('SELECT value FROM settings WHERE key=?',(key,)).fetchone()
        return json.loads(r[0]) if r else default

    def set_setting(self,key,value):
        self.db.execute('INSERT OR REPLACE INTO settings VALUES(?,?)',(key,json.dumps(value,ensure_ascii=False)))

    def mutable(self):
        if self.setting('freeze'): raise ValueError('Protocol frozen. Data, references, splits and candidates cannot change.')

    def episodes(self):
        return [json.loads(r[0]) for r in self.db.execute('SELECT payload FROM episodes ORDER BY id')]

    def import_episodes(self, episodes):
        self.mutable()
        if self.setting('split'): raise ValueError('Import all episodes before assigning splits. Use a new database for a new cohort.')
        seen=set(); valid=[]
        for e in episodes:
            required=['episode_id','participant_id','language','module','sampling_weight','turns']
            if not all(k in e for k in required): raise ValueError(f'Episode fields required: {required}')
            eid=str(e['episode_id']).strip(); pid=str(e['participant_id']).strip()
            if not eid or not pid or eid in seen: raise ValueError('Empty IDs or duplicate episode IDs.')
            if self.db.execute('SELECT 1 FROM episodes WHERE id=?',(eid,)).fetchone(): raise ValueError(f'{eid} already imported; imports never overwrite.')
            seen.add(eid); w=float(e['sampling_weight'])
            if not math.isfinite(w) or w<=0: raise ValueError('Sampling weights must be finite and positive.')
            if not isinstance(e['turns'],list) or not e['turns']: raise ValueError(f'{eid}: original-language turns required.')
            tids=set()
            for t in e['turns']:
                if not all(k in t for k in ['turn_id','role','text']): raise ValueError('Each turn requires turn_id, role, text.')
                t['turn_id']=str(t['turn_id'])
                if t['turn_id'] in tids or not t['turn_id'] or t['role'] not in ['user','assistant','system','tool'] or not isinstance(t['text'],str): raise ValueError('Invalid or duplicate turn.')
                tids.add(t['turn_id'])
            e={**e,'episode_id':eid,'participant_id':pid,'sampling_weight':w}
            valid.append(e)
        with self.db:
            for e in valid: self.db.execute('INSERT INTO episodes VALUES(?,?)',(e['episode_id'],json.dumps(e,ensure_ascii=False)))
            self.log('import_episodes',{'count':len(valid),'hash':digest(valid)})
        return len(valid)

    def import_ratings(self, text):
        self.mutable()
        rows=list(csv.DictReader(io.StringIO(text.lstrip('\ufeff')))); accepted=[]; excluded=0
        for row in rows:
            if row.get('review_layer') != 'primary' or row.get('annotation_status') != 'complete' or row.get('rater_name','').strip().lower().startswith(('test','dummy')):
                excluded+=1; continue
            eid=row.get('episode_id',''); rid=row.get('rater_id','')
            if not eid or not rid: raise ValueError('Primary rows require episode_id and rater_id.')
            for m,v in values(row).items():
                if v and v not in ALLOWED[m]: raise ValueError(f'{eid}: invalid {m} value {v}')
            # Remove personal rater contact details; ID is sufficient for linkage.
            row={k:v for k,v in row.items() if k not in ['rater_email','rater_name']}
            key=digest([eid,rid,'primary']); prior=self.db.execute('SELECT payload FROM ratings WHERE id=?',(key,)).fetchone()
            if prior and json.loads(prior[0])!=row: raise ValueError('Conflicting saved ratings; export one consistent version.')
            accepted.append((key,eid,json.dumps(row,ensure_ascii=False)))
        with self.db:
            self.db.executemany('INSERT OR IGNORE INTO ratings VALUES(?,?,?)',accepted)
            self.log('import_ratings',{'accepted':len(accepted),'excluded':excluded})
        return len(accepted), excluded

    def ratings(self,eid):
        return [json.loads(r[0]) for r in self.db.execute('SELECT payload FROM ratings WHERE episode=? ORDER BY id',(eid,))]

    def create_split(self, seed=20261011):
        self.mutable()
        if self.setting('split'): raise ValueError('Split already assigned; reshuffling is blocked.')
        episodes=self.episodes()
        if not episodes: raise ValueError('Import episodes first.')
        groups={}
        for e in episodes: groups.setdefault(e['participant_id'],[]).append(e)
        if len(groups)<4: raise ValueError('At least four distinct participants required.')
        ids=sorted(groups);random.Random(seed).shuffle(ids)
        # Allocate whole participants; episode counts may differ from 150/75/75.
        n=len(ids); a=max(1,int(n*.5)); b=max(1,int(n*.25)); mapping={}
        for i,pid in enumerate(ids):
            split='train' if i<a else 'validation' if i<a+b else 'test'
            for e in groups[pid]: mapping[e['episode_id']]=split
        with self.db:
            self.set_setting('split',mapping); self.set_setting('split_seed',seed)
            self.log('assign_split',{'seed':seed,'hash':digest(mapping),'participants':n})
        return mapping

    def audit_agreement(self,eid):
        seed=self.setting('split_seed',20261011)
        return int(digest([seed,eid,'agreement-audit'])[:8],16)%10==0

    def save_reference(self,eid,measure,value,reviewer,evidence,reason):
        self.mutable()
        if value not in ALLOWED[measure] or not reviewer.strip() or not reason.strip(): raise ValueError('Valid value, reviewer and rationale required.')
        ep=next((e for e in self.episodes() if e['episode_id']==eid),None)
        if not ep: raise ValueError('Conversation missing.')
        tids={str(t['turn_id']) for t in ep['turns']}
        if not set(evidence)<=tids: raise ValueError('Evidence must use existing turn IDs.')
        if value not in ['unresolved','na','cannot_determine'] and not evidence: raise ValueError('Resolved observed labels require turn evidence.')
        if reviewer in {r['rater_id'] for r in self.ratings(eid)}: raise ValueError('Adjudicator must be independent of primary raters.')
        with self.db:
            old=self.db.execute('SELECT * FROM refs WHERE episode=? AND measure=?',(eid,measure)).fetchone()
            self.db.execute('INSERT OR REPLACE INTO refs VALUES(?,?,?,?,?,?,?,?)',(eid,measure,value,reviewer,json.dumps(evidence),reason,'adjudicated',now()))
            self.log('adjudicate',{'episode':eid,'measure':measure,'before':dict(old) if old else None,'value':value,'reviewer':reviewer,'reason':reason,'evidence':evidence})

    def reference(self,eid,measure):
        saved=self.db.execute('SELECT * FROM refs WHERE episode=? AND measure=?',(eid,measure)).fetchone()
        if saved: return dict(saved)
        rs=self.ratings(eid); vs=[values(r)[measure] for r in rs]
        # No automatic safety/critical negatives, unknowns, mixed rubrics or agreement audit cases.
        from .schema import BINARY
        if len(rs)==2 and len(set(vs))==1 and vs[0] not in ['','unresolved','cannot_determine','na','1'] and measure not in BINARY+['safety'] and len({r.get('rubric_version') for r in rs})==1 and not self.audit_agreement(eid):
            return {'episode':eid,'measure':measure,'value':vs[0],'source':'primary_agreement','reviewer':'two_primary_raters','evidence':'[]','reason':'Two matching primary ratings; agreement audit not selected.'}
        return None

    def add_candidate(self, provider,model,prompt,config):
        self.mutable()
        if provider not in ['openai','anthropic'] or not model.strip() or not prompt.strip(): raise ValueError('Provider, exact model ID and rubric prompt required.')
        from .judges import SYSTEM
        c={'provider':provider,'model':model.strip(),'prompt':prompt,'system':SYSTEM,'config':config}
        cid=digest(c)[:16]
        with self.db:
            self.db.execute('INSERT OR IGNORE INTO candidates VALUES(?,?)',(cid,json.dumps(c,ensure_ascii=False)))
            self.log('register_candidate',{'id':cid,'hash':digest(c)})
        return cid

    def candidates(self):
        return {r[0]:json.loads(r[1]) for r in self.db.execute('SELECT * FROM candidates')}

    def freeze(self, primary,secondary,gates,attestation):
        self.mutable()
        splits=self.setting('split'); cs=self.candidates()
        if not splits or primary not in cs or secondary not in cs or primary==secondary: raise ValueError('Split and two distinct candidates required.')
        if cs[primary]['provider']==cs[secondary]['provider']: raise ValueError('Select one OpenAI and one Anthropic candidate.')
        if not attestation.strip(): raise ValueError('Document holdout exposure and selection rationale.')
        # All reference decisions, including explicit unresolved, must precede scoring the locked test.
        for e in self.episodes():
            for m in MEASURES:
                if self.reference(e['episode_id'],m) is None: raise ValueError(f"Reference not reviewed: {e['episode_id']} / {m}")
        if not self.db.execute("SELECT 1 FROM predictions WHERE candidate=? AND status='ok' AND episode IN (SELECT id FROM episodes)",(primary,)).fetchone(): raise ValueError('Run candidates on development/validation before selection.')
        for cid in [primary,secondary]:
            for eid,split in splits.items():
                if split=='validation' and not self.db.execute("SELECT 1 FROM predictions WHERE candidate=? AND episode=? AND status='ok'",(cid,eid)).fetchone(): raise ValueError('Both chosen candidates need complete validation predictions.')
        snapshot={'primary':primary,'secondary':secondary,'candidates':{cid:cs[cid] for cid in [primary,secondary]},
                  'gates':gates,'split':splits,'episodes_hash':digest(self.episodes()),
                  'references':{e['episode_id']:{m:self.reference(e['episode_id'],m) for m in MEASURES} for e in self.episodes()},
                  'attestation':attestation,'frozen_at':now(),'engine_hash':engine_hash()}
        snapshot['hash']=digest(snapshot)
        with self.db: self.set_setting('freeze',snapshot);self.log('freeze_protocol',snapshot)
        return snapshot

    def save_prediction(self,cid,eid,payload,status='ok',error=''):
        # Never overwrite a successful frozen test result; transient failures can be retried.
        prev=self.db.execute('SELECT status FROM predictions WHERE candidate=? AND episode=?',(cid,eid)).fetchone()
        if prev and prev[0]=='ok': raise ValueError('Successful prediction already saved; reruns are blocked.')
        with self.db:
            self.db.execute('INSERT OR REPLACE INTO predictions VALUES(?,?,?,?,?,?)',(cid,eid,json.dumps(payload,ensure_ascii=False),status,error,now()))
            self.log('judge_result',{'candidate':cid,'episode':eid,'status':status,'error':error})

    def prediction(self,cid,eid):
        r=self.db.execute("SELECT payload FROM predictions WHERE candidate=? AND episode=? AND status='ok'",(cid,eid)).fetchone()
        return json.loads(r[0]) if r else None

    def report_records(self,cid,split,measure):
        frozen=self.setting('freeze'); mapping=self.setting('split',{});records=[]
        if frozen and frozen['engine_hash']!=engine_hash(): raise ValueError('Evaluation code changed after freeze; use the archived frozen version or a fresh study.')
        if split=='test' and not frozen: raise ValueError('Test results locked until protocol freeze.')
        for e in self.episodes():
            eid=e['episode_id']
            if mapping.get(eid)!=split: continue
            ref=frozen['references'][eid][measure] if frozen else self.reference(eid,measure)
            pred=self.prediction(cid,eid)
            records.append({'human':ref['value'] if ref else 'unresolved','prediction':pred['judgments'][measure]['value'] if pred else 'missing',
                            'participant_id':e['participant_id'],'weight':e['sampling_weight']})
        return records

    def export(self):
        return {t:[dict(r) for r in self.db.execute(f'SELECT * FROM {t}')] for t in ['settings','episodes','ratings','refs','candidates','predictions','audit']}
