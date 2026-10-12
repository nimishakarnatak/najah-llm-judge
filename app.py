import hmac
import json
import os
import time
from collections import Counter
from pathlib import Path
import pandas as pd
import streamlit as st
from najah.schema import MEASURES, ORDINAL, BINARY, ALLOWED, values
from najah.store import Store
from najah.judges import run_batch
from najah.metrics import evaluate

st.set_page_config(page_title='Najah Stage 2',page_icon='ن',layout='wide')

def authenticate():
    expected=os.environ.get('APP_PASSWORD','')
    adjudicator=os.environ.get('ADJUDICATOR_PASSWORD','')
    if not expected or not adjudicator or hmac.compare_digest(expected,adjudicator):
        st.error('Set distinct APP_PASSWORD and ADJUDICATOR_PASSWORD values in Replit Secrets.');st.stop()
    if not st.session_state.get('authenticated'):
        st.title('Najah Stage 2')
        role=st.selectbox('Workspace',['Study analyst','Human adjudicator'])
        entered=st.text_input('Study workspace password',type='password')
        if st.button('Sign in'):
            if time.time()<st.session_state.get('retry_after',0):
                st.error('Please wait before trying again.')
            elif hmac.compare_digest(entered,expected if role=='Study analyst' else adjudicator):
                st.session_state['authenticated']=True;st.session_state['role']=role;st.rerun()
            else:
                st.session_state['retry_after']=time.time()+3;st.error('Password not recognized.')
        st.stop()

authenticate()
store=Store(os.environ.get('NAJAH_DB_PATH','data/study.sqlite'))
frozen=store.setting('freeze');episodes=store.episodes();split=store.setting('split',{})
gates=json.loads(Path('config/gates.json').read_text())
if frozen: gates=frozen['gates']
st.title('Najah Stage 2')
st.caption('Build the human reference, compare judges, then evaluate the locked test.')
pages=['Human adjudication'] if st.session_state['role']=='Human adjudicator' else ['Data and splits','Reference quality','Judge development','Freeze protocol','Locked test','Exports']
page=st.sidebar.radio('Study workflow',pages)
st.sidebar.write('Protocol frozen' if frozen else 'Development open')
st.sidebar.metric('Episodes',len(episodes))
if st.sidebar.button('Sign out'):
    st.session_state.clear();st.rerun()

def display_report(cid,part):
    results=[evaluate(m,store.report_records(cid,part,m),gates[m]) for m in MEASURES]
    rows=[]
    for r in results:
        rows.append({'Measure':r['measure'],'Decision':r['status'],'Reference cases':r['n_reference'],
                     'Paired cases':r['n_paired'],'Participant clusters':r['n_clusters'],
                     'Prediction coverage':round(r['prediction_coverage'],3),
                     'Kappa':r.get('kappa'),'MAE':r.get('mae'),'Recall':r.get('recall'),
                     'Macro F1':r.get('macro_f1'),'Reasons':'; '.join(r['reasons'])})
    st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True)
    st.download_button('Download metrics with confidence intervals',json.dumps(results,indent=2),f'{cid}-{part}-metrics.json',mime='application/json')
    with st.expander('Confidence intervals and class counts'): st.json(results)
    return results

try:
    if page=='Data and splits':
        st.subheader('Import study records')
        st.write('Import de-identified original-language episodes as JSON and primary ratings as CSV. Imports preserve existing records and reject conflicts. Test and demo ratings are excluded; judge reviews do not become reference labels automatically.')
        if frozen: st.info('Imports are closed for this frozen study.')
        else:
            ep=st.file_uploader('Episodes JSON',type=['json'])
            if ep and st.button('Import episodes'):
                data=json.loads(ep.getvalue().decode('utf-8-sig'))
                n=store.import_episodes(data['episodes'] if isinstance(data,dict) else data)
                st.success(f'Imported {n} episodes.');st.rerun()
            ratings=st.file_uploader('Primary or combined ratings CSV',type=['csv'])
            if ratings and st.button('Import ratings'):
                n,x=store.import_ratings(ratings.getvalue().decode('utf-8-sig'));st.success(f'Accepted {n} primary records; excluded {x} non-study or incomplete records.')
            if not split:
                st.write('Assign approximately 50% of participants to training, 25% to validation and 25% to test. All episodes from a participant stay together. Allocation is performed once; check group balance afterward.')
                seed=st.number_input('Split seed',value=20261011,min_value=0,step=1)
                if st.button('Assign participant-separated splits',disabled=not episodes): store.create_split(int(seed));st.rerun()
        if episodes:
            metadata=pd.DataFrame([{k:e.get(k,'') for k in ['episode_id','participant_id','module','language','treatment','sampling_weight']} | {'split':split.get(e['episode_id'],'unassigned')} for e in episodes])
            st.dataframe(metadata.groupby(['split','module','language'],dropna=False).size().reset_index(name='episodes'),hide_index=True)
            for part in ['train','validation','test']:
                subset=metadata[metadata['split']==part];st.write(f"{part}: {len(subset)} episodes / {subset.participant_id.nunique()} participants")
            unknown=sum(e['language'] in ['','und','unknown'] for e in episodes)
            if unknown: st.warning(f'{unknown} episodes have unknown language. Language-specific conclusions remain unavailable.')
            rated={r[0] for r in store.db.execute('SELECT DISTINCT episode FROM ratings')}
            missing=rated-{e['episode_id'] for e in episodes}
            if missing: st.warning(f'{len(missing)} rated episodes are missing conversation records.')
    elif page=='Reference quality':
        st.subheader('Human reference quality')
        st.caption('Before freeze, only training and validation summaries are available to analysts. Missing, uncertain and inapplicable labels remain visible.')
        parts=['train','validation','test'] if frozen else ['train','validation']
        part=st.selectbox('Reference split',parts)
        from najah.metrics import ordinal_stats
        summaries=[]
        for m in MEASURES:
            pairs=[];adjudicated=0;unresolved=0;missing=0
            for e in episodes:
                eid=e['episode_id']
                if split.get(eid)!=part: continue
                rs=store.ratings(eid);vs=[values(r)[m] for r in rs]
                ref=store.reference(eid,m)
                adjudicated+=bool(ref and ref['source']=='adjudicated')
                unresolved+=bool(ref and ref['value'] in ['unresolved','cannot_determine'])
                missing+=ref is None
                if len(vs)==2: pairs.append((vs[0],vs[1],e['sampling_weight']))
            row={'measure':m,'two_rater_cases':len(pairs),'exact_agreement':sum(a==b for a,b,w in pairs)/len(pairs) if pairs else None,
                 'adjudicated':adjudicated,'unresolved':unresolved,'awaiting_review':missing}
            numeric=[(a,b,w) for a,b,w in pairs if a in ['1','2','3'] and b in ['1','2','3']]
            if m in ORDINAL and numeric:
                a,b,w=zip(*numeric);row.update(ordinal_stats(a,b,w));row['numeric_pairs']=len(numeric)
            if m in BINARY:
                row['any_positive']=sum(a=='yes' or b=='yes' for a,b,w in pairs)
                row['both_positive']=sum(a==b=='yes' for a,b,w in pairs)
            summaries.append(row)
        st.dataframe(pd.DataFrame(summaries),hide_index=True,use_container_width=True)
        st.download_button('Download human agreement summary',pd.DataFrame(summaries).to_csv(index=False),'human-agreement.csv',mime='text/csv')
    elif page=='Human adjudication':
        st.subheader('Build an independent human reference')
        st.write('Review disagreements, missing or uncertain labels, every safety and critical-failure label, and a reproducible 10% sample of agreement cases. Reviewers see original text and human ratings, without LLM predictions or treatment labels. Use your own reviewer ID; do not enter a primary rater ID.')
        if not episodes: st.info('Import conversation records first.');st.stop()
        pending_only=st.checkbox('Show episodes needing reference decisions',value=True)
        available=[e for e in episodes if not pending_only or any(store.reference(e['episode_id'],m) is None for m in MEASURES)]
        if not available: st.success('All reference decisions recorded.');st.stop()
        eid=st.selectbox('Episode',[e['episode_id'] for e in available])
        episode=next(e for e in available if e['episode_id']==eid)
        left,right=st.columns([1.2,1])
        with left:
            st.write(f"Module: {episode['module']} · Language: {episode['language']}")
            show_english=False
            if any(t.get('text_en') for t in episode['turns']):
                show_english=st.checkbox('Show English reading support',value=False)
                if show_english: st.caption('Machine translation; consult the original when meaning is uncertain. Judges use original text.')
            for turn in episode['turns']:
                st.text(f"Turn {turn['turn_id']} · {turn['role']} · {turn.get('segment','episode')}")
                st.text(turn['text'])
                if show_english and turn.get('text_en'): st.text('English: '+turn['text_en'])
            if episode.get('delivery_evidence'): st.json(episode['delivery_evidence'])
        with right:
            rs=store.ratings(eid)
            st.dataframe(pd.DataFrame([{'rater_id':r['rater_id'],'rubric':r.get('rubric_version'),**values(r)} for r in rs]),hide_index=True)
            m=st.selectbox('Measure',MEASURES)
            ref=store.reference(eid,m)
            if ref: st.write(f"Current reference: {ref['value']} ({ref['source']})")
            if frozen: st.info('Reference decisions frozen.');st.stop()
            with st.form(f'adjudicate-{eid}-{m}'):
                v=st.selectbox('Reference label',ALLOWED[m],index=ALLOWED[m].index(ref['value']) if ref else len(ALLOWED[m])-1)
                reviewer=st.text_input('Independent reviewer ID')
                evidence=st.multiselect('Evidence turn IDs',[t['turn_id'] for t in episode['turns']])
                reason=st.text_area('Decision rationale or reason unresolved')
                confirmed=st.checkbox('I reviewed the original record independently of LLM outputs and treatment assignment.')
                if st.form_submit_button('Save reference decision'):
                    if not confirmed: raise ValueError('Independent review attestation required.')
                    store.save_reference(eid,m,v,reviewer,evidence,reason);st.rerun()
    elif page=='Judge development':
        st.subheader('Compare OpenAI and Anthropic judges')
        st.write('Training means rubric and prompt development, not automatic model fine-tuning. Select the primary judge using validation results before freezing. Every prompt change creates a separate candidate.')
        if frozen: st.info('Development closed. Use Locked test for the frozen judges.');st.stop()
        with st.form('candidate'):
            provider=st.selectbox('Provider',['openai','anthropic'])
            model=st.text_input('Exact model ID available in your provider account')
            prompt=st.text_area('Rubric and scoring instructions',Path('config/rubric.txt').read_text(encoding='utf-8'),height=250)
            cap=st.number_input('Maximum output tokens',min_value=1000,max_value=32000,value=6000,step=500)
            acknowledged=st.checkbox('I reviewed the rubric anchors and categorical definitions for this candidate.')
            if st.form_submit_button('Register candidate'):
                if not acknowledged: raise ValueError('Review rubric before registering.')
                cid=store.add_candidate(provider,model,prompt,{'max_output_tokens':int(cap)});st.success(f'Registered {cid}')
        cs=store.candidates()
        if cs:
            cid=st.selectbox('Candidate',list(cs),format_func=lambda x:f"{cs[x]['provider']} · {cs[x]['model']} · {x}")
            part=st.selectbox('Development split',['train','validation'])
            limit=st.number_input('Episodes in this batch',min_value=1,max_value=100,value=10,step=1)
            st.caption('Running a batch sends original-language conversation data to the selected provider and incurs provider charges. Failed requests remain visible and may be retried; successful results are retained.')
            if st.button('Run judge batch',disabled=not split):
                progress=st.progress(0.0)
                n=run_batch(store,cid,part,int(limit),lambda i,n:progress.progress(i/n));st.success(f'Processed {n} episodes.')
            if st.button('Calculate development metrics'): display_report(cid,part)
            if st.button('Compare all candidates on validation'):
                comparison=[]
                for candidate_id,candidate in cs.items():
                    for m in MEASURES:
                        r=evaluate(m,store.report_records(candidate_id,'validation',m),gates[m])
                        comparison.append({'candidate':candidate_id,'provider':candidate['provider'],'model':candidate['model'],
                                           'measure':m,'decision':r['status'],'coverage':r['prediction_coverage'],'kappa':r.get('kappa'),
                                           'mae':r.get('mae'),'recall':r.get('recall'),'macro_f1':r.get('macro_f1')})
                st.dataframe(pd.DataFrame(comparison),hide_index=True,use_container_width=True)
                st.download_button('Download validation comparison',pd.DataFrame(comparison).to_csv(index=False),'validation-comparison.csv',mime='text/csv')
            with st.expander('Saved provider errors'):
                st.dataframe(pd.DataFrame([dict(r) for r in store.db.execute("SELECT candidate,episode,error,updated FROM predictions WHERE status='error'")]),hide_index=True)
    elif page=='Freeze protocol':
        st.subheader('Freeze before opening the test')
        st.write('Freeze the selected judges, rubric, prompts, split, human reference and per-measure gates. The app then prevents edits and development scoring. Failed test requests can resume; successful test judgments cannot be rerun. A failed validation requires a new study with a fresh test cohort.')
        if frozen:
            st.success(f"Protocol frozen at {frozen['frozen_at']}");st.json({k:frozen[k] for k in ['primary','secondary','hash','attestation']});st.stop()
        st.write('Proposed gates are research decisions, not established universal standards. Review config/gates.json before freezing.')
        st.json(gates)
        cs=store.candidates()
        if len(cs)<2: st.info('Register and score at least one candidate from each provider.');st.stop()
        primary=st.selectbox('Primary judge selected on validation',list(cs),format_func=lambda x:f"{cs[x]['provider']} / {cs[x]['model']} / {x}")
        secondary=st.selectbox('Secondary judge from the other provider',list(cs),index=1,format_func=lambda x:f"{cs[x]['provider']} / {cs[x]['model']} / {x}")
        rationale=st.text_area('Selection rationale and prior exposure of proposed test episodes','State which criteria drove selection, how validation results were compared, and whether any test episode was previously used for rubric or prompt tuning. If exposed, label the test retrospective and plan fresh confirmation.')
        attested=st.checkbox('Reference review is complete; the team has approved these gates before inspecting test judgments.')
        if st.button('Freeze study protocol'):
            if not attested or rationale.startswith('State which'): raise ValueError('Record a substantive rationale and complete the attestation.')
            store.freeze(primary,secondary,gates,rationale);st.rerun()
    elif page=='Locked test':
        st.subheader('Evaluate frozen judges')
        if not frozen: st.info('Test scoring and results remain locked until protocol freeze.');st.stop()
        cid=st.selectbox('Frozen judge',[frozen['primary'],frozen['secondary']],format_func=lambda x:f"{'Primary' if x==frozen['primary'] else 'Secondary'} · {frozen['candidates'][x]['model']}")
        limit=st.number_input('Episodes in this batch',min_value=1,max_value=100,value=10,step=1)
        if st.button('Run locked test batch'):
            progress=st.progress(0.0);n=run_batch(store,cid,'test',int(limit),lambda i,n:progress.progress(i/n));st.success(f'Processed {n} episodes.')
        if st.button('Evaluate each measure'):
            results=display_report(cid,'test')
            passed=[r['measure'] for r in results if r['status']=='pass']
            st.write('Measures meeting the proposed statistical gates: '+(', '.join(passed) if passed else 'none'))
            st.caption('A statistical pass requires subsequent human evidence audit and confirmation of intended language/module scope. Safety score validation does not establish critical-failure detection. This app does not automatically approve corpus deployment.')
    elif page=='Exports':
        st.subheader('Download study records')
        st.write('Exports contain conversation records and reference labels. Keep them within the approved research team. The audit trail preserves changes, freeze decisions and judge runs. Back up the database before stopping or migrating the Replit workspace.')
        if frozen:
            st.download_button('Download complete study backup JSON',json.dumps(store.export(),ensure_ascii=False,indent=2),'najah-study-backup.json',mime='application/json')
        else:
            st.info('Full backups and test reference exports are sealed in the analyst UI until freeze. The data custodian can back up the SQLite file directly without disclosing it to judge developers.')
        if split: st.download_button('Download split manifest',pd.DataFrame(list(split.items()),columns=['episode_id','split']).to_csv(index=False),'split-manifest.csv',mime='text/csv')
        refs=[store.reference(e['episode_id'],m) for e in episodes if frozen or split.get(e['episode_id']) in ['train','validation'] for m in MEASURES]
        st.download_button('Download reference labels',pd.DataFrame([r for r in refs if r]).to_csv(index=False),'reference-labels.csv',mime='text/csv')
        if frozen: st.download_button('Download frozen protocol',json.dumps(frozen,ensure_ascii=False,indent=2),'frozen-protocol.json',mime='application/json')
except (ValueError,KeyError,json.JSONDecodeError) as exc:
    st.error(str(exc))
finally:
    store.db.close()
