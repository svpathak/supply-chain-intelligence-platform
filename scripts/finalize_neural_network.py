"""Final, bounded MLP comparison: 4 vs 8 hidden units with pre-booking origin context."""
from __future__ import annotations
import json, warnings
from datetime import datetime, timezone
from pathlib import Path
import sys
ROOT_FOR_IMPORT = Path(__file__).resolve().parents[1]
if str(ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(ROOT_FOR_IMPORT))
import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.metrics import (roc_auc_score, average_precision_score, log_loss,
    precision_score, recall_score, f1_score, confusion_matrix)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from app.ml.features import (CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES,
    TARGET_COLUMN, load_training_frame, split_chronologically)
from app.ml.port_event_features import EVENT_FEATURE_COLUMNS, add_point_in_time_port_event_features
from app.ml.train import _random_baseline

ROOT = Path(__file__).resolve().parents[1]
ORIGIN_EVENT_FEATURES = [c for c in EVENT_FEATURE_COLUMNS if c.startswith('origin_')] + [
    'event_observation_7d_coverage', 'event_observation_30d_coverage']
FEATURES = list(FEATURE_COLUMNS) + ORIGIN_EVENT_FEATURES
NUMERIC = list(NUMERIC_FEATURES) + ORIGIN_EVENT_FEATURES

def build_model(units: int) -> Pipeline:
    pre = ColumnTransformer([
        ('numeric', Pipeline([('imputer', SimpleImputer(strategy='median', add_indicator=True)),
                              ('scaler', StandardScaler())]), NUMERIC),
        ('categorical', Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='UNKNOWN')),
                                  ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]),
         CATEGORICAL_FEATURES),
    ], sparse_threshold=0)
    return Pipeline([('preprocess', pre), ('classifier', MLPClassifier(
        hidden_layer_sizes=(units,), activation='relu', solver='adam', alpha=0.01,
        batch_size='auto', learning_rate_init=0.001, max_iter=2500, tol=1e-5,
        n_iter_no_change=50, early_stopping=False, random_state=42))])

def evaluate(model, part):
    y = part[TARGET_COLUMN].astype(int)
    p = model.predict_proba(part[FEATURES])[:, 1]
    pred = (p >= 0.5).astype('int8')
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0,1]).ravel()
    return {'rows':len(y), 'actual_class_counts':{'0':int((y==0).sum()),'1':int((y==1).sum())},
        'predicted_class_counts':{'0':int((pred==0).sum()),'1':int((pred==1).sum())},
        'roc_auc':float(roc_auc_score(y,p)), 'average_precision':float(average_precision_score(y,p)),
        'log_loss':float(log_loss(y,p,labels=[0,1])), 'precision':float(precision_score(y,pred,zero_division=0)),
        'recall':float(recall_score(y,pred,zero_division=0)), 'f1':float(f1_score(y,pred,zero_division=0)),
        'confusion_matrix':{'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp)},
        'random_50_percent_baseline':_random_baseline(y,seed=42)}

def main():
    frame, prep = load_training_frame(ROOT/'data/raw')
    enriched, audit = add_point_in_time_port_event_features(frame, ROOT/'data/raw')
    train, val, _test, split = split_chronologically(enriched)
    results, models, warnings_by_model = {}, {}, {}
    for units in (4,8):
        name=f'mlp_{units}_units_plus_origin_events'
        model=build_model(units)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            model.fit(train[FEATURES],train[TARGET_COLUMN].astype(int))
        warnings_by_model[name]=[str(w.message) for w in caught if issubclass(w.category,ConvergenceWarning)]
        results[name]=evaluate(model,val)
        clf=model.named_steps['classifier']
        results[name].update({'hidden_units':units,'training_iterations':int(clf.n_iter_),
            'final_training_loss':float(clf.loss_), 'transformed_dimensions':int(model.named_steps['preprocess'].transform(train[FEATURES].iloc[:1]).shape[1])})
        models[name]=model
    # The 8-unit architecture is the final experimental candidate. It is only
    # promoted if validation evidence clears both no-skill and the 4-unit model.
    candidate=results['mlp_8_units_plus_origin_events']
    reference=results['mlp_4_units_plus_origin_events']
    promote=(candidate['roc_auc']>0.5 and candidate['average_precision']>candidate['random_50_percent_baseline']['expected_average_precision']
             and candidate['roc_auc']>reference['roc_auc'] and candidate['log_loss']<reference['log_loss'])
    artifact='artifacts/delay_mlp_8_units_plus_origin_events_experiment.joblib'
    if promote:
        joblib.dump(models['mlp_8_units_plus_origin_events'],ROOT/artifact)
    else:
        # Preserve as a reproducible experiment, not the API production artifact.
        joblib.dump(models['mlp_8_units_plus_origin_events'],ROOT/artifact)
    report={'generated_at_utc':datetime.now(timezone.utc).isoformat(),
        'status':'completed_not_promoted' if not promote else 'validation_candidate_promoted_for_final_review',
        'dataset_preparation':prep,'point_in_time_event_audit':audit,'chronological_split':split,
        'feature_count':len(FEATURES),'origin_event_features':ORIGIN_EVENT_FEATURES,
        'architecture':{'hidden_layers':1,'candidate_hidden_units':8,'activation':'ReLU','output':'sigmoid probability',
            'optimizer':'Adam','learning_rate':0.001,'l2_alpha':0.01,'early_stopping':False,
            'random_state':42,'selection':'single bounded capacity step from 4 to 8 hidden units'},
        'validation_results':results,'convergence_warnings':warnings_by_model,
        'promotion_rule':'Candidate must beat ROC-AUC 0.5, exceed expected random AP, improve ROC-AUC over 4-unit model, and improve log loss.',
        'promoted':bool(promote),'artifact':artifact,
        'test_policy':'No test metrics used or reported; the chronological test set was previously inspected and is not pristine.',
        'limitations':['Port-event features are port-level context, not shipment- or vessel-leg-attributed events.',
          'Naive source timestamps have no timezone metadata.',
          'Validation has been reused in earlier experiments; results are exploratory and not an unbiased final estimate.']}
    (ROOT/'reports/final_neural_network_decision.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
if __name__=='__main__': main()
