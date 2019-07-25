#!/usr/bin/env python3

"""
analysis.py

This is for analyzing results.

"""

import os
import pandas as pd
import numpy as np
from src.iofunc import open_pkl
from src.model import Model


class Analysis:

    def __init__(self, p_data=''):

        self.p_data = p_data if p_data and p_data.endswith('/') else p_data + '/'
        self.use_f1 = False

    def compile_batch_train_results(self, d_mods):
        p_mods = [d_mods + x for x in os.listdir(d_mods) if x.endswith('.pkl')]
        data = None

        n_samples = 0

        if p_mods:
            # Get first one
            data, n = self.compile_train_results(p_mod=p_mods[0])
            n_samples += n
            print('Gathered: %s' % p_mods[0])

            # Get the other ones if there are others
            if len(p_mods) > 1:
                for p_mod in p_mods[1:]:
                    tmp, n = self.compile_train_results(p_mod=p_mod)
                    data = data.append(tmp, ignore_index=True)
                    n_samples += n
                    print('Gathered: %s' % p_mod)

        print('\nNumber of samples: %d\n' % n_samples)

        return data

    @staticmethod
    def compile_train_results(p_mod):
        n_samples = 0

        m_pkg, results = open_pkl(p_file=p_mod)

        model = Model()
        model.import_model(m_pkg)
        modelid = model.id.split('-')[-1]
        aim = model.aim if model.aim else 'no-aim'
        filename = os.path.split(p_mod)[1]

        data = {'clf': [],
                'type': [],
                'value': [],
                'kcv': [],
                'irep': [],
                'aim': [],
                'lab': [],
                'modelid': [],
                'file':[],
                }

        layerkey = None
        if model.datas[model.train_idx].columns['layers'] in results.columns:
            layerkey = model.datas[model.train_idx].columns['layers']
            data['layers'] = []

        clf_code = {'ybkg': 'Baseline',
                    'yinf': 'Inf',
                    'ymatch': 'Match',
                    'ynet': 'Net',
                    'yopt': 'Optimized',
                    }

        for x in ['ybkg', 'yinf', 'ymatch', 'ynet', 'yopt']:

            if x in ['yinf', 'ymatch', 'ynet']:
                if x == 'yinf':
                    idx = results['inf'].values
                elif x == 'ymatch':
                    idx = results['match'].values
                else:
                    idx = results['net'].values
            else:
                # Evaluate opt on all samples
                # this looks silly but easier to just do this given how I stores the arrays in pandas dataframe
                idx = results['net'].values | np.invert(results['net'].values)

            irep = 0
            for i, (y0, y1) in enumerate(zip(results['ytruth'], results[x])):

                r = model.scores(y0[idx[i]], y1[idx[i]])
                predictable = sum(idx[i]) / len(idx[i]) * 100
                predictables = np.sum(y0[idx[i]], axis=0) / len(y0) * 100
                vlayer = results[layerkey][i] if layerkey else None

                if layerkey:
                    # This is a bit more complicated
                    # For each layer, there n-reps on all other layers and an eval on the one-outed layer
                    # The one-outed layer should not count since it didn't participate in training
                    j = i % (model.kfold_cv * model.n_repeat + 1)
                    irep += 1 if j % model.kfold_cv == 0 and j != (model.kfold_cv * model.n_repeat) else 0
                else:
                    irep = i // model.kfold_cv + 1

                # Append AUC-ROC
                data['clf'].append(clf_code[x])
                data['type'].append('aucroc')
                data['value'].append(r['aucroc'])
                data['kcv'].append(model.kfold_cv)
                data['irep'].append(irep)
                data['aim'].append(aim)
                data['lab'].append('all')
                if layerkey:
                    data['layers'].append(vlayer)

                # Append F1
                data['clf'].append(clf_code[x])
                data['type'].append('f1')
                data['value'].append(r['f1'])
                data['kcv'].append(model.kfold_cv)
                data['irep'].append(irep)
                data['aim'].append(aim)
                data['lab'].append('all')
                if layerkey:
                    data['layers'].append(vlayer)

                # Append Precision
                data['clf'].append(clf_code[x])
                data['type'].append('precision')
                data['value'].append(r['precision'])
                data['kcv'].append(model.kfold_cv)
                data['irep'].append(irep)
                data['aim'].append(aim)
                data['lab'].append('all')
                if layerkey:
                    data['layers'].append(vlayer)

                # Append Recall
                data['clf'].append(clf_code[x])
                data['type'].append('recall')
                data['value'].append(r['recall'])
                data['kcv'].append(model.kfold_cv)
                data['irep'].append(irep)
                data['aim'].append(aim)
                data['lab'].append('all')
                if layerkey:
                    data['layers'].append(vlayer)

                # Append Coverage
                data['clf'].append(clf_code[x])
                data['type'].append('coverage')
                data['value'].append(model.calc_coverage(y1) * 100)
                data['kcv'].append(model.kfold_cv)
                data['irep'].append(irep)
                data['aim'].append(aim)
                data['lab'].append('all')
                if layerkey:
                    data['layers'].append(vlayer)

                # Append Predictable
                data['clf'].append(clf_code[x])
                data['type'].append('predictable')
                data['value'].append(predictable)
                data['kcv'].append(model.kfold_cv)
                data['irep'].append(irep)
                data['aim'].append(aim)
                data['lab'].append('all')
                if layerkey:
                    data['layers'].append(vlayer)

                # AUC-ROC per lab
                cov_idx = np.any(y1, axis=1)
                coverages = np.sum(y0[cov_idx], axis=0) / len(y0) * 100
                for j, lab in enumerate(model.datas[model.train_idx].labels):
                    data['clf'].append(clf_code[x])
                    data['type'].append('aucroc_labs')
                    data['value'].append(r['aucroc_labs'][j])
                    data['kcv'].append(model.kfold_cv)
                    data['irep'].append(irep)
                    data['aim'].append(aim)
                    data['lab'].append(lab)
                    if layerkey:
                        data['layers'].append(vlayer)

                    data['clf'].append(clf_code[x])
                    data['type'].append('f1_labs')
                    data['value'].append(r['f1_labs'][j])
                    data['kcv'].append(model.kfold_cv)
                    data['irep'].append(irep)
                    data['aim'].append(aim)
                    data['lab'].append(lab)
                    if layerkey:
                        data['layers'].append(vlayer)

                    data['clf'].append(clf_code[x])
                    data['type'].append('precision_labs')
                    data['value'].append(r['precision_labs'][j])
                    data['kcv'].append(model.kfold_cv)
                    data['irep'].append(irep)
                    data['aim'].append(aim)
                    data['lab'].append(lab)
                    if layerkey:
                        data['layers'].append(vlayer)

                    data['clf'].append(clf_code[x])
                    data['type'].append('recall_labs')
                    data['value'].append(r['recall_labs'][j])
                    data['kcv'].append(model.kfold_cv)
                    data['irep'].append(irep)
                    data['aim'].append(aim)
                    data['lab'].append(lab)
                    if layerkey:
                        data['layers'].append(vlayer)

                    data['clf'].append(clf_code[x])
                    data['type'].append('lab_predictable')
                    data['value'].append(predictables[j])
                    data['kcv'].append(model.kfold_cv)
                    data['irep'].append(irep)
                    data['aim'].append(aim)
                    data['lab'].append(lab)
                    if layerkey:
                        data['layers'].append(vlayer)

                    data['clf'].append(clf_code[x])
                    data['type'].append('lab_coverage')
                    data['value'].append(coverages[j])
                    data['kcv'].append(model.kfold_cv)
                    data['irep'].append(irep)
                    data['aim'].append(aim)
                    data['lab'].append(lab)
                    if layerkey:
                        data['layers'].append(vlayer)

                n_samples += 1

        data['modelid'] = modelid
        data['file'] = filename

        return pd.DataFrame(data), n_samples

    @staticmethod
    def normalize(data, clf='Optimized', bl='Baseline'):
        # This works because all the data for each clf/bkg are loaded in the same order
        # Normalize every metrics to baseline or whatever set as baseline except AUC-ROC
        # Because AUC-ROC already has an absolute baseline for random at 0.5
        clfidx = data['clf'] == clf
        blidx = data['clf'] == bl

        types = ['aucroc', 'f1', 'precision', 'recall']
        dfx = pd.DataFrame()
        for k in types:
            idx = data['type'] == k
            d = data[clfidx & idx]
            if k != 'aucroc':
                d['value'] = data[clfidx & idx]['value'].values - data[blidx & idx]['value'].values
            dfx = dfx.append(d, ignore_index=True)
        dfx = dfx.append(data[clfidx & (data['type'] == 'predictable')], ignore_index=True)
        dfx = dfx.append(data[clfidx & (data['type'] == 'coverage')], ignore_index=True)

        types = ['aucroc_labs', 'f1_labs', 'precision_labs', 'recall_labs']
        for k in types:
            idx = data['type'] == k
            d = data[clfidx & idx]
            if k != 'aucroc_labs':
                d['value'] = data[clfidx & idx]['value'].values - data[blidx & idx]['value'].values
            dfx = dfx.append(d, ignore_index=True)
        dfx = dfx.append(data[clfidx & (data['type'] == 'lab_predictable')], ignore_index=True)
        dfx = dfx.append(data[clfidx & (data['type'] == 'lab_coverage')], ignore_index=True)

        return dfx
