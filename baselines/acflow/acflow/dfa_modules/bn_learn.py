import numpy as np
import tensorflow as tf
import copy
import itertools
import logging
from pgmpy.base import DAG
from pgmpy.estimators import ConstraintBasedEstimator

class GS(object):
    def __init__(self, hps, model):
        self.hps = hps
        self.model = model

    def mat2dict(self, mat):
        return {i:set(np.where(v)[0]) for i, v in enumerate(mat)}

    def dict2mat(self, dic):
        d = len(dic)
        mat = np.zeros([d, d], dtype=np.bool)
        for i, v in dic.items():
            mat[i, list(v)] = 1

        return mat

    def _batch_CMI(self, x, y, b, m):
        B, d = x.shape
        Nt = y.shape[1]
        N = b.shape[0]

        x = np.repeat(np.expand_dims(x, axis=1), N, axis=1)
        x = np.reshape(x, [B*N, d])
        y = np.repeat(np.expand_dims(y, axis=1), N, axis=1)
        y = np.reshape(y, [B*N, Nt])
        b = np.repeat(np.expand_dims(b, axis=0), B, axis=0)
        b = np.reshape(b, [B*N, d+Nt])
        m = np.repeat(np.expand_dims(m, axis=0), B, axis=0)
        m = np.reshape(m, [B*N, d+Nt])
        logp = self.model.run(self.model.logpu,
            {self.model.x:x, self.model.y:y, self.model.bxy:b, self.model.mxy:m})
        logp = np.reshape(logp, [B, N])
        logp1 = logp[:,:N//2]
        logp2 = logp[:,N//2:]
        log_ratio = logp1 - logp2

        return log_ratio

    def _CMI(self, x, y, i, j, cond):
        '''
        compute CMI for a list of conditions
        '''
        N = len(cond)
        d = x.shape[1]
        Nt = y.shape[1]
        b2 = np.zeros([N, d+Nt], dtype=np.float32)
        for n, c in enumerate(cond):
            if c:
                np.put_along_axis(b2[n], np.array(c), 1., 0)
        m2 = b2.copy()
        m2[:,i] = 1.
        b1 = b2.copy()
        b1[:,j] = 1.
        m1 = b1.copy()
        m1[:,i] = 1.
        m = np.concatenate([m1, m2], axis=0) # [N*2, d]
        b = np.concatenate([b1, b2], axis=0) # [N*2, d]
        # run
        B = self.hps.batch_size
        num_batches = x.shape[0] // B        
        cmi_est = []
        for n in range(num_batches):
            xx = x[n*B:(n+1)*B]
            yy = y[n*B:(n+1)*B]
            cmi = self._batch_CMI(xx, yy, b, m)
            cmi_est.append(cmi)
        cmi_est = np.concatenate(cmi_est, axis=0)
        cmi = np.mean(cmi_est, axis=0)

        return cmi

    def _CNMI(self, x, y, i, j, cond):
        cmi = self._CMI(x, y, i, j, cond)
        if self.hps.normalize:
            ent_i = self._CMI(x, y, i, i, cond)
            ent_j = self._CMI(x, y, j, j, cond)
            cmi = cmi / np.sqrt(ent_i * ent_j)

        return cmi

    def resolve_mb(self, x, y, mb):
        '''
        test if i, j are direct neighbors
        compute CMI(i, j | cond)
        if any (CMI) < thresh: False
        else: True
        '''
        d = x.shape[1]
        Nt = y.shape[1]
        mb_dict = self.mat2dict(mb)
        edge_dict = dict([(n, set()) for n in range(d+Nt)])
        for i, mb_i in mb_dict.items():
            for j in mb_i:
                T = min(mb_i-{j}, mb_dict[j]-{i}, key=len)
                cond = itertools.chain(*[itertools.combinations(T, t) for t in range(len(T)+1)])
                cond = list(cond)
                cmi = self._CNMI(x, y, i, j, cond)
                if np.all(cmi > self.hps.cmi_thresh):
                    edge_dict[i].add(j)
                    edge_dict[j].add(i)
                # is_edge = True
                # for c in cond:
                #     cmi = self._CNMI(x, y, i, j, [c])
                #     if cmi <= self.hps.cmi_thresh:
                #         is_edge = False
                #         break
                # if is_edge:
                #     edge_dict[i].add(j)
                #     edge_dict[j].add(i)              
        
        return edge_dict

    def orient_edges(self, x, y, mb, edge_dict):
        mb_dict = self.mat2dict(mb)
        oriented = copy.deepcopy(edge_dict)
        for i, edge_i in edge_dict.items():
            for j in edge_i:
                nxy = edge_i - edge_dict[j] - {j}
                for k in nxy:
                    oriented[j] -= {i}
                    by = mb_dict[j] - {i} - {k}
                    bz = mb_dict[k] - {i} - {j}
                    T = min(by, bz, key=len)
                    cond = itertools.chain(*[itertools.combinations(T, t) for t in range(len(T)+1)])
                    cond = [c+(i,) for c in cond]
                    cmi = self._CNMI(x, y, j, k, cond)
                    if np.any(cmi <= self.hps.cmi_thresh):
                        oriented[j].add(i)
                    else:
                        break
        
        return oriented

    def pdag2dag(self, edge_dict):
        pdag_edges = [(pi,n) for n,p in edge_dict.items() for pi in p]
        pdag = DAG(pdag_edges)
        dag_edges = ConstraintBasedEstimator.pdag_to_dag(pdag).edges()
        dag = dict([(n, set()) for n in range(len(edge_dict))])
        for e in dag_edges:
            dag[e[1]].add(e[0])

        return dag

    def __call__(self, x, y, mb):
        edge_dict = self.resolve_mb(x, y, mb)
        logging.info('edge_dict:')
        logging.info(edge_dict)
        oriented_edges = self.orient_edges(x, y, mb, edge_dict)
        logging.info('oriented: ')
        logging.info(oriented_edges)
        DAG = self.pdag2dag(oriented_edges)
        logging.info('dag: ')
        logging.info(DAG)
        graph = self.dict2mat(DAG)

        return graph