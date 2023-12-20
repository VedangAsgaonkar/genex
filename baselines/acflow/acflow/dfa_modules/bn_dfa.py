import numpy as np
import tensorflow as tf
from scipy.special import softmax
from scipy.stats import entropy
from .BN import BN

class DFA(object):
    def __init__(self, hps, model, graph):
        self.hps = hps
        self.n_node = len(graph)
        self.graph = graph
        self.model = model
        self.cache = {}

        self.dag = BN()
        for n in range(self.n_node):
            self.dag.add_node(str(n))
        for n in range(self.n_node):
            for p in range(self.n_node):
                if graph[n, p]:
                    self.dag.add_edge((str(p), str(n)))
        
        self.num_steps = self.n_node - 1
        if hps.num_steps > 0:
            self.num_steps = min(self.num_steps, hps.num_steps)

    def next_candidates(self, mask):
        o = np.where(mask >= 0)[0]
        u = np.where(mask < 0)[0]
        if tuple(o) in self.cache: return self.cache[tuple(o)]
        candidates = []
        for ui in u:
            for i in range(self.hps.n_target):
                if not self.dag.is_dsep(str(ui), str(self.n_node-1-i), [str(oi) for oi in o]):
                    candidates.append(ui)
                    break
        self.cache[tuple(o)] = candidates

        return candidates

    def compute_cmi(self, x, candidates, mask):
        num_cand = len(candidates)
        if num_cand <= 1: return np.zeros(num_cand)
        N = self.hps.num_samples
        d = self.hps.dimension
        Nt = self.hps.n_target
        y = -np.ones([num_cand, Nt], dtype=np.float32)
        x = np.repeat(np.expand_dims(x, axis=0), num_cand, axis=0)
        bj = (mask >= 0).astype(np.float32)
        bj = np.repeat(np.expand_dims(bj, axis=0), num_cand, axis=0)
        mj = bj.copy()
        mj[np.arange(num_cand), candidates] = 1.
        sam_j = self.model.run(self.model.sam_j, 
            {self.model.x:x, self.model.y:y, self.model.b:bj, self.model.m:mj})
        sam_x = sam_j[:,:,:-Nt]
        sam_y = sam_j[:,:,-Nt:]
        sam_x = sam_x.reshape([num_cand*N, d])
        sam_y = sam_y.reshape([num_cand*N, Nt])
        bs = np.repeat(np.expand_dims(bj, axis=1), N, axis=1).reshape([num_cand*N, d])
        ms = np.repeat(np.expand_dims(mj, axis=1), N, axis=1).reshape([num_cand*N, d])
        log_ratio = self.model.run(self.model.log_ratio, 
            {self.model.x:sam_x, self.model.y:sam_y, self.model.b:bs, self.model.m:ms})
        log_ratio = log_ratio.reshape([num_cand,N])
        cmi = np.mean(log_ratio, axis=1)

        return cmi

    def predict(self, x, mask):
        d = x.shape[0]
        Nt = self.hps.n_target
        y = -np.ones([d, Nt], dtype=np.float32)
        t1 = np.zeros([d, 1])
        t2 = np.expand_dims(np.arange(d), axis=1)
        b = np.logical_and(mask >= t1, mask <= t2).astype(np.float32) # [d,d]
        x = np.repeat(np.expand_dims(x, axis=0), d, axis=0)
        pred = self.model.run(self.model.mean_y, 
            {self.model.x:x, self.model.y:y, self.model.b:b, self.model.m:b})

        return pred
    
    def _call(self, x):
        '''
        x: [d]
        '''
        d = x.shape[0]
        mask = np.ones([d], dtype=np.int32) * -1
        for step in range(self.num_steps):
            candidates = self.next_candidates(mask)
            if len(candidates) == 0: break
            cmi = self.compute_cmi(x, candidates, mask)
            sel_cand = candidates[np.argmax(cmi)]
            mask[sel_cand] = step
        pred = self.predict(x, mask)

        return mask, pred
    
    def __call__(self, x):
        '''
        x: [B, d]
        '''
        batch_size, d = x.shape
        assert d == self.n_node - self.hps.n_target
        masks = []
        preds = []
        for i in range(batch_size):
            mask, pred = self._call(x[i])
            masks.append(mask)
            preds.append(pred)
        masks = np.stack(masks, axis=0)
        preds = np.stack(preds, axis=0)

        return masks, preds