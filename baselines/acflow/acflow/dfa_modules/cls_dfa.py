import numpy as np
import tensorflow as tf
from scipy.special import softmax
from scipy.stats import entropy

class DFA(object):
    def __init__(self, hps, model):
        '''
        hps: hyperparameters
        model: p(x_i, y | x_o)
        '''
        self.hps = hps
        self.model = model

        self.num_steps = hps.dimension
        if hps.num_steps > 0:
            self.num_steps = min(self.num_steps, hps.num_steps)

    def next_candidates(self, b):
        num = np.sum(b[0] == 0)
        for i in range(num):
            m = b.copy()
            for mi in m:
                d = np.where(mi == 0)[0]
                mi[d[i]] = 1
            yield m
    
    def _next(self, x, b):
        B = x.shape[0]
        N = self.hps.num_samples
        d = self.hps.dimension
        C = self.hps.n_classes

        best_cmi = -np.ones([B]) * np.inf
        best_mask = np.zeros_like(b)
        # compute p (y | x_o)
        pre_prob = self.model.run(self.model.prob,
            {self.model.x:x, self.model.b:b, self.model.m:b})
        pre_prob = np.repeat(np.expand_dims(pre_prob, axis=1), N, axis=1)
        pre_prob = np.transpose(pre_prob, [2,0,1])
        # sample p(x_u | x_o)
        m = np.ones_like(b)
        sam = self.model.run(self.model.sam,
            {self.model.x:x, self.model.b:b, self.model.m:m})
        # select
        for m in self.next_candidates(b):
            q = np.expand_dims(m*(1-b), axis=1)
            x_tile = sam.copy() * q + np.expand_dims(x, axis=1) * (1-q)
            x_tile = x_tile.reshape([B*N, d])
            b_tile = np.repeat(np.expand_dims(b, axis=1), N, axis=1).reshape([B*N, d])
            m_tile = np.repeat(np.expand_dims(m, axis=1), N, axis=1).reshape([B*N, d])
            post_prob = self.model.run(self.model.prob, 
                {self.model.x:x_tile, self.model.b:b_tile, self.model.m:m_tile})
            post_prob = post_prob.reshape([B,N,C])
            post_prob = np.transpose(post_prob, [2,0,1])
            # compute cmi
            cmi = entropy(post_prob, pre_prob)
            cmi = np.mean(cmi, axis=1)
            # compare
            ind = cmi > best_cmi
            best_mask[ind] = m[ind].copy()
            best_cmi[ind] = cmi[ind].copy()

        return best_mask

    def __call__(self, x):
        '''
        x: [B,d]
        '''
        batch_size, d = x.shape
        masks = np.ones([batch_size, d], dtype=np.int32) * -1
        preds = []
        for step in range(self.num_steps):
            b = (masks >= 0).astype(np.float32)
            m = self._next(x, b)
            pred = self.model.run(self.model.pred, 
                {self.model.x:x, self.model.b:b, self.model.m:m})
            preds.append(pred)
            q = m * (1-b)
            masks = masks * (1-q) + np.ones_like(masks) * step * q
        preds = np.stack(preds, axis=1)

        return masks, preds

