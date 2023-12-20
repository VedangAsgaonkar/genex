import numpy as np
import tensorflow as tf
from scipy.special import softmax
from scipy.stats import entropy

class SFA(object):
    def __init__(self, hps, model):
        self.hps = hps
        self.model = model

        self.num_steps = hps.dimension
        if hps.num_steps > 0:
            self.num_steps = min(self.num_steps, hps.num_steps)
    
    def _batch_CMI(self, x, b, candidates):
        B,d = x.shape
        N = self.hps.num_samples
        C = self.hps.n_classes

        # compute p (y | x_o)
        pre_prob = self.model.run(self.model.prob,
            {self.model.x:x, self.model.b:b, self.model.m:b})
        pre_prob = np.repeat(np.expand_dims(pre_prob, axis=1), N, axis=1)
        pre_prob = np.transpose(pre_prob, [2,0,1])
        # sample p(x_u | x_o)
        m = np.ones_like(b)
        sam = self.model.run(self.model.sam,
            {self.model.x:x, self.model.b:b, self.model.m:m})
        # compute cmi
        cmi_list = []
        for c in candidates:
            b_tile = np.repeat(np.expand_dims(b, axis=1), N, axis=1).reshape([B*N, d])
            m_tile = b_tile.copy()
            m_tile[:,c] = 1.
            q_tile = m_tile * (1-b_tile)
            x_tile = np.repeat(np.expand_dims(x, axis=1), N, axis=1).reshape([B*N, d])
            x_tile = sam.reshape([B*N,d]) *  q_tile + x_tile * (1-q_tile)
            # compute p(y | x_o, x_i)
            post_prob = self.model.run(self.model.prob, 
                {self.model.x:x_tile, self.model.b:b_tile, self.model.m:m_tile})
            post_prob = post_prob.reshape([B,N,C])
            post_prob = np.transpose(post_prob, [2,0,1])
            cmi = entropy(post_prob, pre_prob)
            cmi = np.mean(cmi, axis=1)
            cmi_list.append(cmi)
        
        return np.stack(cmi_list, axis=1)

    def _CMI(self, x, b, candidates):
        B = x.shape[0]
        cmi = []
        for n in range(B//self.hps.batch_size):
            batch_x = x[n*self.hps.batch_size:(n+1)*self.hps.batch_size]
            batch_b = b[n*self.hps.batch_size:(n+1)*self.hps.batch_size]
            batch_cmi = self._batch_CMI(batch_x, batch_b, candidates)
            cmi.append(batch_cmi)
        cmi = np.concatenate(cmi, axis=0)
        cmi = np.mean(cmi, axis=0)

        return cmi

    def next_candidates(self, b):
        return np.where(b[0] == 0)[0]
    
    def _next(self, x, b):
        B,d = x.shape
        candidates = self.next_candidates(b)
        cmi = self._CMI(x, b, candidates)
        best_idx = candidates[np.argmax(cmi)]
        m = b.copy()
        m[:,best_idx] = 1.

        return m

    def predict(self, x, masks):
        B, d = x.shape
        masks = np.expand_dims(masks, axis=1) # [B,1,d]
        t1 = np.zeros([d, 1])
        t2 = np.expand_dims(np.arange(d), axis=1)
        b = np.logical_and(masks >= t1, masks <= t2).astype(np.float32) # [B,d,d]
        preds = []
        for n in range(B//self.hps.batch_size):
            batch_x = x[n*self.hps.batch_size:(n+1)*self.hps.batch_size]
            batch_x = np.repeat(np.expand_dims(batch_x, axis=1), d, axis=1)
            batch_x = np.reshape(batch_x, [self.hps.batch_size*d, d])
            batch_b = b[n*self.hps.batch_size:(n+1)*self.hps.batch_size]
            batch_b = np.reshape(batch_b, [self.hps.batch_size*d, d])
            pred = self.model.run(self.model.pred,
                {self.model.x:batch_x, self.model.b:batch_b, self.model.m:batch_b})
            pred = np.reshape(pred, [self.hps.batch_size, d])
            preds.append(pred)
        preds = np.concatenate(preds, axis=0)

        return preds

    def __call__(self, x):
        '''
        x: [B,d]
        '''
        B, d = x.shape
        masks = np.ones([B, d], dtype=np.int32) * -1
        for step in range(self.num_steps):
            b = (masks >= 0).astype(np.float32)
            m = self._next(x, b)
            q = m * (1-b)
            masks = masks * (1-q) + np.ones_like(masks) * step * q
        preds = self.predict(x, masks)

        return masks, preds
