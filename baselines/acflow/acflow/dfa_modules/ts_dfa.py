import numpy as np
import tensorflow as tf
from scipy.special import softmax
from scipy.stats import entropy

class DFA(object):
    def __init__(self, hps, model):
        self.hps = hps
        self.model = model

        self.dim = self.hps.dimension // self.hps.time_steps
        self.num_steps = self.hps.time_steps
        if hps.num_steps > 0:
            self.num_steps = min(self.num_steps, hps.num_steps)

    def next_candidates(self, mask):
        o = np.where(mask >= 0)[0]
        max_t = np.max(o) // self.dim if len(o) > 0 else -1

        return list(range(max_t+1, self.hps.time_steps))
    
    def compute_cmi(self, x, candidates, mask):
        num_cand = len(candidates)
        if num_cand <= 1: return np.zeros(num_cand)
        N = self.hps.num_samples
        T = self.hps.time_steps
        d = self.hps.dimension
        C = self.hps.n_classes

        # compute p(y | x_o) & sample x_i
        x = np.expand_dims(x, axis=0)
        b = (mask >= 0).astype(np.float32)
        b = np.expand_dims(b, axis=0)
        m = np.ones_like(b)
        logpo, sam = self.model.run([self.model.logpo, self.model.sam],
            {self.model.x:x, self.model.b:b, self.model.m:m})
        pre_prob = softmax(logpo, axis=1) # [1,C]
        pre_prob = np.repeat(pre_prob, N, axis=0) # [N,C]
        pre_prob = np.repeat(np.expand_dims(pre_prob, axis=0), num_cand, axis=0)
        pre_prob = np.transpose(pre_prob, [2,0,1])
        
        # compute p(y | x_i, x_o)
        x = np.repeat(sam, num_cand, axis=0) # [B,N,d]
        x = np.reshape(x, [num_cand*N, d])
        b = np.repeat(b, num_cand*N, axis=0) # [B*N,d]
        m = b.copy()
        m = np.reshape(m, [num_cand, N, T, self.dim])
        m[np.arange(num_cand),:,candidates,:] = 1.
        m = np.reshape(m, [num_cand*N, d])
        post_prob = self.model.run(self.model.prob,
            {self.model.x:x, self.model.b:b, self.model.m:m})
        post_prob = post_prob.reshape([num_cand,N,C])
        post_prob = np.transpose(post_prob, [2,0,1])
        
        # compute cmi
        cmi = entropy(post_prob, pre_prob)
        cmi = np.mean(cmi, axis=1)

        return cmi
    
    def predict(self, x, mask):
        t1 = np.zeros([self.hps.time_steps, 1])
        t2 = np.expand_dims(np.arange(self.hps.time_steps), axis=1)
        b = np.logical_and(mask >= t1, mask <= t2).astype(np.float32) # [t,d]
        x = np.repeat(np.expand_dims(x, axis=0), self.hps.time_steps, axis=0)
        pred = self.model.run(self.model.pred, 
                {self.model.x:x, self.model.b:b, self.model.m:b})

        return pred
    
    def _call(self, x):
        '''
        x: [d]
        '''
        mask = np.ones(*x.shape, dtype=np.int32) * -1
        for step in range(self.num_steps):
            candidates = self.next_candidates(mask)
            if len(candidates) == 0: break
            cmi = self.compute_cmi(x, candidates, mask)
            alpha0 = self.hps.alpha * (self.hps.time_steps - np.array(candidates))
            observations = np.random.multinomial(len(candidates)*self.hps.num_samples, softmax(cmi))
            alpha1 = alpha0 + observations
            p = np.random.dirichlet(alpha1)
            sel_cand = candidates[np.argmax(p)]
            
            mask[sel_cand*self.dim:(sel_cand+1)*self.dim] = step
        pred = self.predict(x, mask)

        return mask, pred

    def __call__(self, x):
        '''
        x: [B, d]
        '''
        batch_size, d = x.shape
        masks = []
        preds = []
        for i in range(batch_size):
            mask, pred = self._call(x[i])
            masks.append(mask)
            preds.append(pred)
        masks = np.stack(masks, axis=0)
        preds = np.stack(preds, axis=0)

        return masks, preds