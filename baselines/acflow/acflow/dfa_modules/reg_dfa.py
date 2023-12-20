import numpy as np
import tensorflow as tf

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
        Nt = self.hps.n_target
        y = -np.ones([B, Nt], dtype=np.float32)

        best_cmi = -np.ones([B]) * np.inf
        best_mask = np.zeros_like(b)
        for m in self.next_candidates(b):
            sam_j = self.model.run(self.model.sam_j, 
                {self.model.x:x, self.model.y:y, self.model.b:b, self.model.m:m})
            sam_x = sam_j[:,:,:-Nt]
            sam_y = sam_j[:,:,-Nt:]
            q = np.expand_dims(m*(1-b), axis=1)
            sam_x = sam_x * q + np.expand_dims(x, axis=1) * (1-q)
            sam_x = sam_x.reshape([B*N, d])
            sam_y = sam_y.reshape([B*N, Nt])
            b_tile = np.repeat(np.expand_dims(b, axis=1), N, axis=1).reshape([B*N, d])
            m_tile = np.repeat(np.expand_dims(m, axis=1), N, axis=1).reshape([B*N, d])
            # compute cmi
            log_ratio = self.model.run(self.model.log_ratio, 
                {self.model.x:sam_x, self.model.y:sam_y, self.model.b:b_tile, self.model.m:m_tile})
            log_ratio = log_ratio.reshape([B,N])
            cmi = np.mean(log_ratio, axis=1)
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
        Nt = self.hps.n_target
        y = -np.ones([batch_size, Nt], dtype=np.float32)
        masks = np.ones([batch_size, d], dtype=np.int32) * -1
        preds = []
        for step in range(self.num_steps):
            b = (masks >= 0).astype(np.float32)
            m = self._next(x, b)
            pred = self.model.run(self.model.mean_y, 
                {self.model.x:x, self.model.y:y, self.model.b:b, self.model.m:m})
            preds.append(pred)
            q = m * (1-b)
            masks = masks * (1-q) + np.ones_like(masks) * step * q
        preds = np.stack(preds, axis=1)

        return masks, preds