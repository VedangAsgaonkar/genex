import numpy as np
import tensorflow as tf

class SFA(object):
    def __init__(self, hps, model):
        self.hps = hps
        self.model = model

        self.num_steps = hps.dimension
        if hps.num_steps > 0:
            self.num_steps = min(self.num_steps, hps.num_steps)

    def _batch_CMI(self, x, b, candidates):
        B, d = x.shape
        Nt = self.hps.n_target
        N = self.hps.num_samples
        y = -np.ones([B, Nt], dtype=np.float32)
        cmi_list = []
        for c in candidates:
            m = b.copy()
            m[:,c] = 1.
            sam_j = self.model.run(self.model.sam_j, 
                {self.model.x:x, self.model.y:y, self.model.b:b, self.model.m:m})
            sam_x = sam_j[:,:,:-Nt]
            sam_y = sam_j[:,:,-Nt:]
            sam_x = sam_x.reshape([B*N, d])
            sam_y = sam_y.reshape([B*N, Nt])
            b_tile = np.repeat(np.expand_dims(b, axis=1), N, axis=1).reshape([B*N, d])
            m_tile = np.repeat(np.expand_dims(m, axis=1), N, axis=1).reshape([B*N, d])
            # compute cmi
            log_ratio = self.model.run(self.model.log_ratio, 
                {self.model.x:sam_x, self.model.y:sam_y, self.model.b:b_tile, self.model.m:m_tile})
            log_ratio = log_ratio.reshape([B,N])
            cmi = np.mean(log_ratio, axis=1)
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
        Nt = self.hps.n_target
        masks = np.expand_dims(masks, axis=1) # [B,1,d]
        t1 = np.zeros([d, 1])
        t2 = np.expand_dims(np.arange(d), axis=1)
        b = np.logical_and(masks >= t1, masks <= t2).astype(np.float32) # [B,d,d]
        preds = []
        for n in range(B//self.hps.batch_size):
            batch_x = x[n*self.hps.batch_size:(n+1)*self.hps.batch_size]
            batch_x = np.repeat(np.expand_dims(batch_x, axis=1), d, axis=1)
            batch_x = np.reshape(batch_x, [self.hps.batch_size*d, d])
            batch_y = -np.ones([batch_x.shape[0], Nt], dtype=np.float32)
            batch_b = b[n*self.hps.batch_size:(n+1)*self.hps.batch_size]
            batch_b = np.reshape(batch_b, [self.hps.batch_size*d, d])
            pred = self.model.run(self.model.mean_y, 
                {self.model.x:batch_x, self.model.y:batch_y, self.model.b:batch_b, self.model.m:batch_b})
            pred = np.reshape(pred, [self.hps.batch_size, d, Nt])
            preds.append(pred)
        preds = np.concatenate(preds, axis=0)

        return preds

    def __call__(self, x):
        B, d = x.shape
        Nt = self.hps.n_target
        masks = np.ones([B, d], dtype=np.int32) * -1
        for step in range(self.num_steps):
            b = (masks >= 0).astype(np.float32)
            m = self._next(x, b)
            q = m * (1-b)
            masks = masks * (1-q) + np.ones_like(masks) * step * q
        preds = self.predict(x, masks)

        return masks, preds