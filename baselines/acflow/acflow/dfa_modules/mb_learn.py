import numpy as np
import tensorflow as tf
import logging

class MB(object):
    def __init__(self, hps, model):
        self.hps = hps
        self.model = model

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

    def _CMI(self, x, y, i):
        B = self.hps.batch_size
        batch_size, d = x.shape
        Nt = y.shape[1]
        num_batches = batch_size // B
        m1 = np.ones([d+Nt, d+Nt], dtype=np.float32)
        b1 = np.ones([d+Nt, d+Nt], dtype=np.float32)
        b1[:, i] = 0.
        m2 = 1 - np.eye(d+Nt, dtype=np.float32)
        b2 = m2.copy()
        b2[:, i] = 0.
        b = np.concatenate([b1, b2], axis=0)
        m = np.concatenate([m1, m2], axis=0)
        # run
        cmi_est = []
        for n in range(num_batches):
            xx = x[n*B:(n+1)*B]
            yy = y[n*B:(n+1)*B]
            cmi = self._batch_CMI(xx, yy, b, m)
            cmi_est.append(cmi)
        cmi_est = np.concatenate(cmi_est, axis=0)

        return np.mean(cmi_est, axis=0)

    def __call__(self, x, y):
        batch_size, d = x.shape
        Nt = y.shape[1]
        cmi = np.zeros([d+Nt, d+Nt], dtype=np.float32)
        for i in range(d+Nt):
            cmi_i = self._CMI(x, y, i)
            cmi[i] = cmi_i.copy()
        cmi = (cmi + cmi.T) / 2
        logging.info('cmi:')
        logging.info(cmi)
        # normalize
        if self.hps.normalize:
            ent = np.diag(cmi)
            norm = np.sqrt(np.outer(ent, ent))
            cmi /= norm
            logging.info('cnmi:')
            logging.info(cmi)
        # markov blanket
        mb = cmi > self.hps.cmi_thresh
        mb[np.eye(d+Nt, dtype=np.bool)] = 0
        
        return cmi, mb