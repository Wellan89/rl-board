import numpy as np


class nph:
    @staticmethod
    def matmul_single(a, b):
        return np.einsum('i,ij->j', a, b)

    @staticmethod
    def matmul(a, b):
        return np.einsum('ij,jk->ik', a, b)

    @staticmethod
    def one_hot(x, depth):
        return np.eye(depth)[x]

    @staticmethod
    def flatten_all_but_0(x):
        return np.reshape(x, [-1, np.prod(x.shape[1:])])

    @staticmethod
    def leaky_relu(x, alpha):
        return 0.5 * ((1.0 + alpha) * x + (1.0 - alpha) * abs(x))

    @classmethod
    def dense(cls, x, weights, name):
        return cls.matmul(x, weights['{}/kernel:0'.format(name)]) + weights['{}/bias:0'.format(name)]

    @staticmethod
    def _calc_pad(pad, in_siz, out_siz, stride, ksize):
        if pad == 'SAME':
            return (out_siz - 1) * stride + ksize - in_siz
        elif pad == 'VALID':
            return 0
        else:
            return pad

    @staticmethod
    def _calc_size(h, kh, pad, sh):
        if pad == 'VALID':
            return np.ceil((h - kh + 1) / sh)
        elif pad == 'SAME':
            return np.ceil(h / sh)
        else:
            return int(np.ceil((h - kh + pad + 1) / sh))

    @classmethod
    def _extract_sliding_windows(cls, x, ksize, pad, stride, floor_first=True):
        n = x.shape[0]
        h = x.shape[1]
        w = x.shape[2]
        c = x.shape[3]
        kh = ksize[0]
        kw = ksize[1]
        sh = stride[0]
        sw = stride[1]

        h2 = int(cls._calc_size(h, kh, pad, sh))
        w2 = int(cls._calc_size(w, kw, pad, sw))
        ph = int(cls._calc_pad(pad, h, h2, sh, kh))
        pw = int(cls._calc_pad(pad, w, w2, sw, kw))

        ph0 = int(np.floor(ph / 2))
        ph1 = int(np.ceil(ph / 2))
        pw0 = int(np.floor(pw / 2))
        pw1 = int(np.ceil(pw / 2))

        if floor_first:
            pph = (ph0, ph1)
            ppw = (pw0, pw1)
        else:
            pph = (ph1, ph0)
            ppw = (pw1, pw0)
        x = np.pad(
            x, ((0, 0), pph, ppw, (0, 0)),
            mode='constant',
            constant_values=(0.0, ))

        y = np.zeros([n, h2, w2, kh, kw, c])
        for ii in range(h2):
            for jj in range(w2):
                xx = ii * sh
                yy = jj * sw
                y[:, ii, jj, :, :, :] = x[:, xx:xx + kh, yy:yy + kw, :]
        return y

    @classmethod
    def _conv2d(cls, x, w, pad='SAME', stride=(1, 1)):
        ksize = w.shape[:2]
        x = cls._extract_sliding_windows(x, ksize, pad, stride)
        ws = w.shape
        w = w.reshape([ws[0] * ws[1] * ws[2], ws[3]])
        xs = x.shape
        x = x.reshape([xs[0] * xs[1] * xs[2], -1])
        y = x.dot(w)
        y = y.reshape([xs[0], xs[1], xs[2], -1])
        return y

    @classmethod
    def conv2d(cls, x, weights, name, pad):
        return cls._conv2d(x, weights['{}/W:0'.format(name)], pad=pad) + weights['{}/b:0'.format(name)]
