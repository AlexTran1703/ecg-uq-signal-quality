"""B1 Dirichlet leaf posterior and B2 Bayesian CART (Metropolis-Hastings over tree structures)."""
from __future__ import annotations

import math

import numpy as np
from joblib import Parallel, delayed
from scipy import special

from ..config import EPS, N_CLASSES


class _Node:
    __slots__ = ("idx", "depth", "feat", "t", "left", "right", "lm", "counts")

    def __init__(self, idx, depth, lm, counts):
        self.idx, self.depth, self.lm, self.counts = idx, depth, lm, counts
        self.feat = self.t = self.left = self.right = None

    @property
    def is_leaf(self):
        return self.left is None


class BayesCART:

    def __init__(self, n_classes: int = N_CLASSES, dirichlet_alpha: float = 1.0,
                 tree_alpha: float = 0.95, tree_beta: float = 2.0, max_depth: int = 12,
                 min_leaf: int = 10, n_thresholds: int = 15, seed: int = 0,
                 n_chains: int = 8):
        self.C = n_classes
        self.a0 = float(dirichlet_alpha) / n_classes
        self.tree_alpha, self.tree_beta = tree_alpha, tree_beta
        self.max_depth, self.min_leaf, self.n_thresholds = max_depth, min_leaf, n_thresholds
        self.seed = seed
        self.n_chains = n_chains


    def _ps(self, depth):
        return 0.0 if depth >= self.max_depth else min(0.999, self.tree_alpha / (1 + depth) ** self.tree_beta)

    def _leaf_lm(self, idx):
        c = np.bincount(self.y_[idx], minlength=self.C).astype(float)
        a = np.full(self.C, self.a0)
        return float(special.gammaln(a.sum()) - special.gammaln(a.sum() + c.sum())
                     + np.sum(special.gammaln(a + c) - special.gammaln(a))), c

    def _new(self, idx, depth):
        lm, c = self._leaf_lm(idx)
        return _Node(idx, depth, lm, c)

    def _dprior_grow(self, depth, j):
        ps, ps1 = self._ps(depth), self._ps(depth + 1)
        return (math.log(max(ps, EPS)) - math.log(self.d_) - math.log(self.nb_[j])
                - math.log(max(1 - ps, EPS)) + 2 * math.log(max(1 - ps1, EPS)))


    def _leaves(self, node, out):
        if node.is_leaf:
            out.append(node)
        else:
            self._leaves(node.left, out)
            self._leaves(node.right, out)
        return out

    def _growable(self, root):
        return [l for l in self._leaves(root, [])
                if l.depth < self.max_depth and len(l.idx) >= 2 * self.min_leaf]

    def _prunable(self, node, out):
        if not node.is_leaf:
            if node.left.is_leaf and node.right.is_leaf:
                out.append(node)
            self._prunable(node.left, out)
            self._prunable(node.right, out)
        return out

    def _valid_splits(self, node):
        sub = self.code_[node.idx]
        m = len(node.idx)
        js, ts = [], []
        for j in range(self.d_):
            cum = np.cumsum(np.bincount(sub[:, j], minlength=self.nb_[j] + 1))[: self.nb_[j]]
            ok = np.flatnonzero((cum >= self.min_leaf) & (m - cum >= self.min_leaf))
            if len(ok):
                js.append(np.full(len(ok), j))
                ts.append(ok)
        if not js:
            return np.empty(0, int), np.empty(0, int)
        return np.concatenate(js), np.concatenate(ts)


    def _internal(self, node, out):
        if not node.is_leaf:
            out.append(node)
            self._internal(node.left, out)
            self._internal(node.right, out)
        return out

    def _reroute(self, node, idx):
        node.idx = idx
        if node.is_leaf:
            if len(idx) < self.min_leaf:
                return None
            node.lm, node.counts = self._leaf_lm(idx)
            return node.lm
        go = self.code_[idx, node.feat] <= node.t
        a = self._reroute(node.left, idx[go])
        if a is None:
            return None
        b = self._reroute(node.right, idx[~go])
        return None if b is None else a + b

    def _subtree_lm(self, node):
        return node.lm if node.is_leaf else (self._subtree_lm(node.left)
                                             + self._subtree_lm(node.right))

    def _snapshot(self, node):
        return (node.feat, node.t, node.idx, node.lm, node.counts,
                None if node.is_leaf else (self._snapshot(node.left),
                                           self._snapshot(node.right)))

    def _restore(self, node, snap):
        node.feat, node.t, node.idx, node.lm, node.counts, kids = snap
        if kids is not None:
            self._restore(node.left, kids[0])
            self._restore(node.right, kids[1])

    def fit(self, X, y, n_iter: int = 2000, burn: int = 500, thin: int = 10,
            verbose: bool = False, n_jobs: int = 1):
        X = np.asarray(X, np.float64)
        self.y_ = np.asarray(y, np.int64)
        n, self.d_ = X.shape


        qs = np.linspace(0.05, 0.95, self.n_thresholds)
        self.thr_ = [np.unique(np.nanquantile(X[:, j], qs)) for j in range(self.d_)]
        self.thr_ = [t if len(t) else np.array([np.nanmedian(X[:, j])]) for j, t in enumerate(self.thr_)]
        self.nb_ = np.array([len(t) for t in self.thr_])
        self.code_ = np.empty((n, self.d_), np.int16)
        for j in range(self.d_):
            self.code_[:, j] = np.searchsorted(self.thr_[j], X[:, j], side="left")


        self.samples_, acc, tot = [], 0, 0
        self.chain_summary_ = []
        seeds = [self.seed + 7919 * c for c in range(self.n_chains)]
        if n_jobs == 1:
            results = [self._run_chain(n, n_iter, burn, thin, s) for s in seeds]
        else:


            results = Parallel(n_jobs=n_jobs)(
                delayed(self._run_chain)(n, n_iter, burn, thin, s) for s in seeds)
        for chain, (a, t, drawn) in enumerate(results):
            acc += a
            tot += t
            self.samples_.extend(drawn)
            sizes = [int(s["n_leaves"]) for s in drawn] or [0]
            depths = [int(s["depth"].max()) for s in drawn] or [0]
            self.chain_summary_.append({
                "chain": chain,
                "accept": a / max(1, t),
                "n_samples": len(drawn),
                "n_distinct": len({(tuple(s["feat"]), tuple(s["t"])) for s in drawn}),
                "leaves_mean": float(np.mean(sizes)),
                "leaves_max": int(np.max(sizes)),
                "depth_mean": float(np.mean(depths)),
                "depth_max": int(np.max(depths)),
            })
        self.accept_rate_ = acc / max(1, tot)
        if not self.samples_:
            self.samples_.append(self._compact(self._new(np.arange(n), 0)))
        if verbose:
            print(f"      CART: {self.n_chains} chains, accept {100*self.accept_rate_:.0f}%, "
                  f"{len(self.samples_)} trees, "
                  f"{len(set((tuple(s['feat']), tuple(s['t'])) for s in self.samples_))} distinct",
                  flush=True)
        return self

    def _run_chain(self, n, n_iter, burn, thin, seed):
        rng = np.random.default_rng(seed)
        root = self._new(np.arange(n), 0)
        accepted = 0
        chain_samples = []

        for it in range(n_iter):
            grow = self._growable(root)
            prune = self._prunable(root, [])
            change = self._internal(root, [])
            kinds = ([("grow",)] if grow else []) + ([("prune",)] if prune else []) \
                + ([("change",)] if change else [])
            n_moves = len(kinds)
            if n_moves == 0:
                break
            move = kinds[rng.integers(n_moves)][0]

            if move == "change":


                node = change[rng.integers(len(change))]
                js, ts = self._valid_splits(node)
                if not len(js):
                    continue
                pick = rng.integers(len(js))
                j, t = int(js[pick]), int(ts[pick])
                if j == node.feat and t == node.t:
                    continue
                snap = self._snapshot(node)
                old_lm = self._subtree_lm(node)
                d_pri = math.log(self.nb_[node.feat]) - math.log(self.nb_[j])
                node.feat, node.t = j, t
                new_lm = self._reroute(node, node.idx)
                if new_lm is not None and \
                        math.log(rng.random()) < min(0.0, new_lm - old_lm + d_pri):
                    accepted += 1
                else:
                    self._restore(node, snap)

            elif move == "grow":
                leaf = grow[rng.integers(len(grow))]
                js, ts = self._valid_splits(leaf)
                if not len(js):
                    continue
                pick = rng.integers(len(js))
                j, t = int(js[pick]), int(ts[pick])
                go_left = self.code_[leaf.idx, j] <= t
                left = self._new(leaf.idx[go_left], leaf.depth + 1)
                right = self._new(leaf.idx[~go_left], leaf.depth + 1)
                d_lik = left.lm + right.lm - leaf.lm
                d_pri = self._dprior_grow(leaf.depth, j)

                leaf.feat, leaf.t, leaf.left, leaf.right = j, t, left, right

                n_moves2 = (1 if self._growable(root) else 0) + 2
                log_q = (-math.log(n_moves2) - math.log(len(self._prunable(root, []))))
                log_q -= (-math.log(n_moves) - math.log(len(grow)) - math.log(len(js)))

                if math.log(rng.random()) < min(0.0, d_lik + d_pri + log_q):
                    accepted += 1
                else:
                    leaf.feat = leaf.t = leaf.left = leaf.right = None
            else:
                node = prune[rng.integers(len(prune))]
                j, t, left, right = node.feat, node.t, node.left, node.right
                d_lik = node.lm - left.lm - right.lm
                d_pri = -self._dprior_grow(node.depth, j)

                node.feat = node.t = node.left = node.right = None
                grow2 = self._growable(root)
                js2, _ = self._valid_splits(node)
                n_moves2 = 1 + (1 if self._prunable(root, []) else 0) \
                    + (1 if self._internal(root, []) else 0)
                if len(js2) and len(grow2):
                    log_q = (-math.log(n_moves2) - math.log(len(grow2)) - math.log(len(js2)))
                    log_q -= (-math.log(n_moves) - math.log(len(prune)))
                    accept = math.log(rng.random()) < min(0.0, d_lik + d_pri + log_q)
                else:
                    accept = False
                if accept:
                    accepted += 1
                else:
                    node.feat, node.t, node.left, node.right = j, t, left, right

            if it >= burn and (it - burn) % thin == 0:
                chain_samples.append(self._compact(root))

        return accepted, n_iter, chain_samples

    def _compact(self, root):
        feat, thr, left, right, cnt, depth = [], [], [], [], [], []

        def rec(node):
            i = len(feat)
            feat.append(-1); thr.append(0.0); left.append(-1); right.append(-1)
            cnt.append(node.counts); depth.append(node.depth)
            if not node.is_leaf:
                li = rec(node.left)
                ri = rec(node.right)
                feat[i], thr[i], left[i], right[i] = node.feat, node.t, li, ri
            return i

        rec(root)
        return {"feat": np.array(feat, np.int32), "t": np.array(thr, np.int32),
                "left": np.array(left, np.int32), "right": np.array(right, np.int32),
                "counts": np.array(cnt, np.float64), "depth": np.array(depth, np.int32),
                "n_leaves": int((np.array(feat) < 0).sum())}

    def _dirichlet_of(self, tree, code):
        a = np.empty((len(code), self.C))
        stack = [(0, np.arange(len(code)))]
        while stack:
            node, rows = stack.pop()
            if tree["feat"][node] < 0 or not len(rows):
                if len(rows):
                    a[rows] = tree["counts"][node] + self.a0
                continue
            go = code[rows, tree["feat"][node]] <= tree["t"][node]
            stack.append((tree["left"][node], rows[go]))
            stack.append((tree["right"][node], rows[~go]))
        return a

    def _code_of(self, X):
        X = np.asarray(X, np.float64)
        code = np.empty(X.shape, np.int16)
        for j in range(self.d_):
            code[:, j] = np.searchsorted(self.thr_[j], X[:, j], side="left")
        return code

    def predict_members(self, X, n_samples: int = 100, seed: int = 0):
        rng = np.random.default_rng(seed)
        code = self._code_of(X)
        pick = rng.integers(0, len(self.samples_), size=n_samples)
        out = np.empty((n_samples, len(X), self.C), np.float32)
        cache = {}
        for m, j in enumerate(pick):
            if j not in cache:
                cache[j] = self._dirichlet_of(self.samples_[j], code)
            g = rng.standard_gamma(cache[j])
            out[m] = (g / np.clip(g.sum(1, keepdims=True), EPS, None)).astype(np.float32)
        return out


def dirichlet_leaf_members(tree, Ztr, ytr, Zte, n_samples: int, seed: int,
                           dirichlet_alpha: float = 1.0) -> np.ndarray:
    a0 = float(dirichlet_alpha) / N_CLASSES
    leaf_tr, leaf_te = tree.apply(Ztr), tree.apply(Zte)
    leaves, inv = np.unique(leaf_te, return_inverse=True)

    counts = np.zeros((len(leaves), N_CLASSES), np.float64)
    order = np.argsort(leaf_tr)
    sorted_leaf, sorted_y = leaf_tr[order], np.asarray(ytr)[order]
    lo = np.searchsorted(sorted_leaf, leaves, "left")
    hi = np.searchsorted(sorted_leaf, leaves, "right")
    for i, (a, b) in enumerate(zip(lo, hi)):
        if b > a:
            counts[i] = np.bincount(sorted_y[a:b], minlength=N_CLASSES)

    post = counts + a0
    rng = np.random.default_rng(seed)
    out = np.empty((n_samples, len(Zte), N_CLASSES), np.float32)
    for m in range(n_samples):
        g = rng.standard_gamma(post)
        pi = g / np.clip(g.sum(1, keepdims=True), EPS, None)
        out[m] = pi[inv].astype(np.float32)
    return out
