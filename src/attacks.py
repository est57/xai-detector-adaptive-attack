import torch
import torch.nn.functional as F

from src.explain import integrated_gradients

BENIGN = 0
CLEAN = 0


def make_success_fn(ids_model, detector=None, threshold=None):
    """Kriteria keberhasilan: lolos IDS (non-adaptif) atau lolos IDS dan detektor (adaptif)."""

    def success(x):
        with torch.no_grad():
            evade = torch.softmax(ids_model(x), dim=1)[:, 1] < 0.5
        if detector is None:
            return evade
        det_score = detector.margin(detector(x)).detach()
        return evade & (det_score < threshold)

    return success


def constrained_pgd(ids_model, x, constraints, eps, steps, success_fn, detector=None, lam=1.0):
    """PGD dengan batasan domain dan pemilihan iterasi terbaik (best-iterate).

    detector=None  -> serangan non-adaptif (hanya mengecoh IDS).
    detector!=None -> serangan adaptif (mengecoh IDS sekaligus detektor berbasis penjelasan).
    """
    step_size = 2.5 * eps / steps
    target = torch.full((len(x),), BENIGN, dtype=torch.long)
    x_adv = x.clone()
    x_best = constraints.finalize(x_adv, x, eps)
    done = success_fn(x_best)

    for _ in range(steps):
        active = ~done
        if not active.any():
            break
        x_adv.requires_grad_(True)
        loss = F.cross_entropy(ids_model(x_adv), target)
        if detector is not None:
            det_logits = detector(x_adv, create_graph=True)
            loss = loss + lam * F.cross_entropy(det_logits, torch.full_like(target, CLEAN))
        grad, = torch.autograd.grad(loss, x_adv)
        x_adv = x_adv.detach() - step_size * grad.sign()
        x_adv = constraints.project(x_adv, x, eps)

        candidate = constraints.finalize(x_adv, x, eps)
        x_best = torch.where(active.unsqueeze(1), candidate, x_best)
        done = done | (success_fn(candidate) & active)

    return x_best, done


def run_attack(ids_model, x, constraints, eps, steps, success_fn, detector=None, lam=1.0, batch_size=512):
    """Menjalankan serangan secara bertahap per batch."""
    x = torch.as_tensor(x, dtype=torch.float32)
    advs, dones = [], []
    for i in range(0, len(x), batch_size):
        x_adv, done = constrained_pgd(ids_model, x[i:i + batch_size], constraints, eps, steps,
                                      success_fn, detector, lam)
        advs.append(x_adv)
        dones.append(done)
    return torch.cat(advs), torch.cat(dones)


def adaptive_attack(ids_model, detector, x, constraints, eps, steps, success_fn, lambdas):
    """Serangan adaptif dengan beberapa nilai lambda; per sampel diambil hasil yang berhasil."""
    x = torch.as_tensor(x, dtype=torch.float32)
    x_best, done = None, torch.zeros(len(x), dtype=torch.bool)
    per_lambda = {}
    for lam in lambdas:
        x_adv, success = run_attack(ids_model, x, constraints, eps, steps, success_fn, detector, lam)
        per_lambda[lam] = float(success.float().mean())
        if x_best is None:
            x_best = x_adv
        else:
            x_best = torch.where((success & ~done).unsqueeze(1), x_adv, x_best)
        done = done | success
    return x_best, done, per_lambda


class AttributionTarget:
    """Pusat atribusi trafik benign; dipakai serangan bergaya ASA-IDS (tanpa pengetahuan detektor)."""

    def __init__(self, ids_model, benign_attr, ig_steps):
        self.ids_model = ids_model
        self.ig_steps = ig_steps
        self.centroid = benign_attr.mean(dim=0)
        self.scale = benign_attr.std(dim=0).clamp(min=1e-6)

    def distance(self, x, create_graph=False):
        """Jarak kuadrat rata-rata (ternormalisasi) antara atribusi sampel dan pusat atribusi benign."""
        attr = integrated_gradients(self.ids_model, x, self.ig_steps, create_graph=create_graph)
        return (((attr - self.centroid) / self.scale) ** 2).mean(dim=1)


def stealth_pgd(ids_model, x, constraints, eps, steps, target, mu):
    """PGD bergaya ASA-IDS: lolos IDS sambil menarik atribusi ke pusat atribusi benign.

    Penyerang tidak mengetahui detektor; per sampel dipilih iterasi yang lolos IDS dengan jarak atribusi terkecil.
    """
    step_size = 2.5 * eps / steps
    label = torch.full((len(x),), BENIGN, dtype=torch.long)
    x_adv = x.clone()
    x_best = constraints.finalize(x_adv, x, eps)
    best_dist = torch.full((len(x),), float("inf"))
    evaded = torch.zeros(len(x), dtype=torch.bool)

    for _ in range(steps):
        x_adv.requires_grad_(True)
        loss = F.cross_entropy(ids_model(x_adv), label) + mu * target.distance(x_adv, create_graph=True).mean()
        grad, = torch.autograd.grad(loss, x_adv)
        x_adv = constraints.project(x_adv.detach() - step_size * grad.sign(), x, eps)

        candidate = constraints.finalize(x_adv, x, eps)
        with torch.no_grad():
            evade = torch.softmax(ids_model(candidate), dim=1)[:, 1] < 0.5
        dist = target.distance(candidate).detach()
        better = evade & (dist < best_dist)
        x_best = torch.where(better.unsqueeze(1) | (~evaded & ~evade).unsqueeze(1), candidate, x_best)
        best_dist = torch.where(better, dist, best_dist)
        evaded = evaded | evade

    return x_best, evaded, best_dist


def stealth_attack(ids_model, x, constraints, eps, steps, target, mus, batch_size=512):
    """Serangan bergaya ASA-IDS untuk beberapa nilai mu; per sampel diambil hasil lolos IDS dengan atribusi paling mirip benign."""
    x = torch.as_tensor(x, dtype=torch.float32)
    x_best, best_dist = x.clone(), torch.full((len(x),), float("inf"))
    evaded = torch.zeros(len(x), dtype=torch.bool)
    per_mu = {}
    for mu in mus:
        parts = [stealth_pgd(ids_model, x[i:i + batch_size], constraints, eps, steps, target, mu)
                 for i in range(0, len(x), batch_size)]
        x_mu = torch.cat([p[0] for p in parts])
        ev_mu = torch.cat([p[1] for p in parts])
        d_mu = torch.cat([p[2] for p in parts])
        per_mu[mu] = float(ev_mu.float().mean())
        take = (ev_mu & (d_mu < best_dist)) | (~evaded & ~ev_mu)
        x_best = torch.where(take.unsqueeze(1), x_mu, x_best)
        best_dist = torch.where(ev_mu & (d_mu < best_dist), d_mu, best_dist)
        evaded = evaded | ev_mu
    return x_best, evaded, per_mu
