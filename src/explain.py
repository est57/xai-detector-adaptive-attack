import torch


def integrated_gradients(model, x, steps, target=1, create_graph=False):
    """Atribusi Integrated Gradients terhadap kelas target dengan baseline nol.

    create_graph=True membuat atribusi dapat diturunkan lagi (dibutuhkan serangan adaptif).
    """
    if not x.requires_grad:
        x = x.detach().requires_grad_(True)
    baseline = torch.zeros_like(x)
    alphas = torch.linspace(1.0 / steps, 1.0, steps, device=x.device)

    total_grad = torch.zeros_like(x)
    for alpha in alphas:
        point = baseline + alpha * (x - baseline)
        logit = model(point)[:, target].sum()
        grad, = torch.autograd.grad(logit, point, create_graph=create_graph)
        total_grad = total_grad + grad

    return (x - baseline) * total_grad / steps


def explain_batches(model, x, steps, batch_size=2048):
    """Menghitung atribusi untuk data besar secara bertahap (tanpa graf gradien)."""
    x = torch.as_tensor(x, dtype=torch.float32)
    out = [integrated_gradients(model, x[i:i + batch_size], steps).detach()
           for i in range(0, len(x), batch_size)]
    return torch.cat(out)
