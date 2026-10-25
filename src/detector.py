import torch
from torch import nn

from src.explain import explain_batches, integrated_gradients
from src.models import MLP, train_classifier


class ExplanationDetector(nn.Module):
    """Detektor adversarial: input -> atribusi IG dari model IDS -> classifier (0 = bersih, 1 = adversarial)."""

    def __init__(self, ids_model, in_dim, hidden, ig_steps):
        super().__init__()
        self.ids_model = ids_model
        self.ig_steps = ig_steps
        self.classifier = MLP(in_dim, hidden)
        self.register_buffer("mean", torch.zeros(in_dim))
        self.register_buffer("std", torch.ones(in_dim))

    def normalize(self, attr):
        return (attr - self.mean) / self.std

    def forward(self, x, create_graph=False):
        attr = integrated_gradients(self.ids_model, x, self.ig_steps, create_graph=create_graph)
        return self.classifier(self.normalize(attr))

    def fit(self, x_clean, x_adv, epochs, lr, batch_size, seed):
        """Melatih detektor dari atribusi sampel bersih dan sampel adversarial."""
        attr_clean = explain_batches(self.ids_model, x_clean, self.ig_steps)
        attr_adv = explain_batches(self.ids_model, x_adv, self.ig_steps)
        attr = torch.cat([attr_clean, attr_adv])
        labels = torch.cat([torch.zeros(len(attr_clean)), torch.ones(len(attr_adv))]).long()

        self.mean = attr.mean(dim=0)
        self.std = attr.std(dim=0).clamp(min=1e-6)
        train_classifier(self.classifier, self.normalize(attr), labels, epochs, lr, batch_size, seed)
        self.eval()
        return self

    @staticmethod
    def margin(logits):
        """Skor detektor: selisih logit adversarial dan bersih (tidak jenuh seperti softmax)."""
        return logits[:, 1] - logits[:, 0]

    def score(self, x, batch_size=2048):
        """Skor adversarial untuk seluruh data; makin besar makin dicurigai adversarial."""
        x = torch.as_tensor(x, dtype=torch.float32)
        out = [self.margin(self(x[i:i + batch_size])).detach() for i in range(0, len(x), batch_size)]
        return torch.cat(out).numpy()
