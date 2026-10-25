import torch


class DomainConstraints:
    """Batasan domain trafik jaringan: immutable, increase-only, batas nilai, integer, dan fitur turunan."""

    def __init__(self, dataset):
        spec, names = dataset.spec, dataset.feature_names
        self.derive_fn = spec.derive
        self.index = {n: i for i, n in enumerate(names)}
        self.mutable = torch.tensor([n in spec.increase_only + spec.free for n in names])
        self.increase_only = torch.tensor([n in spec.increase_only for n in names])
        self.integer = torch.tensor([n in spec.integer for n in names])
        self.derived = torch.tensor([n in spec.derived for n in names])
        self.changeable = self.mutable | self.derived
        self.log_mask = torch.tensor(dataset.log_mask)
        self.raw_min = torch.tensor(dataset.raw_min)
        self.raw_max = torch.tensor(dataset.raw_max)
        self.span = torch.where(self.raw_max - self.raw_min > 0,
                                self.raw_max - self.raw_min, torch.ones_like(self.raw_max))

    def to_raw(self, x):
        """Mengembalikan fitur ternormalisasi ke nilai aslinya."""
        raw = x * self.span + self.raw_min
        return torch.where(self.log_mask, torch.expm1(raw), raw)

    def from_raw(self, raw):
        """Mengubah nilai asli ke ruang fitur ternormalisasi."""
        raw = torch.where(self.log_mask, torch.log1p(raw.clamp(min=0)), raw)
        return (raw - self.raw_min) / self.span

    def clamp_range(self, x):
        """Fitur primer dibatasi [0, 1]; fitur turunan boleh melewati batas data latih."""
        return torch.where(self.derived, x, torch.clamp(x, 0.0, 1.0))

    def apply_derived(self, x, x_orig):
        """Menghitung ulang fitur turunan dari fitur primer (jika dataset memilikinya)."""
        if self.derive_fn is None:
            return x
        raw = self.derive_fn(self.to_raw(x), self.to_raw(x_orig), self.index)
        x_new = self.clamp_range(self.from_raw(raw))
        return torch.where(self.changeable, x_new, x)

    def project(self, x_adv, x_orig, eps):
        """Proyeksi sampel ke ruang yang valid (dipanggil setiap iterasi serangan)."""
        delta = torch.clamp(x_adv - x_orig, -eps, eps)
        delta = torch.where(self.mutable, delta, torch.zeros_like(delta))
        delta = torch.where(self.increase_only, delta.clamp(min=0), delta)
        x = torch.where(self.derived, x_orig, torch.clamp(x_orig + delta, 0.0, 1.0))
        return self.apply_derived(x, x_orig)

    def round_integer(self, x_adv, x_orig):
        """Membulatkan fitur hitungan ke bilangan bulat tanpa melanggar arah perubahan."""
        raw = self.to_raw(x_adv)
        raw_orig = self.to_raw(x_orig)
        rounded = torch.round(raw)
        rounded = torch.where(self.increase_only, torch.maximum(rounded, torch.round(raw_orig)), rounded)
        x_int = torch.where(self.integer & self.mutable, self.from_raw(rounded), x_adv)
        return self.clamp_range(x_int)

    def finalize(self, x_adv, x_orig, eps):
        """Langkah akhir serangan: proyeksi, pembulatan integer, fitur turunan, lalu fitur immutable dikunci."""
        x = self.round_integer(self.project(x_adv, x_orig, eps), x_orig)
        x = self.apply_derived(x, x_orig)
        return torch.where(self.changeable, x, x_orig)

    def violations(self, x_adv, x_orig, eps, tol=1e-4):
        """Menghitung jumlah sampel yang melanggar batasan domain."""
        diff = x_adv - x_orig
        immutable_bad = (diff[:, ~self.changeable].abs() > tol).any(dim=1)
        decrease_bad = (diff[:, self.increase_only] < -tol).any(dim=1)
        primary = x_adv[:, ~self.derived]
        range_bad = ((primary < -tol) | (primary > 1 + tol)).any(dim=1)
        raw = self.to_raw(x_adv)[:, self.integer]
        integer_bad = ((raw - torch.round(raw)).abs() > 1e-3 * raw.abs().clamp(min=1)).any(dim=1)
        derived_bad = (self.apply_derived(x_adv, x_orig) - x_adv)[:, self.derived].abs().gt(1e-3).any(dim=1)
        return {
            "immutable": int(immutable_bad.sum()),
            "increase_only": int(decrease_bad.sum()),
            "range": int(range_bad.sum()),
            "integer": int(integer_bad.sum()),
            "derived": int(derived_bad.sum()),
        }
