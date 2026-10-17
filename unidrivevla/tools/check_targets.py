"""CPU check: variable-size targets survive padding, batching and the model's unpadding masks."""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from unidrivevla.adapters.unidrivevla import pad_rows
from unidrivevla.data.registry import collate_dicts


def main():
    for minimum, shape in ((128, (9,)), (64, (38, 20, 2))):
        samples, expected = [], []
        for count in (minimum + 3, 2, 0):
            value = np.arange(count * int(np.prod(shape)), dtype=np.float32).reshape((count,) + shape)
            padded, mask = pad_rows(value, minimum)
            samples.append({"values": torch.from_numpy(padded), "mask": torch.from_numpy(mask)})
            expected.append(torch.from_numpy(value))
        batch = collate_dicts(samples)
        assert batch["values"].shape[1] == minimum + 3
        for index, value in enumerate(expected):
            restored = batch["values"][index][batch["mask"][index].bool()]
            torch.testing.assert_close(restored, value, rtol=0, atol=0)
    print("OK: oversized, small and empty box/map targets survive batching without truncation")


if __name__ == "__main__":
    main()
