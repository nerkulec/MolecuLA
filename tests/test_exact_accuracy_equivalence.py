"""Regression tests for full-sequence exact-match evaluation."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from models.autoregressive_vae import VaeTransformer
from train_lignin_vae import exact_match_categories


# Midpoints of 20 equally populated slices of the 976,794-row validation split.
# They are the 2.5%, 7.5%, ..., 97.5% sequence-length quantiles measured from
# artifacts/lignin_retraining/encoded on 2026-09-01.
REPRESENTATIVE_LENGTHS = [
    83,
    110,
    123,
    134,
    142,
    148,
    153,
    158,
    163,
    169,
    174,
    180,
    186,
    192,
    200,
    210,
    221,
    237,
    255,
    288,
]


def greedy_categories_from_tokens(
    generated: torch.Tensor, targets: torch.Tensor, eos_id: int = 2, pad_id: int = 0
) -> torch.Tensor:
    """Apply the historical greedy exact-match comparison to generated rows."""
    categories = []
    for target, prediction in zip(targets, generated, strict=True):
        target = target[target.ne(pad_id)]
        eos = torch.nonzero(prediction.eq(eos_id), as_tuple=False)
        if len(eos):
            prediction = prediction[: int(eos[0]) + 1]
        categories.append(torch.equal(target, prediction))
    return torch.tensor(categories, dtype=torch.bool)


class ExactAccuracyEquivalenceTest(unittest.TestCase):
    def test_new_encoder_is_invariant_to_right_padding(self):
        torch.manual_seed(3)
        model = VaeTransformer(
            vocab_size=11,
            hidden_size=16,
            latent_size=16,
            max_len=12,
            attn_heads=4,
            num_slots=4,
            encoder_layers=1,
            decoder_layers=1,
            padding_invariant_encoder=True,
        ).eval()
        unpadded = torch.tensor([[1, 4, 5, 2]])
        padded = torch.tensor([[1, 4, 5, 2, 0, 0, 0]])
        with torch.inference_mode():
            mu_unpadded, logvar_unpadded = model.encode(unpadded)
            mu_padded, logvar_padded = model.encode(padded)
        torch.testing.assert_close(mu_unpadded, mu_padded, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(
            logvar_unpadded, logvar_padded, rtol=1e-6, atol=1e-6
        )

    def test_real_decoder_categories_agree_on_balanced_examples(self):
        """Teacher forcing and greedy decoding classify the same fixed-z rows."""
        torch.manual_seed(7)
        model = VaeTransformer(
            vocab_size=11,
            hidden_size=16,
            latent_size=16,
            max_len=12,
            attn_heads=4,
            num_slots=4,
            encoder_layers=1,
            decoder_layers=1,
        ).eval()
        with torch.no_grad():
            # Valid encoded sequences never contain PAD before EOS. Keep the
            # randomly initialized decoder from generating PAD as a real token.
            model.fc_output.weight[0].zero_()
            model.fc_output.bias[0] = -1e6
        z = torch.randn(10, 16)

        with torch.inference_mode():
            generated = model.decode(z, max_len=8)
            targets = torch.zeros_like(generated)
            lengths = []
            for row, prediction in enumerate(generated):
                eos = torch.nonzero(prediction.eq(2), as_tuple=False)
                length = int(eos[0]) + 1 if len(eos) else len(prediction)
                targets[row, :length] = prediction[:length]
                lengths.append(length)
            # Corrupt only the final non-padding target. Its prediction depends
            # solely on the unchanged prefix, so the affected rows must be wrong.
            for row in range(5, 10):
                column = lengths[row] - 1
                targets[row, column] = (
                    targets[row, column] % (model.fc_output.out_features - 1)
                ) + 1
            logits = model.decode(z, x_in=targets[:, :-1])

        teacher = exact_match_categories(logits.argmax(-1), targets[:, 1:])
        greedy = greedy_categories_from_tokens(generated, targets)
        self.assertTrue(torch.equal(teacher, greedy))
        self.assertEqual(int(teacher.sum()), 5)
        self.assertGreaterEqual(float(teacher.float().mean()), 0.4)
        self.assertGreaterEqual(float((~teacher).float().mean()), 0.4)

    def test_representative_lengths_cover_both_categories_equally(self):
        """The representative fixture pairs one correct and wrong row per quantile."""
        lengths = torch.tensor(REPRESENTATIVE_LENGTHS).repeat_interleave(2)
        expected = torch.tensor([True, False] * len(REPRESENTATIVE_LENGTHS))

        # Construct predictions directly because this test validates padding and
        # category aggregation over the real dataset's length distribution.
        width = int(lengths.max())
        encoded = torch.zeros(len(lengths), width, dtype=torch.long)
        for row, length in enumerate(lengths.tolist()):
            encoded[row, 0] = 1
            encoded[row, 1 : length - 1] = 4
            encoded[row, length - 1] = 2
        targets = encoded[:, 1:]
        predicted = targets.clone()
        predicted[~expected, 1] = 5
        generated = torch.cat([encoded[:, :1], predicted], dim=1)

        teacher = exact_match_categories(predicted, targets)
        greedy = greedy_categories_from_tokens(generated, encoded)
        self.assertTrue(torch.equal(teacher, greedy))
        self.assertTrue(torch.equal(teacher, expected))
        self.assertEqual(float(teacher.float().mean()), 0.5)
        self.assertEqual(float((~teacher).float().mean()), 0.5)
        self.assertTrue(
            torch.equal(lengths[teacher].sort().values, lengths[~teacher].sort().values)
        )


if __name__ == "__main__":
    unittest.main()
