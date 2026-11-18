"""Check that the diagnostic detects changes and preserves real mel layout."""
import unittest

import torch

from tools.reports.context_acoustic_numeric_probe import chunks, difference


class AcousticNumericProbeTests(unittest.TestCase):
    def test_chunking_preserves_channel_and_time_identity(self):
        source = torch.arange(128 * 300, dtype=torch.float32).reshape(1, 128, 300)
        actual = chunks(source)
        self.assertEqual(tuple(actual.shape), (3, 1, 128, 100))
        reconstructed = torch.cat([part[0] for part in actual], dim=-1)
        self.assertTrue(torch.equal(reconstructed, source[0]))

    def test_difference_detects_one_bf16_change_and_coordinate(self):
        ref = torch.zeros(3, 1, 4, 8, dtype=torch.bfloat16)
        candidate = ref.clone()
        candidate[2, 0, 3, 7] = 0.125
        report = difference(ref, candidate)
        self.assertFalse(report['exact'])
        self.assertEqual(report['count'], 1)
        self.assertEqual(report['max_abs'], 0.125)
        self.assertEqual(report['examples'][0]['index'], [2, 0, 3, 7])
        self.assertTrue(difference(ref, ref.clone())['exact'])

    def test_invalid_comparisons_fail_instead_of_passing(self):
        ref = torch.ones(2, 3)
        for other in (torch.ones(3, 2), ref.to(torch.bfloat16),
                      torch.full_like(ref, float('nan')), torch.full_like(ref, float('inf'))):
            with self.assertRaises(ValueError):
                difference(ref, other)
        for value in (torch.ones(2, 128, 100), torch.ones(1, 128, 99)):
            with self.assertRaises(ValueError):
                chunks(value)


if __name__ == '__main__':
    unittest.main()
