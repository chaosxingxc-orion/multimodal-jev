"""Independent comparison must expose a single intermediate-layer corruption."""
import tempfile
import unittest
from pathlib import Path
import torch
from tools.reports.context_audio_stage_probe import compare


class StageComparisonTests(unittest.TestCase):
    def test_single_layer_corruption_is_detected_in_serial_and_batch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            names=['encoder_input',*[f'layer_{i:02d}' for i in range(18)],'ln_post','projected']
            for mode in ('hf','baseline','acoustic'):
                (root/mode).mkdir()
                rows=[]
                for i in range(2):
                    values={name:[torch.full((i+2,3),float(i),dtype=torch.bfloat16)] for name in names}
                    if mode=='acoustic' and i==1:values['layer_03'][0][1,2]+=0.125
                    torch.save(values,root/mode/f'row-{i:02d}.pt');rows.append(values)
                if mode!='hf':torch.save({name:[torch.cat([v[name][0] for v in rows])] for name in names},root/mode/'batch.pt')
            result=compare(root,{'rows':[{'id':'a'},{'id':'b'}]})
            self.assertTrue(result['rows'][0]['comparisons']['acoustic']['layer_03']['exact'])
            self.assertFalse(result['rows'][1]['comparisons']['acoustic']['layer_03']['exact'])
            self.assertEqual(result['batch']['acoustic']['layer_03']['count'],1)
            self.assertTrue(result['batch']['baseline']['layer_03']['exact'])
            self.assertFalse(result['full_model_parity_verified'])


if __name__=='__main__':unittest.main()
