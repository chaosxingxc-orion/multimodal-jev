import copy
import unittest
from tools.reports.summarize_context_audio_stages import STAGES, summarize


class SummaryTests(unittest.TestCase):
    def test_first_difference_separate_from_probability_impact(self):
        exact=dict(exact=True,count=0,max_abs=0.)
        stages={name:dict(exact) for name in STAGES}
        record=dict(id='a',logits=dict(exact),prob=dict(exact))
        report=dict(status='actual_model_stage_comparison',rows=[dict(id='a',comparisons={
            'baseline':copy.deepcopy(stages),'acoustic':copy.deepcopy(stages)})],
            batch={'baseline':copy.deepcopy(stages),'acoustic':copy.deepcopy(stages)},
            scores={mode:{part:[copy.deepcopy(record)] for part in ('records','batch-records')}
                    for mode in ('baseline','acoustic')})
        report['batch']['acoustic']['layer_02']=dict(exact=False,count=1,max_abs=.03125)
        report['batch']['acoustic']['layer_08']=dict(exact=False,count=3,max_abs=.5)
        result=summarize(report)
        self.assertEqual(result['modes']['acoustic']['batch_first_divergence'],'layer_02')
        self.assertEqual(result['modes']['acoustic']['scores']['batch-records']['different_queries'],0)
        self.assertIsNone(result['modes']['baseline']['batch_first_divergence'])
        self.assertFalse(result['full_model_admitted'])
        report['scores']['baseline']['records'][0]['id']='wrong'
        with self.assertRaises(ValueError):summarize(report)


if __name__=='__main__':unittest.main()
