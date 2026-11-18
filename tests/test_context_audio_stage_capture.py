"""Capture must observe outputs without aliasing or mutating model execution."""
import unittest
from types import SimpleNamespace
import torch
from tools.reports.context_audio_stage_capture import attach


class CaptureTests(unittest.TestCase):
    def test_hooks_snapshot_before_inplace_change_and_remove_cleanly(self):
        class Tower(torch.nn.Module):
            def __init__(self):
                super().__init__()
                for name in ('conv2d1','conv2d2','conv2d3','conv_out','ln_post'):
                    setattr(self,name,torch.nn.Identity())
                self.layers=torch.nn.ModuleList([torch.nn.Identity()])
                self.positional_embedding=SimpleNamespace(positional_embedding=torch.zeros(2,3))
            def forward(self,x):
                x=self.conv2d1(x);x=x+1
                x=self.conv2d2(x);x=self.conv2d3(x);x=self.conv_out(x)
                x=self.layers[0](x);return self.ln_post(x)
        model=Tower();x=torch.ones(2,3)
        handles,values=attach(model,model)
        y=model(x)
        self.assertTrue(torch.equal(y,torch.full_like(x,2)))
        x.add_(10);y.add_(20)
        self.assertTrue(torch.equal(values['conv2d1'][0],torch.ones(2,3)))
        self.assertTrue(torch.equal(values['projected'][0],torch.full((2,3),2)))
        self.assertEqual(len(values['encoder_input']),1)
        for h in handles:h.remove()
        model(x)
        self.assertEqual(len(values['projected']),1)


if __name__=='__main__':unittest.main()
