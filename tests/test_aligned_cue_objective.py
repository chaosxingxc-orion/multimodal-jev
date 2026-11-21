import unittest
import torch
from multimodal_jev.aligned_cue_objective import cue_pair_loss,remap_loss


class CueObjectiveTests(unittest.TestCase):
    def test_both_directions_get_local_credit(self):
        up=torch.zeros(1,3,requires_grad=True);down=torch.zeros_like(up,requires_grad=True)
        cue_pair_loss(up,down,torch.tensor([2]),torch.tensor([0]),temperature=1.,probability_margin=.1).sum().backward()
        self.assertLess(up.grad[0,2],0);self.assertGreater(up.grad[0,0],0)
        self.assertTrue(torch.equal(up.grad,-down.grad));self.assertEqual(up.grad[0,1],0)

    def test_order_is_invariant_only_after_semantic_remap(self):
        z=torch.tensor([[5.,-1.,-3.]])
        mapping=torch.tensor([[1,2,0]]);permuted=torch.tensor([[-3.,5.,-1.]])
        consistent,gold=remap_loss(z,permuted,mapping,labels=torch.tensor([0]))
        self.assertEqual(consistent.item(),0.)
        wrong_consistent,wrong_gold=remap_loss(z,z,mapping,labels=torch.tensor([0]))
        self.assertGreater(wrong_consistent,consistent);self.assertGreater(wrong_gold,gold)
        with self.assertRaises(ValueError):remap_loss(z,z,torch.tensor([[0,0,1]]),labels=torch.tensor([0]))
