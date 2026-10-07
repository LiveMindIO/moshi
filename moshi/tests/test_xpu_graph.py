"""Checks that the opt-in Mimi wrapper selects the XPU graph backend."""

import unittest
from contextlib import nullcontext
from unittest.mock import patch

import torch

from moshi.utils.compile import CUDAGraphed


class TestXpuGraph(unittest.TestCase):
    def test_xpu_graph_copies_live_inputs_without_recapturing(self):
        graphs = []

        class FakeGraph:
            def __init__(self):
                self.replays = 0
                graphs.append(self)

            def replay(self):
                self.replays += 1

        inputs = []

        def step(value):
            inputs.append(value)
            return value * 2

        with (
            patch.object(torch.xpu, "XPUGraph", FakeGraph),
            patch.object(torch.xpu, "graph", lambda _graph: nullcontext()),
        ):
            wrapped = CUDAGraphed(step, graph_device="xpu")
            self.assertEqual(wrapped(torch.tensor([1])).item(), 2)
            self.assertEqual(wrapped(torch.tensor([2])).item(), 4)
            self.assertEqual(len(graphs), 1)
            self.assertEqual(graphs[0].replays, 1)
            wrapped(torch.tensor([3]))
            self.assertEqual(inputs[1].item(), 3)
            self.assertEqual(graphs[0].replays, 2)


if __name__ == "__main__":
    unittest.main()
