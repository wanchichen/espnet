
import copy
import logging
from typing import Tuple

import torch
from typeguard import check_argument_types

from espnet2.asr.postencoder.abs_postencoder import AbsPostEncoder


class AdapterPostEncoder(AbsPostEncoder):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        cluster_type: str = "soft",
        adadpter_type: str = "full",
        num_adapter: int = 1
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.adapter_type = adadpter_type
        self.cluster_type = cluster_type
        self.mask = None
        if self.adapter_type == "diagonal":
            self.mask = torch.eye(input_size, dtype=bool).cuda()
        self.adapters = [torch.nn.Linear(input_size, output_size) for i in range(num_adapter)]
        self.adapters = torch.nn.ModuleList(self.adapters)

    def output_size(self) -> int:
        return self.output_size

    def forward(
        self, enc_out: torch.Tensor, enc_lengths: torch.Tensor, cluster_ids:  torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        adapter_outs = []
        if len(cluster_ids.shape) == 1:
            cluster_ids = cluster_ids.unsqueeze(0)
        assert len(self.adapters) == cluster_ids.shape[1]# should be of shape bs x num_clusters
        
        for i, adapter in enumerate(self.adapters):

            # mask non-diagonal entries
            if self.adapter_type == "diagonal":
                adapter.weight.data *= self.mask

            # weighting step
            if self.cluster_type == "soft":
                adapter_out = adapter(enc_out)
                cluster_probs = cluster_ids[:, i] # bs
                cluster_probs = cluster_probs.unsqueeze(1).unsqueeze(2).expand(adapter_out.shape)
                adapter_out *= cluster_probs
                adapter_outs.append(adapter_out)
            if self.cluster_type == "hard":
                # do some assert to make sure all cluster ids are the same
                adapter_id = cluster_ids[0]
                adapter_out = self.adapters[adapter_id](enc_out)

        # sum step
        if self.cluster_type == "soft":
            adapter_out = adapter_outs[0]
            for a_out in adapter_outs[1:]:
                adapter_out += a_out
            adapter_out /= len(self.adapters)
        return adapter_out, enc_lengths
