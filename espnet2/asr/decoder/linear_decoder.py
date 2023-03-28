import torch

from typing import Optional
from espnet2.asr.decoder.abs_decoder import AbsDecoder


class LinearDecoder(AbsDecoder):
    """Linear decoder for speaker diarization"""

    def __init__(
        self,
        encoder_output_size: int,
        vocab_size: int,
    ):
        super().__init__()
        self.output = torch.nn.Linear(encoder_output_size, vocab_size)
        # TODO1 (checkpoint3): initialize a linear projection layer

    def forward(self, input: torch.Tensor, ilens: Optional[torch.Tensor]):
        """Forward.
        Args:
            input (torch.Tensor): hidden_space [Batch, T, F]
            ilens (torch.Tensor): input lengths [Batch]
        """
        # TODO2 (checkpoint3): compute mean over time-domain (dimension 1)
        input = torch.mean(input, dim=1)

        # TODO3 (checkpoint3): apply the projection layer
        output = self.output(input)

        # TODO4 (checkpoint3): change the return value accordingly
        return output