import torch

class SRN(nn.Module):
    """A simple recurrent network."""

    @property
    def d_embedding(self) -> int:  # noqa: D102
        return self._d_embedding

    @property
    def d_hidden(self) -> int:  # noqa: D102
        return self._d_hidden

    @property
    def n_layers(self) -> int:  # noqa: D102
        return self._n_layers

    @property
    def dropout(self) -> float:  # noqa: D102
        return self._dropout

    @property
    def activation(self) -> str:  # noqa: D102
        return self._activation

    @property
    def n_vocab(self) -> int:  # noqa: D102
        return self._n_vocab

    @property
    def batch_first(self) -> bool:  # noqa: D102
        return self._batch_first

    @property
    def bias(self) -> bool:  # noqa: D102
        return self._bias

    @property
    def weight_scale(self) -> float:  # noqa: D102
        return self._weight_scale

    @property
    def num_parameters(self) -> int:  # noqa: D102
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __init__(
        self,
        activation: str,
        batch_first: bool,
        bias: bool,
        dropout: float | None,
        d_embedding: int,
        d_hidden: int,
        n_layers: int,
        n_vocab: int,
        weight_scale: float,
    ):
        """Initialize an SRN.

        Args:
            activation (str): The activation function.
            batch_first (bool): Whether the batch is first.
            bias (bool): Whether to use bias.
            dropout (float | None): The dropout rate.
            d_embedding (int): The embedding dimension.
            d_hidden (int): The hidden dimension.
            n_layers (int): The number of layers.
            n_vocab (int): The number of vocabulary.
            weight_scale (float): The weight scale.

        """
        self._activation = activation
        self._batch_first = batch_first
        self._bias = bias
        self._dropout = dropout
        self._d_embedding = d_embedding
        self._d_hidden = d_hidden
        self._n_layers = n_layers
        self._n_vocab = n_vocab
        self._weight_scale = weight_scale

        super().__init__()

        # self.embedding = nn.Embedding(num_embeddings=n_vocab, embedding_dim=d_embedding)
        # self.rnn = nn.GRU(d_embedding, d_hidden, num_layers=n_layers, bias=True, batch_first=True, dropout=0.0, bidirectional=False, device=None, dtype=None)
        self.rnn = nn.RNN(
            input_size=d_embedding,
            hidden_size=d_hidden,
            num_layers=n_layers,
            dropout=dropout,
            batch_first=batch_first,
            bias=bias,
            bidirectional=False,
            nonlinearity="tanh",
        )

        with torch.no_grad():
            for p in self.rnn.parameters():
                p.copy_(p * 0.01)

    def forward(self, x: torch.Tensor, h: torch.Tensor | None = None):
        """Forward pass."""
        if h is None:
            h = x.new_zeros(self.n_layers, x.size(0), self.d_hidden)

        x, h = self.rnn(input=x, hx=h)
        return x, h