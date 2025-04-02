import torch
import torch.nn as nn
import torch.nn.functional as F


class VariableLengthXORNet(nn.Module):
    def __init__(self, n_bits: int, hidden_dim: int = 32):
        super().__init__()
        self.embed = nn.Linear(n_bits, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, n_bits)
        self.out = nn.Linear(n_bits, 1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        """
        x: Tensor de formă (batch, seq_len, n_bits)
        lengths: Tensor de formă (batch,) care indică lungimea reală a fiecărei secvențe
        """
        x = self.embed(x)  # (batch, seq_len, hidden_dim)
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.gru(packed)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)

        # Extragem ultima stare validă (pentru fiecare secvență)
        idx = (lengths - 1).unsqueeze(1).unsqueeze(2).expand(-1, 1, output.size(2))  # (batch, 1, hidden_dim)
        last_hidden = output.gather(1, idx).squeeze(1)  # (batch, hidden_dim)

        xor_out = self.fc(last_hidden)  # (batch, n_bits)
        xor_bit = torch.sigmoid(self.out(xor_out))  # (batch, 1)
        return xor_bit


# Generează date: listă de vectori binari de lungimi variabile
def generate_batch(batch_size: int, max_seq_len: int, n_bits: int):
    sequences = []
    lengths = []
    targets = []
    for _ in range(batch_size):
        length = torch.randint(1, max_seq_len + 1, (1,)).item()
        seq = torch.randint(0, 2, (length, n_bits)).float()
        xor_result = seq[0]
        for i in range(1, length):
            xor_result = xor_result != seq[i]
        label = int(xor_result.any())  # 1 dacă există cel puțin un 1 în XOR-ul total

        padded = F.pad(seq, (0, 0, 0, max_seq_len - length))  # pad la dreapta
        sequences.append(padded)
        lengths.append(length)
        targets.append([label])

    x = torch.stack(sequences)  # (batch, max_seq_len, n_bits)
    y = torch.tensor(targets).float()
    lengths = torch.tensor(lengths)
    return x, y, lengths


# Antrenare
def train_model(n_bits=8, max_seq_len=10, epochs=300, batch_size=64):
    model = VariableLengthXORNet(n_bits)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.BCELoss()

    for epoch in range(epochs):
        x, y, lengths = generate_batch(batch_size, max_seq_len, n_bits)
        pred = model(x, lengths)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 50 == 0:
            acc = ((pred > 0.5) == y).float().mean().item()
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}, Accuracy: {acc:.2f}")

    return model


# Testare
def test_model(model, n_bits=8, max_seq_len=10, num_samples=5):
    model.eval()
    with torch.no_grad():
        x, y, lengths = generate_batch(num_samples, max_seq_len, n_bits)
        pred = model(x, lengths)
        for i in range(num_samples):
            real_len = lengths[i].item()
            input_seq = x[i][:real_len].int().tolist()
            expected = int(y[i].item())
            predicted = int(pred[i].item() > 0.5)
            print(f"Input: {input_seq} -> Predicted: {predicted}, Expected: {expected}")


if __name__ == "__main__":
    model = train_model(n_bits=8, max_seq_len=10)
    test_model(model, n_bits=8, max_seq_len=10)
