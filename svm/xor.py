import numpy as np
import itertools
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# import torch
# import torch.nn as nn
# import torch.optim as optim

TOKEN_LEN, TOKEN_CNT = 3, 200

"""******************************************************************************"""

tokens = np.array(list(itertools.product([0, 1], repeat=TOKEN_LEN)))
t1 = np.repeat(tokens, 2, axis=0)
t2 = np.tile(tokens, (2, 1))

X_train = np.stack((t1, t2), axis=1).reshape((-1, 2 * TOKEN_LEN))

y_train_ext = np.bitwise_xor(t1, t2)
y_train = (np.sum(y_train_ext, axis=1) > 0).astype(int)

svc_model = SVC(kernel='rbf', gamma=2.0)
svc_model.fit(X_train, y_train)

svc_preds = svc_model.predict(X_train)
print("SVC accuracy", accuracy_score(svc_preds, y_train))

"""******************************************************************************"""

rng = np.random.default_rng(seed=42)
sequence = tokens[rng.integers(0, len(tokens), size=TOKEN_CNT)]

for i in range(0, TOKEN_CNT):
    print(sequence[i])

state = np.zeros(TOKEN_LEN, dtype=int)

for i in range(0, TOKEN_CNT):
    current_input = sequence[i]
    x_input = np.concatenate([state, current_input]).reshape(1, -1)
    prediction = svc_model.predict(x_input)[0]

    state = np.bitwise_xor(state, current_input)

    print(prediction, state)


# X_tensor = torch.tensor(X_train, dtype=torch.float32)
# y_tensor = torch.tensor(y_train, dtype=torch.float32)
#
# class MLPXOR(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super(MLPXOR, self).__init__()
#         self.net = nn.Sequential(
#             nn.Linear(input_dim, 16),
#             nn.ReLU(),
#             nn.Linear(16, output_dim),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         return self.net(x)
#
#
# model = MLPXOR(input_dim=2 * LEN, output_dim=LEN)
# criterion = nn.BCELoss()
# optimizer = optim.Adam(model.parameters(), lr=0.01)
#
# # 3. Antrenare
# for epoch in range(3000):
#     optimizer.zero_grad()
#     outputs = model(X_tensor)
#     loss = criterion(outputs, y_tensor)
#     loss.backward()
#     optimizer.step()
#
#     if (epoch + 1) % 500 == 0:
#         print(f"Epoch {epoch + 1}/3000, Loss: {loss.item():.4f}")
#
# # 4. Evaluare
# with torch.no_grad():
#     preds = model(X_tensor)
#     preds_binary = (preds > 0.5).int().numpy()
#     accuracy = np.mean(preds_binary == y_train)
#     print("\n🧠 MLP XOR Accuracy:", accuracy)
#     print("Predicții XOR:")
#     print(preds_binary)
#     print("Etichete reale:")
#     print(y_train)