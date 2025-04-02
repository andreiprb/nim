import numpy as np
import itertools
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# import torch
# import torch.nn as nn
# import torch.optim as optim

TOKEN_LEN, TOKEN_CNT = 3, 5

"""******************************************************************************"""

tokens = np.array(list(itertools.product([0, 1], repeat=TOKEN_LEN)))
pairs = np.array(list(itertools.product(tokens, tokens)))

t1 = np.array([p[0] for p in pairs])
t2 = np.array([p[1] for p in pairs])

X_train = np.hstack((t1, t2))

y_train_ext = np.bitwise_xor(t1, t2)
y_train = (np.sum(y_train_ext, axis=1) > 0).astype(int)

svc_model = SVC(kernel='rbf', gamma=2.0)
svc_model.fit(X_train, y_train)

svc_preds = svc_model.predict(X_train)

print(y_train, svc_preds)

print("SVC accuracy", accuracy_score(svc_preds, y_train))

"""******************************************************************************"""

rng = np.random.default_rng(seed=42)
sequence = tokens[rng.integers(0, len(tokens), size=TOKEN_CNT)]

state = sequence[0]

for i in range(1, TOKEN_CNT):
    next_state = sequence[i]
    x_input = np.concatenate([state, next_state]).reshape(1, -1)
    prediction = svc_model.predict(x_input)[0]

    new_state = np.bitwise_xor(state, next_state)

    print(state, next_state, "->", new_state, prediction)

    state = new_state


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