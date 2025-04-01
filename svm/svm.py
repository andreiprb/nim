import numpy as np
import itertools, time
from sklearn import svm

def generate_training_data(num_samples=1000):
    bit_seq1 = np.random.randint(0, 2, (num_samples, 8))
    bit_seq2 = np.random.randint(0, 2, (num_samples, 8))

    X = np.hstack((bit_seq1, bit_seq2))
    y = np.bitwise_xor(bit_seq1, bit_seq2)

    return X, y

# Generate training data
X_train, y_train_original = generate_training_data()

# Generate test data
test_size = 1000000
all_8bit_sequences = np.array(list(itertools.product([0, 1], repeat=8)))
indices1 = np.random.choice(256, test_size, replace=True)
indices2 = np.random.choice(256, test_size, replace=True)

bit_seq1_test = all_8bit_sequences[indices1]
bit_seq2_test = all_8bit_sequences[indices2]
X_test = np.hstack((bit_seq1_test, bit_seq2_test))
y_test_original = np.bitwise_xor(bit_seq1_test, bit_seq2_test)

# Option 1: Train separate models for each bit position
print("Training models for each bit position...")
poly_model = svm.SVC(kernel='poly', degree=2)
for bit_pos in range(8):
    print(f"Training model for bit position {bit_pos}...")
    y_train_bit = y_train_original[:, bit_pos]

    start_time = time.time()
    poly_model.fit(X_train, y_train_bit)

    training_time = time.time() - start_time
    print(f"  Training completed in {training_time:.2f} seconds")

# Function to predict using the 8 separate models
def predict_all_bits(model, X):
    predictions = np.zeros((X.shape[0], 8), dtype=int)

    for bit_pos in range(8):
        print(f"Making predictions for bit position {bit_pos}...")
        start_time = time.time()
        predictions[:, bit_pos] = model.predict(X)
        prediction_time = time.time() - start_time
        print(f"  Prediction completed in {prediction_time:.2f} seconds")

    return predictions

print("\nMaking predictions on test data...")
y_pred = predict_all_bits(poly_model, X_test)

# Calculate accuracy metrics
bit_accuracy = np.mean(y_pred == y_test_original, axis=0)
overall_accuracy = np.mean(y_pred == y_test_original)

# Count exactly correct sequences (all 8 bits correct)
exact_matches = np.all(y_pred == y_test_original, axis=1)
exact_accuracy = np.mean(exact_matches)

# Display some example predictions
print("\nExample predictions:")
for i in range(min(5, len(X_test))):
    seq1 = X_test[i, :8]
    seq2 = X_test[i, 8:]

    # Format as binary strings for better readability
    seq1_str = ''.join(map(str, seq1))
    seq2_str = ''.join(map(str, seq2))
    true_xor_str = ''.join(map(str, y_test_original[i]))
    pred_xor_str = ''.join(map(str, y_pred[i]))

    print(f"Input 1:      {seq1_str}")
    print(f"Input 2:      {seq2_str}")
    print(f"Expected XOR: {true_xor_str}")
    print(f"Predicted:    {pred_xor_str}")
    print(f"Correct:      {'✓' if np.array_equal(y_pred[i], y_test_original[i]) else '✗'}")
    print("-" * 40)

# Print accuracy metrics
print("\nAccuracy by bit position:")
for i, acc in enumerate(bit_accuracy):
    print(f"Bit position {i}: {acc:.2f}")

print(f"\nOverall bit-level accuracy: {overall_accuracy:.4f}")
print(
    f"Exact sequence match accuracy: {exact_accuracy:.4f} ({int(np.sum(exact_matches))} out of {len(X_test)} sequences)")

# Count accuracy by Hamming distance (number of bit errors)
hamming_distances = np.sum(y_pred != y_test_original, axis=1)
for dist in range(9):  # 0 to 8 possible bit errors
    count = np.sum(hamming_distances == dist)
    percentage = count / len(X_test) * 100
    print(f"Sequences with {dist} bit errors: {count} ({percentage:.2f}%)")