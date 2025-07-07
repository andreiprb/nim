import numpy as np

from Nim.NimLogic import NimLogic


def generate_sorted_arrays(length, max_val, min_val=0):
    if length == 0:
        yield []
        return

    for first in range(min_val, max_val+1):
        for rest in generate_sorted_arrays(length - 1, max_val, first):
            yield [first] + rest


def test_over_range(max_value, array_length):
    originals = dict()
    results = dict()

    for test_array in generate_sorted_arrays(array_length, max_value):
        original = np.array(test_array)
        original_xor = np.bitwise_xor.reduce(original)

        result, _ = NimLogic.reduce_state(test_array.copy(), list(range(len(test_array))))
        result_xor = np.bitwise_xor.reduce(np.array(result))

        originals[str(original)] = 0
        results[str(result)] = 0

        assert original_xor == result_xor, f"XOR not preserved for {original}: {original_xor} != {result_xor}"

    print(f"Sorted arrays tested: {len(originals)}")
    print(f"Unique results: {len(results)}")


if __name__ == '__main__':
    original = [6, 8, 4, 3]
    print("Original:", original)

    canonical, index_mapping = NimLogic.canonicalize_state(original)
    print("Canonical:", canonical)
    print("Index Mapping:", index_mapping)

    reduced, index_mapping = NimLogic.reduce_state(canonical.copy(), index_mapping)
    print("Reduced:", reduced)
    print("Index Mapping:", index_mapping)
