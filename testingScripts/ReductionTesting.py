import numpy as np

from HelperLogic.HelperLogic import HelperLogic


def test_over_range(max_value, array_length):
    originals = dict()
    results = dict()

    for test_array in HelperLogic.generate_sorted_arrays_desc(array_length, max_value):
        original = np.array(test_array)
        original_xor = np.bitwise_xor.reduce(original)

        result = HelperLogic.reduce_state(test_array.copy())
        result_xor = np.bitwise_xor.reduce(np.array(result))

        originals[str(original)] = 0
        results[str(result)] = 0

        assert original_xor == result_xor, f"XOR not preserved for {original}: {original_xor} != {result_xor}"

    print(f"Sorted arrays tested: {len(originals)}")
    print(f"Unique results: {len(results)}")


if __name__ == '__main__':
    original = [2, 7, 6, 5]
    print("Original:", original)

    canonical, index_mapping = HelperLogic.canonicalize_state(original)
    print("Canonical:", canonical)
    print("Index Mapping:", index_mapping)

    reduced = HelperLogic.reduce_state(canonical.copy())
    print("Reduced:", reduced)

    test_over_range(7, 4)
