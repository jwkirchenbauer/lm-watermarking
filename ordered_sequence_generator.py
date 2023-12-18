import poseidon

class OrderedSequenceGenerator:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        # Initialize Poseidon hash function (assuming pre-generated parameters)
        self.poseidon, t = poseidon.parameters.case_simple()

    def generate_ordered_sequence(self):
        # Generate hash for each number and store in a list with its index
        indexed_hashes = [(i, self.poseidon.run_hash([i])) for i in range(self.vocab_size)]

        # This lambda function tells the `sorted` function to sort the list 
        # based on the second element of each list in `indexed_hashes`.
        sorted_indexes = sorted(indexed_hashes, key=lambda x: x[1])

        # Extract the sorted indices
        ordered_sequence = [index for index, t in sorted_indexes]

        return ordered_sequence

# Example usage
if __name__ == "__main__":
    vocab_size = 100  # example vocabulary size
    generator = OrderedSequenceGenerator(vocab_size)
    ordered_sequence = generator.generate_ordered_sequence()
    print(ordered_sequence)
