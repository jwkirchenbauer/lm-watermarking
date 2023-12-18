import poseidon

def generate_hashes():
    # Initialize Poseidon with pre-generated parameters
    poseidon_simple, t = poseidon.parameters.case_simple()

    # Generate and print 100 hash values
    for i in range(1, 101):
        input_vec = [i] * t
        poseidon_digest = poseidon_simple.run_hash(input_vec)
        print(f"Input {i}: Hash = {hex(int(poseidon_digest))}")

if __name__ == "__main__":
    generate_hashes()
