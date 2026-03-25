# Compute novelty and diversity metrics for generated names
def evaluate(generator_fn, model, names, save_path=None, num_samples=200):
    # Generate sample names from model
    generated = [generator_fn(model) for _ in range(num_samples)]

    # Remove start/end markers from training names for comparison
    train_set = set([n[1:-1] for n in names])
    # Count unique generated names
    unique = set(generated)

    # Novelty: fraction of generated names not in training set
    novelty = len([g for g in generated if g not in train_set]) / len(generated)
    # Diversity: fraction of unique names generated
    diversity = len(unique) / len(generated)

    # Save results if path provided
    if save_path:
        with open(save_path, "w") as f:
            f.write("Generated Names:\n")
            for g in generated:
                f.write(g + "\n")

            f.write("\n---\n")
            f.write(f"Novelty: {novelty}\n")
            f.write(f"Diversity: {diversity}\n")

    return novelty, diversity, generated