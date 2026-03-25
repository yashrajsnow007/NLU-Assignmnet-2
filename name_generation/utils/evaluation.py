def evaluate(generator_fn, model, names, save_path=None, num_samples=200):
    generated = [generator_fn(model) for _ in range(num_samples)]

    train_set = set([n[1:-1] for n in names])
    unique = set(generated)

    novelty = len([g for g in generated if g not in train_set]) / len(generated)
    diversity = len(unique) / len(generated)

    if save_path:
        with open(save_path, "w") as f:
            f.write("Generated Names:\n")
            for g in generated:
                f.write(g + "\n")

            f.write("\n---\n")
            f.write(f"Novelty: {novelty}\n")
            f.write(f"Diversity: {diversity}\n")

    return novelty, diversity, generated