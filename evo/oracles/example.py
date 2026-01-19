"""Quick example of using protein fitness oracles."""

from evo.oracles import get_oracle


def main():
    """Demonstrate basic oracle usage."""

    print("=" * 80)
    print("PROTEIN FITNESS ORACLES - QUICK EXAMPLE")
    print("=" * 80)

    # # 1. LLM Oracle
    # print("\n1. LLM Oracle (clone)")
    # print("-" * 80)
    # oracle_llm = get_oracle("clone")
    # print(f"Chain type: {oracle_llm.chain_type}")
    # print(f"Higher is better: {oracle_llm.higher_is_better}")
    # print(f"Seed fitnesses: {oracle_llm.seed_fitnesses}")

    # # 2. Random Oracle
    # print("\n2. Random Oracle (rand_0.5)")
    # print("-" * 80)
    # oracle_rand = get_oracle("rand_0.5")
    # print(f"Chain type: {oracle_rand.chain_type}")
    # print(f"Higher is better: {oracle_rand.higher_is_better}")
    # print(f"Seed fitnesses: {oracle_rand.seed_fitnesses}")

    # 3. COVID Oracle - SARSCoV1 (CPU mode)
    print("\n3. COVID Oracle - SARSCoV1")
    print("-" * 80)
    oracle_cov1 = get_oracle("SARSCoV1", device="cuda", use_iglm_weighting=False, enable_mc_dropout=True)
    print(f"Chain type: {oracle_cov1.chain_type}")
    print(f"Higher is better: {oracle_cov1.higher_is_better}")
    print(f"Device: {oracle_cov1.device}")
    print(f"Seed fitnesses: {oracle_cov1.seed_fitnesses}")
    print(oracle_cov1.predict(oracle_cov1.seed_sequences[0]))
    print(oracle_cov1.predict_with_uncertainty(oracle_cov1.seed_sequences[0]))

    # 4. COVID Oracle - SARSCoV2Beta (CPU mode)
    print("\n4. COVID Oracle - SARSCoV2Beta")
    print("-" * 80)
    oracle_cov2 = get_oracle("SARSCoV2Beta", device="cuda", use_iglm_weighting=False, enable_mc_dropout=True)
    print(f"Chain type: {oracle_cov2.chain_type}")
    print(f"Higher is better: {oracle_cov2.higher_is_better}")
    print(f"Device: {oracle_cov2.device}")
    print(f"Seed fitnesses: {oracle_cov2.seed_fitnesses}")
    print(oracle_cov2.predict(oracle_cov2.seed_sequences[0]))
    print(oracle_cov2.predict_with_uncertainty(oracle_cov2.seed_sequences[0]))

    # Demonstrate precomputed fitness lookup
    print("\n" + "=" * 80)
    print("PRECOMPUTED FITNESS LOOKUP (No inference needed!)")
    print("=" * 80)
    # print(f"LLM seed_0:        {oracle_llm.get_seed_fitness('seed_0'):.6f}")
    # print(f"Random seed_0:     {oracle_rand.get_seed_fitness('seed_0'):.6f}")
    print(f"SARSCoV1 seed_0:   {oracle_cov1.get_seed_fitness('seed_0'):.6f}")
    print(f"SARSCoV2Beta seed_0: {oracle_cov2.get_seed_fitness('seed_0'):.6f}")

    print("\n" + "=" * 80)
    print("✅ All oracles loaded successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
