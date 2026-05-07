import gc
import random
from multiprocessing import cpu_count, get_context, Pool
from pathlib import Path
from typing import Dict, List, Optional

from tqdm import tqdm
import numpy as np
import pandas as pd
try:
    from anarci import anarci
except ImportError:
    anarci = None
    print("Warning: ANARCI not found. Germline mapping functions will not work.")
from abnumber import Chain
from promb import init_db
from iglm import IgLM

from .sequence import backtranslate, remove_spaces, translate_sequence

# V gene sequences for Koenig et al. obtained from DASM
IGHV3_23_04_SEQ = """GAGGTGCAGCTGGTGGAGTCTGGGGGAGGCTTGGTACAGCCTGGGGGGTCCCTGAGACTCTCCTGTGCAGCCTCTGGATTCACCTTTAGCAGCTATGCCATGAGCTGGGTCCGCCAGGCTCCAGGGAAGGGGCTGGAGTGGGTCTCAGCTATTAGTGGTAGTGGTGGTAGCACATACTACGCAGACTCCGTGAAGGGCCGGTTCACCATCTCCAGAGACAATTCCAAGAACACGCTGTATCTGCAAATGAACAGCCTGAGAGCCGAGGACACGGCCGTATATTACTGTGCGAAAGA"""
IGKV1_39_01_SEQ = """GACATCCAGATGACCCAGTCTCCATCCTCCCTGTCTGCATCTGTAGGAGACAGAGTCACCATCACTTGCCGGGCAAGTCAGAGCATTAGCAGCTATTTAAATTGGTATCAGCAGAAACCAGGGAAAGCCCCTAAGCTCCTGATCTATGCTGCATCCAGTTTGCAAAGTGGGGTCCCATCAAGGTTCAGTGGCAGTGGATCTGGGACAGATTTCACTCTCACCATCAGCAGTCTGCAACCTGAAGATTTTGCAACTTACTACTGTCAACAGAGTTACAGTACCCCTCC"""

# V and J Gene sequences for Koenig et al. obtained from Aakarsh
KOENIG_IGH_CON_SEQ = """GAGGTGCAGCTGGTGGAGTCTGGGGGAGGCTTGGTACAGCCTGGGGGGTCCCTGAGACTCTCCTGTGCAGCCTCTGGATTCACCATTAGCGACTATTGGATACACTGGGTCCGCCAGGCTCCAGGGAAGGGGCTGGAGTGGGTCGCAGGTATTACTCCTGCTGGTGGTTACACATACTACGCAGACTCCGTGAAGGGCCGGTTCACCATCTCCGCAGACACTTCCAAGAACACGGCGTATCTGCAAATGAACAGCCTGAGAGCCGAGGACACGGCCGTATATTACTGTGCGAGATTCGTGTTCTTCCTGCCCTACGCCATGGACTACTGGGGCCAGGGAACCCTGGTCACCGTCTCCTCA"""
KOENIG_IGK_CON_SEQ = """GACATCCAGATGACCCAGTCTCCATCCTCCCTGTCTGCATCTGTAGGAGACAGAGTCACCATCACTTGCCGGGCAAGTCAGGACGTTAGCACCGCTGTAGCTTGGTATCAGCAGAAACCAGGGAAAGCCCCTAAGCTCCTGATCTATTCTGCATCCTTTTTGTATAGTGGGGTCCCATCAAGGTTCAGTGGCAGTGGATCTGGGACAGATTTCACTCTCACCATCAGCAGTCTGCAACCTGAAGATTTTGCAACTTACTACTGTCAACAGAGTTACACTACCCCTCCCACGTTCGGCCAAGGGACCAAGGTGGAAATCAAACGT"""

# MAP FROM CHAIN+SPECIES TO GERMLINE SEQUENCES FASTA
GERMLINE_FASTA_MAP = {
    "H_human": {
        "v_gene": "IGHV.fasta",
        "j_gene": "IGHJ.fasta",
        "d_gene": "IGHD.fasta",
    },
    "K_human": {
        "v_gene": "IGKV.fasta",
        "j_gene": "IGKJ.fasta",
    },
    "L_human": {
        "v_gene": "IGLV.fasta",
        "j_gene": "IGLJ.fasta",
    }
}


def _parse_fasta_to_dict(fasta_content: str) -> Dict[str, str]:
    """Parse FASTA content and return dict of gene_name -> amino_acid_sequence.

    Translates nucleotide sequences to amino acids, removing dots/gaps.
    """
    gene_dict = {}
    lines = fasta_content.strip().split('\n')
    current_gene = None
    current_seq = []

    for line in lines:
        if line.startswith('>'):
            # Save previous entry if exists
            if current_gene and current_seq:
                nt_seq = ''.join(current_seq).replace('.', '').replace('-', '')
                # Truncate to codon boundary
                nt_seq = nt_seq[:len(nt_seq) - len(nt_seq) % 3]
                try:
                    aa_seq = translate_sequence(nt_seq)
                    gene_dict[current_gene] = aa_seq
                except ValueError:
                    # Skip sequences with stop codons or invalid sequences
                    pass

            # Parse gene name from header (e.g., "IGHV1-18*01" from ">X60503|IGHV1-18*01|...")
            parts = line.split('|')
            if len(parts) >= 2:
                current_gene = parts[1]
            else:
                current_gene = line[1:].split()[0]
            current_seq = []
        else:
            current_seq.append(line.strip())

    # Don't forget the last entry
    if current_gene and current_seq:
        nt_seq = ''.join(current_seq).replace('.', '').replace('-', '')
        nt_seq = nt_seq[:len(nt_seq) - len(nt_seq) % 3]
        try:
            aa_seq = translate_sequence(nt_seq)
            gene_dict[current_gene] = aa_seq
        except ValueError:
            pass

    return gene_dict


def _get_anarci_mapping(sequence: str, scheme: str = "imgt") -> tuple:
    """Run ANARCI and return position mapping and germline gene assignments.

    Args:
        sequence: Amino acid sequence
        scheme: Numbering scheme (default: "imgt")

    Returns:
        Tuple of (position_map, gene_names) where:
        - position_map: dict mapping (position, insertion) -> residue
        - gene_names: tuple of (v_gene_name, j_gene_name)
        Returns (None, None) if sequence is not a valid antibody
    """
    results = anarci([("query", sequence)], scheme=scheme, output=False, assign_germline=True, allowed_species=["human"])

    numbering_list = results[0]
    details_list = results[1]

    if not numbering_list or not numbering_list[0]:
        return None, None

    # Extract numbering: [((position, insertion), residue), ...]
    numbering = numbering_list[0][0][0]

    # Extract gene assignments
    germline_info = details_list[0][0]
    v_gene_info = germline_info["germlines"].get("v_gene", None)
    j_gene_info = germline_info["germlines"].get("j_gene", None)

    if not v_gene_info or not j_gene_info:
        return None, None

    # Extract gene names from format: [(species, gene_name), score]
    v_gene_name = v_gene_info[0][1] if isinstance(v_gene_info[0], tuple) else v_gene_info[0]
    j_gene_name = j_gene_info[0][1] if isinstance(j_gene_info[0], tuple) else j_gene_info[0]

    # Convert numbering list to dict for O(1) lookup, excluding gaps
    position_map = {pos_tuple: residue for pos_tuple, residue in numbering if residue != '-'}

    return position_map, (v_gene_name, j_gene_name)


def get_closest_germline(seq: str, scheme="imgt") -> str:
    """Get the closest germline sequence to the input antibody amino acid sequence.

    This function uses the standard approach of:
    1. Running ANARCI to identify closest V/J genes and number the sequence in IMGT scheme
    2. For each IMGT position:
       - Use germline V gene for positions < 105 (Framework and CDR1/2)
       - Use query sequence for positions 105-117 (CDR3, the hypervariable region)
       - Use germline J gene for positions > 117 (Framework 4)

    This is the standard method for germline reversion / UCA reconstruction in antibody engineering.

    Args:
        seq: Input antibody amino acid sequence (heavy or light chain)
        scheme: Numbering scheme for ANARCI (default: "imgt")

    Returns:
        Germline-reverted amino acid sequence, or original sequence if mapping fails
    """
    # Step 1: Get ANARCI mapping for the query sequence
    query_map, genes = _get_anarci_mapping(seq, scheme=scheme)

    if not query_map:
        print("Error: Not a valid antibody sequence")
        return seq

    v_gene_name, j_gene_name = genes

    # Step 2: Load germline sequences from FASTA files
    module_dir = Path(__file__).parent

    try:
        # Determine which FASTA files to use based on ANARCI's gene assignment
        # Extract chain type from gene name (e.g., "IGHV" -> "H", "IGKV" -> "K", "IGLV" -> "L")
        if v_gene_name.startswith("IGHV"):
            chain_type = "H"
        elif v_gene_name.startswith("IGKV"):
            chain_type = "K"
        elif v_gene_name.startswith("IGLV"):
            chain_type = "L"
        else:
            print(f"Unknown gene type: {v_gene_name}")
            return seq

        germline_v_fasta = GERMLINE_FASTA_MAP[f"{chain_type}_human"]["v_gene"]
        germline_j_fasta = GERMLINE_FASTA_MAP[f"{chain_type}_human"]["j_gene"]

        with open(module_dir / "data" / germline_v_fasta, "r") as f:
            v_gene_dict = _parse_fasta_to_dict(f.read())
        with open(module_dir / "data" / germline_j_fasta, "r") as f:
            j_gene_dict = _parse_fasta_to_dict(f.read())

    except Exception as e:
        print(f"Error loading germline FASTA files: {e}")
        return seq

    # Check if genes exist in database
    if v_gene_name not in v_gene_dict:
        print(f"V gene {v_gene_name} not found in FASTA")
        return seq
    if j_gene_name not in j_gene_dict:
        print(f"J gene {j_gene_name} not found in FASTA")
        return seq

    # Step 3: Get ANARCI mapping for the germline sequences
    # Note: Germline sequences from IMGT FASTA files may be incomplete (V genes end ~position 104)
    # So we map them with ANARCI to get their IMGT positions
    v_germline_map, _ = _get_anarci_mapping(v_gene_dict[v_gene_name], scheme=scheme)
    j_germline_map, _ = _get_anarci_mapping(j_gene_dict[j_gene_name], scheme=scheme)

    # If germline mapping fails, fall back to simple sequential mapping
    # This handles cases where germline sequences are too short to be recognized
    if not v_germline_map:
        # Map V gene sequentially starting from position 1
        v_germline_seq = v_gene_dict[v_gene_name]
        v_germline_map = {(i+1, ' '): aa for i, aa in enumerate(v_germline_seq)}

    if not j_germline_map:
        # Map J gene sequentially starting from position 118 (typical J start)
        j_germline_seq = j_gene_dict[j_gene_name]
        j_germline_map = {(118+i, ' '): aa for i, aa in enumerate(j_germline_seq)}

    # Step 4: Reconstruct sequence using standard logic
    # Get all positions that appear in query (to maintain original length/positions)
    all_positions = sorted(query_map.keys())

    reconstructed = []

    for pos_tuple in all_positions:
        pos_num = pos_tuple[0]  # IMGT position number

        if pos_num < 105:
            # V region (FR1, CDR1, FR2, CDR2, FR3): Use germline
            residue = v_germline_map.get(pos_tuple, query_map.get(pos_tuple))
        elif 105 <= pos_num <= 117:
            # CDR3 region: Use query sequence (preserve hypervariable region)
            residue = query_map.get(pos_tuple)
        else:  # pos_num > 117
            # J region (FR4): Use germline
            residue = j_germline_map.get(pos_tuple, query_map.get(pos_tuple))

        if residue:
            reconstructed.append(residue)

    return "".join(reconstructed)


def generate_naive_sequence(seq: str, scheme: str = "imgt", seed: Optional[int] = None) -> str:
    """Generate a truly naive antibody sequence through simulated V-D-J recombination.

    This function creates a 'naive' antibody that shares germline V/J genes with the input
    but generates a de novo CDR3 region via simulated V-D-J recombination. This represents
    what a new B cell would look like if it used the same V/J genes but underwent fresh
    recombination.

    Differences from get_closest_germline():
    - get_closest_germline(): Preserves the input's CDR3 (reverts somatic mutations only)
    - generate_naive_sequence(): Creates a completely new CDR3 via V-D-J recombination

    Algorithm:
    1. Use germline V/J genes for frameworks and CDR1/2 (same as get_closest_germline)
    2. Generate new CDR3 by concatenating:
       - C-terminal end of V gene (IMGT ~105-110)
       - Random D gene (heavy chain only)
       - N-terminal start of J gene (IMGT ~111-117)
    3. Randomize any N/C-terminal regions outside the V-J domain

    Args:
        seq: Input antibody amino acid sequence (heavy or light chain)
        scheme: Numbering scheme for ANARCI (default: "imgt")
        seed: Random seed for reproducibility (optional)

    Returns:
        Naive antibody sequence with germline frameworks and de novo CDR3
    """
    if seed is not None:
        random.seed(seed)

    # Standard amino acid vocabulary
    AA_VOCAB = "ACDEFGHIKLMNPQRSTVWY"

    # Step 1: Get ANARCI mapping for the query sequence
    query_map, genes = _get_anarci_mapping(seq, scheme=scheme)

    if not query_map:
        print("Error: Not a valid antibody sequence")
        return seq

    v_gene_name, j_gene_name = genes

    # Step 2: Load germline sequences from FASTA files
    module_dir = Path(__file__).parent

    try:
        # Determine chain type from gene name
        if v_gene_name.startswith("IGHV"):
            chain_type = "H"
        elif v_gene_name.startswith("IGKV"):
            chain_type = "K"
        elif v_gene_name.startswith("IGLV"):
            chain_type = "L"
        else:
            print(f"Unknown gene type: {v_gene_name}")
            return seq

        germline_v_fasta = GERMLINE_FASTA_MAP[f"{chain_type}_human"]["v_gene"]
        germline_j_fasta = GERMLINE_FASTA_MAP[f"{chain_type}_human"]["j_gene"]

        with open(module_dir / "data" / germline_v_fasta, "r") as f:
            v_gene_dict = _parse_fasta_to_dict(f.read())
        with open(module_dir / "data" / germline_j_fasta, "r") as f:
            j_gene_dict = _parse_fasta_to_dict(f.read())

        # Load D genes if heavy chain
        d_gene_dict = {}
        if chain_type == "H":
            germline_d_fasta = GERMLINE_FASTA_MAP[f"{chain_type}_human"].get("d_gene", None)
            if germline_d_fasta:
                with open(module_dir / "data" / germline_d_fasta, "r") as f:
                    d_gene_dict = _parse_fasta_to_dict(f.read())

    except Exception as e:
        print(f"Error loading germline FASTA files: {e}")
        return seq

    # Check if genes exist in database
    if v_gene_name not in v_gene_dict:
        print(f"V gene {v_gene_name} not found in FASTA")
        return seq
    if j_gene_name not in j_gene_dict:
        print(f"J gene {j_gene_name} not found in FASTA")
        return seq

    # Step 3: Get ANARCI mapping for the germline sequences
    v_germline_map, _ = _get_anarci_mapping(v_gene_dict[v_gene_name], scheme=scheme)
    j_germline_map, _ = _get_anarci_mapping(j_gene_dict[j_gene_name], scheme=scheme)

    # Fallback if germline mapping fails
    if not v_germline_map:
        v_germline_seq = v_gene_dict[v_gene_name]
        v_germline_map = {(i+1, ' '): aa for i, aa in enumerate(v_germline_seq)}

    if not j_germline_map:
        j_germline_seq = j_gene_dict[j_gene_name]
        j_germline_map = {(118+i, ' '): aa for i, aa in enumerate(j_germline_seq)}

    # Step 4: Generate CDR3 via V-D-J recombination
    # Count how many CDR3 positions exist in the input
    cdr3_positions = [pos_tuple for pos_tuple in query_map.keys() if 105 <= pos_tuple[0] <= 117]
    target_cdr3_length = len(cdr3_positions)

    # Build CDR3 from V-end + D-gene + J-start
    cdr3_parts = []

    # V-gene C-terminal end (positions 105-110, typically ~6 residues)
    v_cdr3_positions = [pt for pt in v_germline_map.keys() if 105 <= pt[0] <= 110]
    v_cdr3_seq = ''.join([v_germline_map[pt] for pt in sorted(v_cdr3_positions)])
    cdr3_parts.append(v_cdr3_seq)

    # D-gene (heavy chain only, randomly selected)
    if chain_type == "H" and d_gene_dict:
        d_gene_seq = random.choice(list(d_gene_dict.values()))
        cdr3_parts.append(d_gene_seq)

    # J-gene N-terminal start (positions 111-117, typically ~7 residues)
    j_cdr3_positions = [pt for pt in j_germline_map.keys() if 111 <= pt[0] <= 117]
    j_cdr3_seq = ''.join([j_germline_map[pt] for pt in sorted(j_cdr3_positions)])
    cdr3_parts.append(j_cdr3_seq)

    # Concatenate and adjust to target length
    new_cdr3 = ''.join(cdr3_parts)

    if len(new_cdr3) > target_cdr3_length:
        # Trim from the middle (D-gene region) to preserve V/J ends
        new_cdr3 = new_cdr3[:target_cdr3_length]
    elif len(new_cdr3) < target_cdr3_length:
        # Pad with random amino acids in the middle (simulating N-additions)
        pad_length = target_cdr3_length - len(new_cdr3)
        mid_point = len(new_cdr3) // 2
        random_insertions = ''.join(random.choices(AA_VOCAB, k=pad_length))
        new_cdr3 = new_cdr3[:mid_point] + random_insertions + new_cdr3[mid_point:]

    # Step 5: Reconstruct full sequence
    all_positions = sorted(query_map.keys())
    reconstructed = []
    cdr3_idx = 0

    for pos_tuple in all_positions:
        pos_num = pos_tuple[0]

        if pos_num < 105:
            # V region: Use germline
            residue = v_germline_map.get(pos_tuple, query_map.get(pos_tuple))
        elif 105 <= pos_num <= 117:
            # CDR3: Use newly generated sequence
            if cdr3_idx < len(new_cdr3):
                residue = new_cdr3[cdr3_idx]
                cdr3_idx += 1
            else:
                residue = query_map.get(pos_tuple)
        else:  # pos_num > 117
            # J region: Use germline
            residue = j_germline_map.get(pos_tuple, query_map.get(pos_tuple))

        if residue:
            reconstructed.append(residue)

    naive_seq = "".join(reconstructed)

    # Step 6: Randomize N/C-terminal regions outside the antibody domain
    # Find where the ANARCI domain is in the original sequence
    anarci_domain = ''.join([query_map[pt] for pt in all_positions])
    domain_start = seq.find(anarci_domain)

    if domain_start == -1:
        # Can't locate domain, return as-is
        return naive_seq

    # Replace N-terminal region
    if domain_start > 0:
        n_term_random = ''.join(random.choices(AA_VOCAB, k=domain_start))
        naive_seq = n_term_random + naive_seq

    # Replace C-terminal region
    domain_end = domain_start + len(anarci_domain)
    if domain_end < len(seq):
        c_term_length = len(seq) - domain_end
        c_term_random = ''.join(random.choices(AA_VOCAB, k=c_term_length))
        naive_seq = naive_seq + c_term_random

    return naive_seq


def sample_naive_sequence_from_oas(oas_naive_fasta_path: str, seed: Optional[int] = None) -> str:
    """Sample a random naive antibody sequence from an OAS naive repertoire FASTA file.
    Args:
        oas_naive_fasta_path: Path to OAS naive repertoire FASTA file
        seed: Random seed for reproducibility (optional)
    Returns:
        Randomly sampled naive antibody amino acid sequence
    """
    if seed is not None:
        random.seed(seed)
    # read fasta file in list of strings using biopython
    records = list(SeqIO.parse(oas_naive_fasta_path, "fasta"))
    if not records:
        raise ValueError(f"No sequences found in FASTA file: {oas_naive_fasta_path}")
    sampled_record = random.choice(records)
    return str(sampled_record.seq)


def _sample_cdr3_length(chain_type: str, seed: Optional[int] = None) -> int:
    """Sample CDR3 length from biologically realistic distributions.

    Uses observed CDR3 length distributions from naive human antibody repertoires.

    Args:
        chain_type: "H" (heavy), "K" (kappa), or "L" (lambda)
        seed: Random seed for reproducibility

    Returns:
        CDR3 length in amino acids

    Distribution sources:
        - Heavy: mean=15±4 aa (range 8-24 common)
        - Kappa: mode=9 aa (70% of sequences), range 8-11
        - Lambda: mode=11 aa (47% of sequences), range 9-13
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    if chain_type == "H":
        # Heavy chain: Normal distribution, mean=15, std=4
        # Constrain to biological range [8, 24]
        length = int(np.random.normal(15, 4))
        length = max(8, min(24, length))
    elif chain_type == "K":
        # Kappa light chain: 70% are 9 aa, rest distributed 8-11
        if np.random.random() < 0.70:
            length = 9
        else:
            length = np.random.choice([8, 10, 11])
    elif chain_type == "L":
        # Lambda light chain: 47% are 11 aa, rest distributed 9-13
        if np.random.random() < 0.47:
            length = 11
        else:
            length = np.random.choice([9, 10, 12, 13])
    else:
        raise ValueError(f"Unknown chain_type: {chain_type}. Must be 'H', 'K', or 'L'")

    return length


def assemble_random_germline_antibody(
    chain_type: str = "heavy",
    v_gene: Optional[str] = None,
    d_gene: Optional[str] = None,
    j_gene: Optional[str] = None,
    seed: Optional[int] = None,
    scheme: str = "imgt"
) -> str:
    """Assemble a random germline antibody sequence from human germline gene segments.

    This function generates realistic naive antibody sequences by randomly selecting
    germline V, D (heavy only), and J genes and assembling them with biologically
    realistic CDR3 lengths. This is the third complementary method alongside
    get_closest_germline() and generate_naive_sequence().

    Key differences from existing methods:
    - get_closest_germline(): Maps input sequence to germline, PRESERVES original CDR3
    - generate_naive_sequence(): Uses input's V/J genes, GENERATES new CDR3
    - assemble_random_germline_antibody(): RANDOM V/D/J genes, RANDOM CDR3 from biological distribution

    Algorithm:
    1. Randomly select germline genes (or use specified genes)
       - Uniform random selection from available genes in IMGT database
       - Heavy: V from IGHV.fasta, D from IGHD.fasta, J from IGHJ.fasta
       - Kappa: V from IGKV.fasta, J from IGKJ.fasta
       - Lambda: V from IGLV.fasta, J from IGLJ.fasta

    2. Sample target CDR3 length from biological distributions
       - Heavy: Normal(mean=15, std=4), range [8, 24] aa
       - Kappa: 70% → 9 aa, 30% → [8, 10, 11] aa
       - Lambda: 47% → 11 aa, 53% → [9, 10, 12, 13] aa

    3. Assemble CDR3 region (positions 105-117 in IMGT numbering)
       - Extract V gene C-terminal contribution (positions 105-110, ~0-6 aa)
       - Add D gene middle section (heavy only, ~4-8 aa)
       - Extract J gene N-terminal contribution (positions 111-117, ~4-7 aa)
       - Fill gaps with random amino acids to reach target CDR3 length

    4. Build full antibody sequence
       - Framework regions (1-104) from germline V gene
       - Generated CDR3 (positions 105-117)
       - Framework regions (118-128) from germline J gene

    5. Validate with ANARCI (ensure sequence is recognized as valid antibody)

    Args:
        chain_type: Antibody chain type
            - "heavy" or "H": Heavy chain (uses V-D-J recombination)
            - "kappa" or "K": Kappa light chain (uses V-J recombination)
            - "lambda" or "L": Lambda light chain (uses V-J recombination)
        v_gene: Optional V gene name (e.g., "IGHV3-23*01"). If None, randomly selected.
        d_gene: Optional D gene name (e.g., "IGHD3-10*01"). Heavy chain only. If None, randomly selected.
        j_gene: Optional J gene name (e.g., "IGHJ4*02"). If None, randomly selected.
        seed: Random seed for reproducibility. Controls gene selection, CDR3 length sampling, and random amino acids.
        scheme: ANARCI numbering scheme (default: "imgt")

    Returns:
        str: Assembled naive antibody amino acid sequence

    Raises:
        ValueError: If chain_type is invalid or required germline genes are not found

    Examples:
        >>> # Fully random heavy chain antibody
        >>> seq = assemble_random_germline_antibody(chain_type="heavy", seed=42)
        >>> print(f"Generated heavy chain: {seq[:50]}...")
        Generated heavy chain: EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVS...

        >>> # Random kappa light chain with specified V gene
        >>> seq = assemble_random_germline_antibody(
        ...     chain_type="kappa",
        ...     v_gene="IGKV1-39*01",
        ...     seed=123
        ... )

        >>> # Generate diverse library of 10 heavy chains
        >>> library = [
        ...     assemble_random_germline_antibody(chain_type="heavy", seed=i)
        ...     for i in range(10)
        ... ]

    Biological Motivation:
        This function simulates the natural V-D-J recombination process that occurs
        during B cell development in the bone marrow. Each B cell randomly selects
        one V, D (heavy only), and J gene segment from the germline repertoire and
        joins them together to create a unique antigen receptor. The resulting
        sequences represent realistic naive antibodies before affinity maturation
        via somatic hypermutation.

    Use Cases:
        - Generate diverse naive antibody libraries for computational studies
        - Create realistic starting sequences for in silico affinity maturation
        - Sample from the theoretical naive antibody repertoire
        - Generate controls for antibody engineering experiments

    See Also:
        - get_closest_germline(): For germline reversion of mature antibodies
        - generate_naive_sequence(): For generating new CDR3s with same V/J genes
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Standard amino acid vocabulary (20 standard amino acids)
    AA_VOCAB = "ACDEFGHIKLMNPQRSTVWY"

    # Normalize chain_type
    chain_type_map = {"heavy": "H", "H": "H", "kappa": "K", "K": "K", "lambda": "L", "L": "L"}
    if chain_type not in chain_type_map:
        raise ValueError(f"Invalid chain_type '{chain_type}'. Must be 'heavy'/'H', 'kappa'/'K', or 'lambda'/'L'")

    chain_type = chain_type_map[chain_type]

    # Load germline gene databases
    module_dir = Path(__file__).parent

    try:
        germline_fastas = GERMLINE_FASTA_MAP[f"{chain_type}_human"]

        # Load V genes
        with open(module_dir / "data" / germline_fastas["v_gene"], "r") as f:
            v_gene_dict = _parse_fasta_to_dict(f.read())

        # Load J genes
        with open(module_dir / "data" / germline_fastas["j_gene"], "r") as f:
            j_gene_dict = _parse_fasta_to_dict(f.read())

        # Load D genes (heavy chain only)
        d_gene_dict = {}
        if chain_type == "H":
            with open(module_dir / "data" / germline_fastas["d_gene"], "r") as f:
                d_gene_dict = _parse_fasta_to_dict(f.read())

    except Exception as e:
        raise ValueError(f"Error loading germline FASTA files for chain type '{chain_type}': {e}")

    # Select genes (random if not specified)
    if v_gene is None:
        v_gene = random.choice(list(v_gene_dict.keys()))
    if v_gene not in v_gene_dict:
        raise ValueError(f"V gene '{v_gene}' not found in database. Available genes: {list(v_gene_dict.keys())[:10]}...")

    if j_gene is None:
        j_gene = random.choice(list(j_gene_dict.keys()))
    if j_gene not in j_gene_dict:
        raise ValueError(f"J gene '{j_gene}' not found in database. Available genes: {list(j_gene_dict.keys())[:10]}...")

    if chain_type == "H":
        if d_gene is None:
            d_gene = random.choice(list(d_gene_dict.keys()))
        if d_gene not in d_gene_dict:
            raise ValueError(f"D gene '{d_gene}' not found in database. Available genes: {list(d_gene_dict.keys())[:10]}...")

    # Get germline gene sequences
    v_seq = v_gene_dict[v_gene]
    j_seq = j_gene_dict[j_gene]
    d_seq = d_gene_dict[d_gene] if chain_type == "H" else None

    # Map germline genes to IMGT positions
    v_map, _ = _get_anarci_mapping(v_seq, scheme=scheme)
    j_map, _ = _get_anarci_mapping(j_seq, scheme=scheme)

    # Fallback for short germline sequences that ANARCI doesn't recognize
    if not v_map:
        # Simple sequential mapping starting from position 1
        v_map = {(i+1, ' '): aa for i, aa in enumerate(v_seq)}

    if not j_map:
        # Simple sequential mapping starting from position 118 (typical J gene start)
        j_map = {(118+i, ' '): aa for i, aa in enumerate(j_seq)}

    # Sample target CDR3 length from biological distribution
    target_cdr3_length = _sample_cdr3_length(chain_type, seed=seed)

    # Extract germline contributions to CDR3
    # V gene C-terminal (positions 105-110, typically last ~6 aa of V gene)
    v_cdr3_positions = [pt for pt in v_map.keys() if 105 <= pt[0] <= 110]
    v_cdr3_segment = ''.join([v_map[pt] for pt in sorted(v_cdr3_positions)])

    # J gene N-terminal (positions 111-117, typically first ~7 aa of J gene in CDR3)
    j_cdr3_positions = [pt for pt in j_map.keys() if 111 <= pt[0] <= 117]
    j_cdr3_segment = ''.join([j_map[pt] for pt in sorted(j_cdr3_positions)])

    # Assemble CDR3
    if chain_type == "H":
        # Heavy chain: V + D + J with random padding
        # Take middle portion of D gene (typically 4-8 aa)
        d_start = max(0, len(d_seq) // 4) if len(d_seq) > 8 else 0
        d_end = min(len(d_seq), len(d_seq) - len(d_seq) // 4) if len(d_seq) > 8 else len(d_seq)
        d_segment = d_seq[d_start:d_end]

        # Concatenate segments
        cdr3 = v_cdr3_segment + d_segment + j_cdr3_segment
    else:
        # Light chain: V + J with random padding (no D gene)
        cdr3 = v_cdr3_segment + j_cdr3_segment

    # Adjust CDR3 to target length
    if len(cdr3) < target_cdr3_length:
        # Add random amino acids in the middle (simulating N-additions at junctions)
        pad_length = target_cdr3_length - len(cdr3)
        mid_point = len(v_cdr3_segment) + (len(cdr3) - len(v_cdr3_segment) - len(j_cdr3_segment)) // 2
        random_additions = ''.join(random.choices(AA_VOCAB, k=pad_length))
        cdr3 = cdr3[:mid_point] + random_additions + cdr3[mid_point:]
    elif len(cdr3) > target_cdr3_length:
        # Trim from the middle (preserve V and J ends)
        # This simulates exonuclease trimming of the D gene
        excess = len(cdr3) - target_cdr3_length
        trim_start = len(v_cdr3_segment)
        trim_end = len(cdr3) - len(j_cdr3_segment)
        middle_segment = cdr3[trim_start:trim_end]

        if len(middle_segment) > excess:
            # Trim from middle segment
            new_middle_len = len(middle_segment) - excess
            middle_start = (len(middle_segment) - new_middle_len) // 2
            middle_segment = middle_segment[middle_start:middle_start + new_middle_len]
            cdr3 = v_cdr3_segment + middle_segment + j_cdr3_segment
        else:
            # If middle is too short, trim V and J contributions
            cdr3 = cdr3[:target_cdr3_length]

    # Assemble full sequence
    # Extract framework regions from V gene (positions 1-104)
    v_framework_positions = [pt for pt in v_map.keys() if pt[0] < 105]
    v_framework = ''.join([v_map[pt] for pt in sorted(v_framework_positions)])

    # Extract framework regions from J gene (positions 118-128)
    j_framework_positions = [pt for pt in j_map.keys() if pt[0] >= 118]
    j_framework = ''.join([j_map[pt] for pt in sorted(j_framework_positions)])

    # Combine: V_framework + CDR3 + J_framework
    full_sequence = v_framework + cdr3 + j_framework

    # Validate with ANARCI (optional check)
    validation_map, validation_genes = _get_anarci_mapping(full_sequence, scheme=scheme)
    if not validation_map:
        # If ANARCI doesn't recognize it, it might still be valid but unusual
        # We'll return it anyway since it was assembled from valid germline genes
        print(f"Warning: Assembled sequence not recognized by ANARCI. May have unusual structure.")
        print(f"Genes used: V={v_gene}, D={d_gene if d_gene else 'N/A'}, J={j_gene}")

    return full_sequence


def get_cdr(seq: str) -> List[str]:
    cdrs = Chain(remove_spaces([seq])[0], scheme="imgt")
    return [cdrs.cdr1_seq, cdrs.cdr2_seq, cdrs.cdr3_seq]


def get_frs(seq: str) -> List[str]:
    frs = Chain(remove_spaces([seq])[0], scheme="imgt")
    return [frs.fr1_seq, frs.fr2_seq, frs.fr3_seq, frs.fr4_seq]


def create_region_masks(sequence: str, scheme="imgt") -> Dict[str, np.ndarray]:
    """Create boolean masks for antibody regions matching original sequence length."""
    seq = remove_spaces([sequence])[0]
    chain = Chain(seq, scheme=scheme)
    
    region_keys = ["FR1", "CDR1", "FR2", "CDR2", "FR3", "CDR3", "FR4", "CDR_overall", "FR_overall"]
    masks: Dict[str, np.ndarray] = {key: np.zeros(len(seq), dtype=bool) for key in region_keys}

    # Find where chain sequence starts in original
    offset = seq.find(str(chain.seq))
    if offset == -1:
        raise ValueError("Chain sequence not found in original")

    # Map chain positions to original sequence indices
    for i, (pos_name, _) in enumerate(chain.positions.items()):
        seq_idx = offset + i
        region_name = pos_name.get_region()
        
        if region_name in masks:
            masks[region_name][seq_idx] = True
        
        if pos_name.is_in_cdr():
            masks["CDR_overall"][seq_idx] = True
        else:
            masks["FR_overall"][seq_idx] = True
    
    return masks


def _safe_create_region_masks(args):
    """Wrapper for create_region_masks that returns (index, result_or_error)."""
    idx, seq, scheme = args
    try:
        result = create_region_masks(seq, scheme=scheme)
        return idx, result, None
    except Exception as e:
        return idx, None, str(e)


def compute_region_masks_batch(
    sequences: List[str],
    num_workers: Optional[int] = None,
    chunksize: int = 100,
    scheme: str = "imgt",
    show_progress: bool = True,
    raise_on_error: bool = False,
) -> List[Dict[str, np.ndarray]]:
    """Compute region masks for a batch of sequences using multiprocessing.

    Args:
        sequences: List of antibody amino acid sequences (heavy or light chains)
        num_workers: Number of worker processes (defaults to cpu_count())
        chunksize: Number of sequences to process per worker chunk
        scheme: Numbering scheme for abnumber (default: 'imgt')
        show_progress: Show progress bar with tqdm (default: True)
        raise_on_error: Raise exception on first error, otherwise skip failed sequences (default: False)

    Returns:
        List of mask dictionaries, one per input sequence. Each dict maps
        region names to boolean numpy arrays. Failed sequences return None at their index.

    Example:
        >>> heavy_chains = ["EVQLV...", "QVQLQ..."]
        >>> masks = compute_region_masks_batch(heavy_chains, num_workers=8)
        >>> cdr3_mask = masks[0]['CDR3']  # boolean array for first sequence
    """
    if num_workers is None:
        num_workers = cpu_count()

    if len(sequences) == 0:
        return []

    # Single sequence - no need for multiprocessing
    if len(sequences) == 1:
        return [create_region_masks(sequences[0], scheme=scheme)]

    # Prepare work items: (index, sequence, scheme)
    work_items = [(i, seq, scheme) for i, seq in enumerate(sequences)]

    # Initialize results list with None placeholders
    results = [None] * len(sequences)
    error_count = 0

    ctx = get_context("spawn")
    with ctx.Pool(processes=num_workers) as pool:
        # Use imap for progress tracking
        iterator = pool.imap(_safe_create_region_masks, work_items, chunksize=chunksize)

        if show_progress:
            try:
                from tqdm import tqdm

                iterator = tqdm(iterator, total=len(sequences), desc="Computing masks")
            except ImportError:
                print(f"Processing {len(sequences)} sequences with {num_workers} workers...")

        for idx, result, error in iterator:
            if error is not None:
                error_count += 1
                if raise_on_error:
                    raise RuntimeError(f"Error processing sequence {idx}: {error}")
                if show_progress and error_count <= 5:  # Show first 5 errors
                    print(f"\nWarning: Failed to process sequence {idx}: {error}")
            else:
                results[idx] = result

    if error_count > 0:
        print(
            f"\nCompleted with {error_count}/{len(sequences)} failed sequences (returned as None)"
        )

    # clean up memory
    del ctx
    gc.collect()

    return results


def backtranslate_with_v_gene(aa_sequence: str, v_gene_seq: str) -> str:
    """Backtranslate protein sequence using V gene sequence where possible."""
    # Truncate v_gene_seq to codon boundary and translate
    v_gene_seq = v_gene_seq[: len(v_gene_seq) - len(v_gene_seq) % 3]
    v_gene_aa = translate_sequence(v_gene_seq)
    consensus_popular_nt = backtranslate(aa_sequence)

    result = ""
    for i, aa in enumerate(aa_sequence):
        if i < len(v_gene_aa) and aa == v_gene_aa[i]:
            result += v_gene_seq[i * 3 : i * 3 + 3]
        else:
            result += consensus_popular_nt[i * 3 : i * 3 + 3]

    assert translate_sequence(result) == aa_sequence
    return result


def parse_anarci_output(anarci_result, scheme_length=149):
    """
    Parses the raw ANARCI output to construct a fixed-length AHo sequence.
    
    Args:
        anarci_result: The tuple (numbering, alignment_details, hit_tables) for a SINGLE sequence.
        scheme_length: AHo uses 149 positions.
    
    Returns:
        A string of length 149 with the aligned sequence, or None if no domain found.
    """
    numbering, alignment_details, hit_tables = anarci_result

    # numbering is a list of domains found. If empty, no antibody domain was found.
    if not numbering:
        return None

    # We typically take the first domain found (index 0) which is the most significant V-domain
    # domain_numbering is a list of tuples: ((pos_id, insertion_code), residue_char)
    domain_numbering = numbering[0][0][0] 
    
    # Initialize a list of gaps
    aligned_seq = ['-'] * scheme_length

    # AHo numbering in ANARCI typically goes from 1 to 149.
    # The 'pos_id' in ANARCI for AHo is the actual position number.
    for (pos_id, insertion_code), residue in domain_numbering:
        # AHo strictly shouldn't have insertion codes if used correctly for standard Ig,
        # but ANARCI returns them in specific formats. 
        # We assume standard integer positions for the AHo scaffolding.
        
        # Adjust 1-based index to 0-based index
        index = pos_id - 1
        
        if 0 <= index < scheme_length:
            aligned_seq[index] = residue
            
    return "".join(aligned_seq)


def process_chunk(chunk_of_sequences):
    """
    Worker function to process a batch of sequences.
    
    Args:
        chunk_of_sequences: List of tuples [('ID_1', 'SEQ_1'), ('ID_2', 'SEQ_2'), ...]
        
    Returns:
        List of results: [('ID', 'ALIGNED_SEQ'), ...]
    """
    # Run ANARCI on the whole chunk at once (more efficient than 1 by 1)
    # output=False prevents printing to stdout, returns python objects instead
    results = anarci(chunk_of_sequences, scheme="aho", output=False, assign_germline=False)
    
    # Unpack results: results is a tuple (numbering_list, details_list, hit_list)
    numbering_list, _, _ = results
    
    processed_data = []
    
    # Iterate through the original input sequences and match with results
    for i, (seq_id, _) in enumerate(chunk_of_sequences):
        # Extract the specific result for this sequence
        # We reconstruct the tuple structure ANARCI would have returned for a single item
        single_result = (
            [numbering_list[i]] if numbering_list[i] else [], 
            None, 
            None
        )
        
        aligned_seq = parse_anarci_output(single_result)
        
        if aligned_seq:
            processed_data.append((seq_id, aligned_seq))
        else:
            # Handle cases where alignment failed (not an antibody or poor quality)
            processed_data.append((seq_id, None))
            
    return processed_data


def parallel_align_sequences(raw_sequences, n_jobs=None, chunk_size=100, verbose=False):
    """
    Parallel alignment with smooth, per-sequence progress tracking.
    
    Args:
        raw_sequences: List of tuples [('ID', 'SEQ'), ...]
        n_jobs: Number of cores (default: all - 1)
        chunk_size: Number of sequences per worker batch. 
                    Smaller = smoother progress bar but slightly more overhead.
                    100-500 is usually a sweet spot for ANARCI.
    Returns:
        List of tuples [('ID', 'ALIGNED_SEQ'), ...] if input is a single sequence,
        List of tuples [(('ID_1', 'ID_2'), 'ALIGNED_SEQ'), ...] if input is a paired sequence
    """
    if n_jobs is None:
        n_jobs = max(1, cpu_count() - 1)
    
    # 1. Create many fixed-size chunks
    # This ensures that workers report back frequently
    chunks = [raw_sequences[i:i + chunk_size] for i in range(0, len(raw_sequences), chunk_size)]
    
    if verbose:
        print(f"Aligning {len(raw_sequences)} sequences on {n_jobs} cores...")
        print(f"Work split into {len(chunks)} batches (approx {chunk_size} seqs/batch).")
    
    results = []
    
    # 2. Setup the Pool and Progress Bar
    # total=len(raw_sequences) lets the bar represent ACTUAL sequences, not chunks
    pbar = tqdm(total=len(raw_sequences), unit="seq", desc="ANARCI Alignment")
    
    with Pool(processes=n_jobs) as pool:
        # 3. Use imap_unordered
        # imap_unordered yields results as soon as *any* worker finishes a batch.
        # This prevents the progress bar from stalling if chunk #1 is slower than chunk #2.
        for batch_result in pool.imap_unordered(process_chunk, chunks):
            
            # Aggregate results
            results.extend(batch_result)
            
            # Update progress bar by the ACTUAL number of sequences in this batch
            pbar.update(len(batch_result))
            
    pbar.close()
    
    return results


def compute_oasis_humanness(sequences: List[str], **kwargs) -> List[float]:
    """
    Computes OASis-like humanness score (% of peptides that are human, exact match)
    against a pre-defined Human OAS DB (antibody peptides found in >=10% of subjects)
    """
    db = init_db('human-oas')
    return [db.compute_peptide_content(seq) for seq in tqdm(sequences, desc="Computing OASis humanness")]


def compute_iglm_humanness(sequences: List[str], chain: str = "heavy", species_token: str = "[HUMAN]", **kwargs) -> List[float]:
    """
    Computes log-likelihood of sequence under the IGLM model, which was trained on human antibody sequences.
    """
    iglm = IgLM()
    chain_token = "[HEAVY]" if chain == "heavy" else "[LIGHT]"
    return [iglm.log_likelihood(seq, chain_token, species_token) for seq in tqdm(sequences, desc="Computing IGLM humanness")]


# --- usage example ---
if __name__ == "__main__":
    # Example raw sequences (Heavy and Light)
    # Note: AHo works for both Heavy and Light chains without changing parameters
    raw_input = [
        ("H_chain_1", "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAKDGYYYYGLDVWGQGTTVTVSS"),
        ("L_chain_1", "DIQMTQSPSSLSASVGDRVTITCRASQGISNYLAWYQQKPGKVPKLLIYAASTLQSGVPSRFSGSGSGTDFTLTISSLQPEDVATYYCQKYNSAPLTFGGGTKVEIK"),
        ("Trastuzumab_H", "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCSRWGGDGFYAMDYWGQGTLVTVSS"),
    ]

    # Run alignment
    aligned_data = parallel_align_sequences(raw_input)
    
    # Run alignment without parallelization
    # aligned_data = process_chunk(raw_input)

    # Display
    print("\n--- Results ---")
    for seq_id, seq_aho in aligned_data:
        if seq_aho:
            print(f">{seq_id}\n{seq_aho} (Length: {len(seq_aho)})")
        else:
            print(f">{seq_id}\nAlignment Failed")

    # Optional: Convert to DataFrame for easy saving
    df = pd.DataFrame(aligned_data, columns=['Id', 'AHo_Sequence'])
    print(df.head())

    breakpoint()