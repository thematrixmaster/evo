import re
from itertools import product, zip_longest
from typing import List, Tuple

import numpy as np
import pandas as pd
from Bio.Data import CodonTable
from Bio.Seq import Seq

_FASTA_VOCAB = "ARNDCQEGHILKMFPSTWYV"
_AA_STR_SORTED = "ACDEFGHIKLMNPQRSTVWY"
_DNA_VOCAB = "ACGT"

# Add additional tokens to this string:
RESERVED_TOKENS = "^"
# Each token in RESERVED_TOKENS will appear once in aa strings, and three times in nt strings.
RESERVED_TOKEN_TRANSLATIONS = {token * 3: token for token in RESERVED_TOKENS}

# Human codon usage frequencies from:
# https://www.kazusa.or.jp/codon/cgi-bin/showcodon.cgi?species=9606&aa=1
# Values are per thousand
HUMAN_CODON_USAGE = {
    "TTT": 17.6,
    "TTC": 20.3,  # Phe
    "TTA": 7.7,
    "TTG": 12.9,  # Leu
    "CTT": 13.2,
    "CTC": 19.6,  # Leu
    "CTA": 7.2,
    "CTG": 39.6,  # Leu
    "ATT": 16.0,
    "ATC": 20.8,  # Ile
    "ATA": 7.5,  # Ile
    "ATG": 22.0,  # Met
    "GTT": 11.0,
    "GTC": 14.5,  # Val
    "GTA": 7.1,
    "GTG": 28.1,  # Val
    "TCT": 15.2,
    "TCC": 17.7,  # Ser
    "TCA": 12.2,
    "TCG": 4.4,  # Ser
    "CCT": 17.5,
    "CCC": 19.8,  # Pro
    "CCA": 16.9,
    "CCG": 6.9,  # Pro
    "ACT": 13.1,
    "ACC": 18.9,  # Thr
    "ACA": 15.1,
    "ACG": 6.1,  # Thr
    "GCT": 18.4,
    "GCC": 27.7,  # Ala
    "GCA": 15.8,
    "GCG": 7.4,  # Ala
    "TAT": 12.2,
    "TAC": 15.3,  # Tyr
    "TAA": 1.0,
    "TAG": 0.8,  # Stop
    "CAT": 10.9,
    "CAC": 15.1,  # His
    "CAA": 12.3,
    "CAG": 34.2,  # Gln
    "AAT": 17.0,
    "AAC": 19.1,  # Asn
    "AAA": 24.4,
    "AAG": 31.9,  # Lys
    "GAT": 21.8,
    "GAC": 25.1,  # Asp
    "GAA": 29.0,
    "GAG": 39.6,  # Glu
    "TGT": 10.6,
    "TGC": 12.6,  # Cys
    "TGA": 1.6,  # Stop
    "TGG": 13.2,  # Trp
    "CGT": 4.5,
    "CGC": 10.4,  # Arg
    "CGA": 6.2,
    "CGG": 11.4,  # Arg
    "AGT": 12.1,
    "AGC": 19.5,  # Ser
    "AGA": 12.2,
    "AGG": 12.0,  # Arg
    "GGT": 10.8,
    "GGC": 22.2,  # Gly
    "GGA": 16.5,
    "GGG": 16.5,  # Gly
}

RC_DICT = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}

# Get standard genetic code from Biopython
standard_table = CodonTable.standard_dna_table
forward_table = standard_table.forward_table

# Create a mapping of amino acids to their most common codon in humans
PREFERRED_CODONS = {}
for aa in set(forward_table.values()):
    possible_codons = [codon for codon, amino in forward_table.items() if amino == aa]
    PREFERRED_CODONS[aa] = max(possible_codons, key=lambda c: HUMAN_CODON_USAGE[c])


def remove_spaces(seqs: List[str]) -> np.ndarray:
    return np.array(["".join(s.strip().split(" ")) for s in seqs])


def add_spaces(seqs: List[str]) -> np.ndarray:
    return np.array([" ".join(list(s)) for s in seqs])


def single_substitution_names(sequence: str, vocab=_FASTA_VOCAB) -> List[str]:
    """Returns the names of all single mutants of a sequence."""
    mutants = []
    for (i, wt), mut in product(enumerate(sequence), vocab):
        if wt == mut:
            continue
        mutant = f"{wt}{i + 1}{mut}"
        mutants.append(mutant)
    return mutants


def single_deletion_names(sequence: str) -> List[str]:
    """Returns the names of all single deletions of a sequence."""
    mutants = []
    for i in range(len(sequence)):
        mutant = f"{sequence[i]}{i + 1}-"
        mutants.append(mutant)
    return mutants


def single_insertion_names(sequence: str, vocab=_FASTA_VOCAB) -> List[str]:
    """Returns the names of all single insertions of a sequence."""
    mutants = []
    for i in range(len(sequence) + 1):
        for mut in vocab:
            mutant = f"-{i + 1}{mut}"
            mutants.append(mutant)
    return mutants


def get_mutant(mutant: str, wildtype: str) -> str:
    assert len(mutant) == len(wildtype), "Mutant and wildtype sequences must be of the same length"
    different_indices = [i for i, (m, w) in enumerate(zip(mutant, wildtype)) if m != w]
    assert len(different_indices) <= 1, "There should be exactly one mutation"
    if len(different_indices) == 0:
        return ""
    idx = different_indices[0]
    wt = wildtype[idx]
    mt = mutant[idx]
    return f"{wt}{idx}{mt}"


def get_mutations(mutant: str, wildtype: str) -> str:
    """Get all mutations as comma-separated string (e.g., 'A105G,T107S').

    Unlike get_mutant(), this function supports any number of mutations.
    Returns empty string if sequences are identical.
    """
    assert len(mutant) == len(wildtype), "Mutant and wildtype sequences must be of the same length"
    different_indices = [i for i, (m, w) in enumerate(zip(mutant, wildtype)) if m != w]
    if len(different_indices) == 0:
        return ""
    mutations = [f"{wildtype[i]}{i}{mutant[i]}" for i in different_indices]
    return ",".join(mutations)


def split_mutant_name(mutant: str) -> Tuple[str, int, str]:
    """Splits a mutant name into the wildtype, position, and mutant."""
    return mutant[0], int(mutant[1:-1]), mutant[-1]


def sort_mutation_names(mutant: str) -> str:
    """Sorts mutation names in a sequence from greatest to smallest position."""
    delimiters = [",", r"\+", ":"]
    expression = re.compile("|".join(delimiters))
    if mutant.upper() == "WT":
        return mutant
    if expression.search(mutant):
        mutants = expression.split(mutant)
        mutants = sorted(mutants, key=lambda x: int(x[1:-1]), reverse=True)
        return ",".join(mutants)
    return mutant


def make_mutation(sequence: str, mutant: str, start_ind: int = 1) -> str:
    """Makes a mutation on a particular sequence. Multiple mutations may be separated
    by ',', ':', or '+', characters.
    """
    if len(mutant) == 0:
        return sequence
    mutant = sort_mutation_names(mutant)
    delimiters = [",", r"\+", ":"]
    expression = re.compile("|".join(delimiters))
    if mutant.upper() == "WT":
        return sequence
    if expression.search(mutant):
        mutants = expression.split(mutant)
        for mutant in mutants:
            sequence = make_mutation(sequence, mutant)
        return sequence
    else:
        wt, pos, mut = split_mutant_name(mutant)
        pos -= start_ind
        if pos < 0 or pos > len(sequence):
            raise ValueError(
                f"Position {pos} out of bounds for sequence of length {len(sequence)}."
            )
        if wt == "-":  # insertion
            return sequence[:pos] + mut + sequence[pos:]
        if mut == "-":  # deletion
            assert sequence[pos] == wt
            return sequence[:pos] + sequence[pos + 1 :]
        else:  # substitution
            assert sequence[pos] == wt
            return sequence[:pos] + mut + sequence[pos + 1 :]


def create_mutant_df(sequence: str, subs_only=False) -> pd.DataFrame:
    """Create a dataframe with mutant names and sequences"""
    names, types = ["WT"], [None]
    subs = single_substitution_names(sequence)
    names += subs
    types += ["substitution"] * len(subs)
    if not subs_only:
        ins = single_insertion_names(sequence)
        dels = single_deletion_names(sequence)
        names += ins + dels
        types += ["insertion"] * len(ins) + ["deletion"] * len(dels)
    sequences = [sequence] + [make_mutation(sequence, mut) for mut in names[1:]]
    return pd.DataFrame({"mutant": names, "sequence": sequences, "type": types})


def seqdiff(seq1: str, seq2: str) -> str:
    diff = []
    for aa1, aa2 in zip_longest(seq1, seq2, fillvalue="-"):
        if aa1 == aa2:
            diff.append(" ")
        else:
            diff.append("|")
    out = f"{seq1}\n{''.join(diff)}\n{seq2}"
    return out


def to_pivoted_mutant_df(df: pd.DataFrame) -> pd.DataFrame:
    df["wt_aa"] = df["mutant"].str.get(0)
    df["mut_aa"] = df["mutant"].str.get(-1)
    df["Position"] = df["mutant"].str.slice(1, -1).astype(int)
    df = df.drop(columns="mutant").pivot(index="mut_aa", columns=["Position", "wt_aa"])
    df = df.loc[list(_FASTA_VOCAB)]
    return df


def pivoted_mutant_df(sequence: str, scores: np.ndarray) -> pd.DataFrame:
    index = pd.Index(list(_FASTA_VOCAB), name="mut_aa")
    columns = pd.MultiIndex.from_arrays(
        [list(range(1, len(sequence) + 1)), list(sequence)], names=["Position", "wt_aa"]
    )
    df = pd.DataFrame(
        data=scores,
        index=index,
        columns=columns,
    )
    return df


def translate_codon(codon):
    """Translate a codon to an amino acid."""
    if codon in RESERVED_TOKEN_TRANSLATIONS:
        return RESERVED_TOKEN_TRANSLATIONS[codon]
    else:
        return str(Seq(codon).translate())


def translate_sequence(nt_sequence):
    if len(nt_sequence) % 3 != 0:
        raise ValueError(f"The sequence '{nt_sequence}' is not a multiple of 3.")
    aa_seq = "".join(translate_codon(nt_sequence[i : i + 3]) for i in range(0, len(nt_sequence), 3))
    if "*" in aa_seq:
        # print(aa_seq.index("*"))
        print(f"The sequence '{nt_sequence}' contains a stop codon at position {aa_seq.index('*')}.")
    return aa_seq


def backtranslate(aa_sequence: str) -> str:
    """Backtranslate an amino acid sequence to nucleotides using human codon
    preferences.

    Args:
        aa_sequence: String of amino acid single letter codes (upper case)

    Returns:
        String of nucleotides representing the backtranslated sequence

    Raises:
        KeyError: If an invalid amino acid code is encountered
    """
    return "".join(PREFERRED_CODONS[aa] for aa in aa_sequence.upper())

def rev_comp(seq):
    return ''.join([RC_DICT[nt] for nt in seq[::-1]])


# Example usage:
if __name__ == "__main__":
    # Write a simple test to make sure that create_mutant_df() works as expected
    sequence = "ACDEFGHIKLMNPQRSTVWY"
    df = create_mutant_df(sequence)
    print(df.head())
