import copy
import math
import random
import subprocess
import threading
from collections import defaultdict
from operator import methodcaller
from pathlib import Path
from tqdm import tqdm
from typing import (
    Any,
    Callable,
    Collection,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

import numba
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.distributed as dist
from Bio import SeqIO

from .align import MSA
from .ffindex import MSAFFindex
from .phylogeny import get_quantile_idx, get_quantization_points_from_geometric_grid
from .tensor import collate_list_of_dicts, collate_tensors, numpy_seed
from .tokenization import Vocab
from .typed import PathLike

T = TypeVar("T")


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


class ThreadsafeFile:
    def __init__(
        self,
        filepath: PathLike,
        open_func: Callable[[PathLike], T],
        close_func: Callable[[T], None] = methodcaller("close"),
    ):
        self._threadlocal = threading.local()
        self._filepath = filepath
        self._open_func = open_func
        self._close_func = close_func

    def __getattr__(self, name: str):
        return getattr(self.file, name)

    @property
    def file(self) -> T:
        if not hasattr(self._threadlocal, "file"):
            self._threadlocal.file = self._open_func(self._filepath)
        return self._threadlocal.file

    def __getstate__(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if k != "_threadlocal"}

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__ = state
        self._threadlocal = threading.local()

    def __del__(self):
        if hasattr(self._threadlocal, "file"):
            self._close_func(self._threadlocal.file)
            del self._threadlocal.file


class SizedDataset(torch.utils.data.Dataset):
    def __init__(self, sizes: np.ndarray, *args, **kwargs):
        super().__init__(*args, **kwargs)  # type: ignore
        self._sizes = sizes

    def __len__(self):
        return len(self.sizes)

    @property
    def sizes(self):
        return self._sizes


class CollatableDataset(torch.utils.data.Dataset):
    def collater(self, batch: List[Any]) -> Any:
        try:
            return torch.stack(batch, 0)
        except Exception:
            return batch


class CollatableVocabDataset(CollatableDataset):
    def __init__(self, vocab: Vocab, *args, **kwargs):
        super().__init__(*args, **kwargs)  # type: ignore
        self.vocab = vocab

    def collater(self, batch: List[Any]) -> Any:
        return collate_tensors(batch, constant_value=self.vocab.pad_idx)


class TorchWrapperDataset(CollatableVocabDataset):
    """TorchWrapperDataset. Wraps an existing torch dataset.

    Args:
        dataset (torch.utils.data.dataset): Dataset to wrap.
    """

    def __init__(self, dataset: torch.utils.data.Dataset, vocab: Vocab):
        super().__init__(vocab)
        self.dataset = dataset

    def __getattr__(self, name: str):
        if "dataset" not in self.__dict__:
            raise AttributeError("No dataset")
        return getattr(self.dataset, name)

    def __getitem__(self, index: int):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)


class BaseWrapperDataset(CollatableVocabDataset):
    """BaseWrapperDataset. Wraps an existing dataset.

    Args:
        dataset (torch.utils.data.dataset): Dataset to wrap.
    """

    def __init__(self, dataset: CollatableVocabDataset):
        super().__init__(dataset.vocab)
        self.dataset = dataset

    def __getattr__(self, name: str):
        if "dataset" not in self.__dict__:
            raise AttributeError("No dataset")
        return getattr(self.dataset, name)

    def __getitem__(self, index: int):
        return self.dataset[index]

    def collater(self, batch):
        return self.dataset.collater(batch)

    def __len__(self):
        return len(self.dataset)


class WeightedConcatDataset(torch.utils.data.ConcatDataset):
    def __init__(self, datasets, weights=None, **kwargs):
        super().__init__(datasets)
        if weights is None:
            weights = [1.0 / len(dataset) for dataset in datasets]
        assert len(datasets) == len(weights)
        # repeat weight len(dataset) times for each dataset
        self._weights = [
            weight for dataset, weight in zip(datasets, weights) for _ in range(len(dataset))
        ]

    @property
    def weights(self):
        return self._weights


class SubsetDataset(BaseWrapperDataset):
    def __init__(
        self,
        dataset: CollatableVocabDataset,
        subset: Sequence[float],
        index: int,
        seed: int = 0,
    ):
        super().__init__(dataset)
        fracs = np.array(subset)
        assert np.isclose(fracs.sum(), 1)
        percentages = np.append(0, np.cumsum(fracs))
        percentages[-1] = 1
        with numpy_seed(seed):
            indices = np.random.permutation(np.arange(len(dataset)))  # type: ignore
            start, end = (percentages[index : index + 2] * len(dataset)).astype(  # type: ignore
                np.int64
            )
            indices = np.sort(indices[start:end])
        self._indices = indices
        self.sizes = dataset.sizes[indices]  # type: ignore

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, index: int):
        index = self._indices[index]
        return super().__getitem__(index)


class NPZDataset(torch.utils.data.Dataset):
    """Creates a dataset from a directory of npz files.
    Args:
        data_file (Union[str, Path]): Path to directory of npz files
        split_files (Optional[Collection[str]]): Subset of files to use,
            can be used to specify training / validation / testing sets.
    """

    def __init__(
        self,
        data_file: PathLike,
        split_files: Optional[Collection[str]] = None,
        lazy: bool = False,
    ):
        data_file = Path(data_file)
        if not data_file.exists():
            raise FileNotFoundError(data_file)
        if not data_file.is_dir():
            raise NotADirectoryError(data_file)

        file_glob = data_file.glob("*.npz")
        if split_files is None:
            file_list = list(file_glob)
        else:
            split_files = set(split_files)
            if len(split_files) == 0:
                raise ValueError("Passed an empty split file set")

            file_list = [f for f in file_glob if f.stem in split_files]
            if len(file_list) != len(split_files):
                num_missing = len(split_files) - len(file_list)
                raise FileNotFoundError(
                    f"{num_missing} specified split files not found in directory"
                )

        if len(file_list) == 0:
            raise FileNotFoundError(f"No .npz files found in {data_file}")

        self._file_list = sorted(file_list)
        self.keys = {f.stem: i for i, f in enumerate(self._file_list)}
        self._lazy = lazy

    def get(self, key: str):
        index = self.keys[key]
        return self[index]

    def key(self, index: int) -> str:
        return self._file_list[index].stem

    def __len__(self) -> int:
        return len(self._file_list)

    def __getitem__(self, index: int):
        if not 0 <= index < len(self):
            raise IndexError(index)

        item = np.load(self._file_list[index])
        if not self._lazy:
            item = dict(item)
        return item


class MSADataset(torch.utils.data.Dataset):
    """Creates a dataset from a directory of a2m/a3m files.
    Args:
        file_ext (str): File ext to use, either 'a2m' or 'a3m'.
        data_file (Union[str, Path]): Path to directory of a2m/a3m files
        split_files (Optional[Collection[str]]): Subset of files to use,
            can be used to specify training / validation / testing sets.
    """

    def __init__(
        self,
        data_file: PathLike,
        file_ext: str = "a3m",
        split_files: Optional[Collection[str]] = None,
        max_seqs_per_msa: Optional[int] = None,
        sample_method: str = "hhfilter",
    ):
        assert sample_method in ("hhfilter", "sample-weights")
        data_file = Path(data_file)
        if not data_file.exists():
            raise FileNotFoundError(data_file)
        if not data_file.is_dir():
            raise NotADirectoryError(data_file)

        file_glob = data_file.glob(f"*.{file_ext}")
        if split_files is None:
            file_list = list(file_glob)
        else:
            split_files = set(split_files)
            if len(split_files) == 0:
                raise ValueError("Passed an empty split file set")

            file_list = [f for f in file_glob if f.stem in split_files]
            if len(file_list) != len(split_files):
                num_missing = len(split_files) - len(file_list)
                raise FileNotFoundError(
                    f"{num_missing} specified split files not found in directory"
                )

        if len(file_list) == 0:
            raise FileNotFoundError(f"No .{file_ext} files found in {data_file}")

        self.file_ext = file_ext
        self._file_list = sorted(file_list)
        self.keys = {f.stem: i for i, f in enumerate(self._file_list)}
        self._max_seqs_per_msa = max_seqs_per_msa
        self._sample_method = sample_method

    def get(self, key: str):
        index = self.keys[key]
        return self[index]

    def key(self, index: int) -> str:
        return self._file_list[index].stem

    def read_msa(self, index: int) -> MSA:
        if self.file_ext == "a3m":
            return MSA.from_fasta(
                self._file_list[index],
                keep_insertions=False,
                uppercase=False,
                remove_lowercase_cols=False,
            )
        elif self.file_ext == "a2m":
            return MSA.from_fasta(
                self._file_list[index],
                keep_insertions=True,
                uppercase=True,
                remove_lowercase_cols=False,
            )
        raise ValueError(f"Unknown file extension: {self.file_ext}")

    def __len__(self) -> int:
        return len(self._file_list)

    def __getitem__(self, index: int):
        if not 0 <= index < len(self):
            raise IndexError(index)
        if self._max_seqs_per_msa == 1:
            seq = str(next(SeqIO.parse(self._file_list[index], "fasta")).seq)
            return seq
        else:
            msa = self.read_msa(index)
            if self._max_seqs_per_msa is not None:
                msa = msa.select_diverse(
                    self._max_seqs_per_msa,
                    method=self._sample_method,
                    file_ext=self.file_ext,
                )
            return msa


class EncodedMSADataset(CollatableVocabDataset, MSADataset):
    def __init__(self, vocab: Vocab, *args, **kwargs):
        super().__init__(vocab=vocab, *args, **kwargs)

    def __getitem__(self, idx):
        return torch.from_numpy(self.vocab.encode(super().__getitem__(idx)))


class PhyloDataset(torch.utils.data.Dataset):
    """Creates a dataset from a directory of paired fasta + npy files.
    Args:
        data_dir (Union[str, Path]): Path to directory of fasta + npy files. Each fasta file should have a corresponding npy file with the same name (but .npy extension).
        split_files (Optional[Collection[str]]): Subset of files to use,
            can be used to specify training / validation / testing sets.
        max_seqs_per_phylo (Optional[int]): Maximum number of sequences per phylo tree in a sample. If None, uses all sequences.
        sample_method (str): Method to sample subtrees from the phylo tree. Options are 'random' (randomly sample a subtree) or 'path' (sample a path from root to leaf). Default is 'random'.
    """

    def __init__(
        self,
        data_dir: PathLike,
        split_files: Optional[Collection[str]] = None,
        max_seqs_per_phylo: Optional[int] = None,
        sample_method: str = "random",
    ):
        assert sample_method in ("random", "path")
        data_dir = Path(data_dir)
        if not data_dir.exists():
            raise FileNotFoundError(data_dir)
        if not data_dir.is_dir():
            raise NotADirectoryError(data_dir)

        file_glob = data_dir.glob("*.fasta")
        if split_files is None:
            file_list = list(file_glob)
        else:
            split_files = set(split_files)
            if len(split_files) == 0:
                raise ValueError("Passed an empty split file set")

            file_list = [f for f in file_glob if f.stem in split_files]
            if len(file_list) != len(split_files):
                num_missing = len(split_files) - len(file_list)
                raise FileNotFoundError(
                    f"{num_missing} specified split files not found in directory"
                )

        if len(file_list) == 0:
            raise FileNotFoundError(f"No .fasta files found in {data_dir}")

        pdm_file_list = [f.with_suffix(".npy") for f in file_list]
        for f in pdm_file_list:
            if not f.exists():
                raise FileNotFoundError(f"Missing corresponding .npy file for {f.stem}")

        if max_seqs_per_phylo is not None:
            assert max_seqs_per_phylo > 1, "max_seqs_per_phylo must be greater than 1 or None"

        self._file_list = sorted(file_list)
        self.keys = {f.stem: i for i, f in enumerate(self._file_list)}
        self._max_seqs_per_phylo = max_seqs_per_phylo
        self._sample_method = sample_method

        # Pre-compute sequence counts for efficient batch sampling
        self.seq_counts = self._compute_seq_counts()

    def get(self, key: str):
        index = self.keys[key]
        return self[index]

    def key(self, index: int) -> str:
        return self._file_list[index].stem

    def _compute_seq_counts(self) -> np.ndarray:
        """Efficiently compute sequence counts for all phylos by counting fasta headers."""
        counts = []
        for fasta_file in self._file_list:
            count = 0
            with open(fasta_file) as f:
                for line in f:
                    if line.startswith(">"):
                        count += 1
            counts.append(count)
        return np.array(counts, dtype=np.int32)

    def read_phylo(self, index: int) -> Tuple[Dict[str, SeqIO.SeqRecord], np.ndarray]:
        fasta_file = self._file_list[index]
        pdm_file = fasta_file.with_suffix(".npy")
        sequences = SeqIO.to_dict(SeqIO.parse(fasta_file, "fasta"))
        pdm: np.ndarray = np.load(pdm_file)
        assert (
            len(sequences) == pdm.shape[0]
        ), f"Number of sequences in {fasta_file} does not match size of PDM in {pdm_file}"
        assert pdm.shape[0] == pdm.shape[1], f"PDM in {pdm_file} must be square"
        assert pdm.diagonal().sum() == 0, f"PDM in {pdm_file} must have zeros on the diagonal"
        return sequences, pdm

    def tree_size(self, index: int) -> int:
        # return number of sequences in the phylo tree at index
        return self.seq_counts[index]

    def select_subtree(
        self, sequences: Dict[str, SeqIO.SeqRecord], pdm: np.ndarray
    ) -> Tuple[Dict[str, SeqIO.SeqRecord], np.ndarray]:
        if self._sample_method == "random":
            indices = np.random.choice(len(sequences), size=self._max_seqs_per_phylo, replace=False)
        elif self._sample_method == "path":
            raise NotImplementedError("Path sampling method not implemented yet")
        sequences = {k: v for i, (k, v) in enumerate(sequences.items()) if i in indices}
        pdm = pdm[np.ix_(indices, indices)]
        return sequences, pdm

    def __len__(self) -> int:
        return len(self._file_list)

    def __getitem__(self, index: int, seq_slice=None):
        if not 0 <= index < len(self):
            raise IndexError(index)

        seqs, pdm = self.read_phylo(index)

        # Handle sequence indexing (for PhyloSampler)
        if seq_slice is not None:
            seq_list = list(seqs.items())

            # Handle both slice objects and index arrays/lists
            if isinstance(seq_slice, slice):
                sliced_items = seq_list[seq_slice]
                indices = list(range(len(seq_list)))[seq_slice]
            else:
                # seq_slice is an array or list of indices
                indices = np.asarray(seq_slice, dtype=np.int64)
                sliced_items = [seq_list[i] for i in indices]

            seqs = dict(sliced_items)
            pdm = pdm[np.ix_(indices, indices)]

        # Handle max_seqs_per_phylo (random subsampling)
        elif self._max_seqs_per_phylo is not None and len(seqs) > self._max_seqs_per_phylo:
            seqs, pdm = self.select_subtree(sequences=seqs, pdm=pdm)

        return seqs, pdm


class ParquetDataset(torch.utils.data.Dataset):
    """Creates a dataset from a parquet file."""

    def __init__(
        self,
        data_file: PathLike,
        sequence_col: str = "sequence",
        prop_cols: List[str] = [],
    ):
        self.data_file = data_file
        self.table = pq.read_table(self.data_file)
        self.sequences = self.table.column(sequence_col).to_pylist()
        self.properties = {col: self.table.column(col).to_pylist() for col in prop_cols}
        self.sequence_col = sequence_col
        self.prop_cols = prop_cols

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int):
        if not 0 <= index < len(self):
            raise IndexError(index)
        sequence = self.sequences[index]
        properties = {k: v[index] for k, v in self.properties.items()}
        return properties, sequence


class EncodedParquetDataset(CollatableVocabDataset, ParquetDataset):
    def __init__(self, vocab, *args, **kwargs):
        super().__init__(vocab, *args, **kwargs)

    def __getitem__(self, index):
        properties, sequence = super().__getitem__(index)
        sequence = torch.from_numpy(self.vocab.encode_single_sequence(sequence))
        return {"sequence": sequence} | properties

    @property
    def batch_keys(self):
        return [self.sequence_col] + self.prop_cols

    def collater(self, batch):
        return collate_list_of_dicts(batch, self.batch_keys, self.vocab.pad_idx)


class CherriesDataset(torch.utils.data.Dataset):
    """Creates a dataset of sequence cherries separated by a distance metric

    Loads pairs of protein sequences + a float from .txt file format:
        num transitions
        seq1 seq2 time
        seq1 seq2 time
    """

    def __init__(
        self,
        data_file: PathLike,
        cache_indices: bool = False,
        min_t: float = 5e-3,
        max_len: Optional[int] = None,
        quantize_t: bool = False,
        permute_xy: bool = False,
    ):
        self.data_file = Path(data_file)
        if not self.data_file.exists():
            raise FileNotFoundError(f"{self.data_file}")
        self.file = None
        self.cache = Path(f"{data_file}.idx.npy")

        self.min_t = min_t
        self.max_len = max_len
        self.permute_xy = permute_xy
        self.quantize_t = quantize_t
        self.time_bins = np.array(get_quantization_points_from_geometric_grid(), dtype=np.float32)

        if cache_indices:
            if self.cache.exists():
                self.offsets = np.load(self.cache)
            else:
                self.offsets = self._build_index()
                np.save(self.cache, self.offsets)
        else:
            self.offsets = self._build_index()

    def __getitem__(self, idx):
        if self.file is None:
            self.file = ThreadsafeFile(self.data_file, open)
        self.file.seek(self.offsets[idx])
        if idx == len(self) - 1:
            data = self.file.read()
        else:
            data = self.file.read(self.offsets[idx + 1] - self.offsets[idx])
        line = data.strip()
        parts = line.split()
        seq1, seq2, t = parts[0], parts[1], str(parts[2])
        if is_number(t):
            t = float(t)
            t = max(t, self.min_t)
        if self.quantize_t:
            t = get_quantile_idx(self.time_bins, t)
        if self.max_len is not None:
            seq1 = seq1[: self.max_len]
            seq2 = seq2[: self.max_len]
        if self.permute_xy and random.random() < 0.5:
            seq1, seq2 = seq2, seq1
        return seq1, seq2, t

    def __len__(self):
        return self.offsets.size

    def _build_index(self):
        offsets = []
        with open(self.data_file, "r") as f:
            # Skip first line (header)
            f.readline()
            while True:
                offset = f.tell()
                line = f.readline()
                if not line:
                    break
                offsets.append(offset)
        return np.array(offsets, dtype=np.int64)


class ComplexCherriesDataset(CherriesDataset):
    """Extension of cherries dataset that handles complex sequences with a chain break separator character."""

    def __init__(self, sep_token: str = ".", chain_id_offset: int = 1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sep_token = sep_token
        self.chain_id_offset = chain_id_offset

    def __getitem__(self, idx):
        if self.file is None:
            self.file = ThreadsafeFile(self.data_file, open)

        self.file.seek(self.offsets[idx])
        if idx == len(self) - 1:
            data = self.file.read()
        else:
            data = self.file.read(self.offsets[idx + 1] - self.offsets[idx])
        line = data.strip()
        parts = line.split()
        seq1, seq2, ts = str(parts[0]), str(parts[1]), parts[2:]

        xs = seq1.split(self.sep_token)
        ys = seq2.split(self.sep_token)
        chain_ids = list(range(self.chain_id_offset, self.chain_id_offset + len(xs)))

        assert len(xs) == len(
            ys
        ), f"Different number of chains in {seq1} and {seq2} is not allowed."

        if len(ts) > len(xs):
            ts = ts[: len(xs)]
        elif len(ts) < len(xs):
            ts = ts + [ts[-1]] * (len(xs) - len(ts))

        for i in range(len(ts)):
            if is_number(ts[i]):
                ts[i] = float(ts[i])
                ts[i] = max(ts[i], self.min_t)
                ts[i] = get_quantile_idx(self.time_bins, ts[i]) if self.quantize_t else ts[i]

        return xs, ys, ts, chain_ids


class ComplexCherriesCollection(torch.utils.data.ConcatDataset):
    """Concatenation of multiple ComplexCherriesDataset from different files."""

    def __init__(
        self,
        data_dir: PathLike,
        file_ext: str = "txt",
        split_files: Optional[Collection[str]] = None,
        split_file: Optional[str] = None,
        sep_token: str = ".",
        chain_id_offset: int = 1,
        *args,
        **kwargs,
    ):
        data_dir = Path(data_dir)
        if not data_dir.exists():
            raise FileNotFoundError(data_dir)
        if not data_dir.is_dir():
            raise NotADirectoryError(data_dir)
        file_glob = data_dir.glob(f"*.{file_ext}")

        if split_files is None and split_file is not None:
            with open(split_file, "r") as f:
                split_files = set(line.strip() for line in f if line.strip())

        data_files = list(file_glob)
        if split_files is not None:
            split_files = set(split_files)
            data_files = [f for f in file_glob if f.stem in split_files]
            print(f"Using {len(data_files)} files from split.")
        else:
            print(f"Using all {len(data_files)} files in directory.")

        self.family_ids = [f.stem for f in data_files]
        datasets = [
            ComplexCherriesDataset(
                data_file=f,
                sep_token=sep_token,
                chain_id_offset=chain_id_offset,
                *args,
                **kwargs,
            )
            for f in tqdm(data_files, desc="Loading ComplexCherriesDatasets")
        ]
        super().__init__(datasets)


class RankedTripletItemDataset(torch.utils.data.Dataset):
    """Dataset for Direct Preference Optimization that returns individual triplet items (x, y1, y2) where
    x is the anchor, y1 is a positive example preferred over x, and y2 is a decoy less preferred than y1.
    """

    def __init__(
        self,
        sequences_file: PathLike,
        triplets_file: PathLike,
        quantized_ts: PathLike,
        ref_lls_file: PathLike,
        fitness_col: str = None,
    ):
        super().__init__()
        self.id_to_seq, self.id_to_fitness = self._process_sequences_csv(sequences_file, fitness_col)
        self.quantized_ts = np.load(quantized_ts).astype(np.float32)
        (
            self.x_id_arr,
            self.pos_id_arr,
            self.neg_id_arr,
            self.bin_idx_arr,
            self.pos_ref_lls,
            self.neg_ref_lls,
            self.group_indices,
        ) = self._process_triplets_csv(triplets_file, ref_lls_file)

    def _process_sequences_csv(self, sequences_file: PathLike, fitness_col: str = None):
        seq_df = pd.read_csv(sequences_file)
        id_to_sequence = dict(zip(seq_df["identifier"].astype(str), seq_df["sequence"].astype(str)))
        id_to_fitness = None
        if fitness_col is not None:
            id_to_fitness = dict(zip(seq_df["identifier"].astype(str), seq_df[fitness_col].values.astype(np.float32)))
        return id_to_sequence, id_to_fitness

    def _process_triplets_csv(self, triplets_file: PathLike, ref_lls_file: PathLike):
        triplets_df = pd.read_csv(triplets_file)
        ref_lls_dict = self._process_ref_lls(ref_lls_file)
        group_indices = triplets_df.groupby(["anchor_x", "bin_b"]).indices
        x_id_arr = triplets_df["anchor_x"].values.astype(str)
        pos_id_arr = triplets_df["positive_y1"].values.astype(str)
        neg_id_arr = triplets_df["decoy_y2"].values.astype(str)
        bin_idx_arr = triplets_df["bin_b"].values.astype(int)
        pos_id_tuples = list(zip(x_id_arr, pos_id_arr, bin_idx_arr))
        neg_id_tuples = list(zip(x_id_arr, neg_id_arr, bin_idx_arr))
        pos_ref_lls = np.array([ref_lls_dict[tid] for tid in pos_id_tuples])
        neg_ref_lls = np.array([ref_lls_dict[tid] for tid in neg_id_tuples])
        assert set(x_id_arr).issubset(set(self.id_to_seq.keys()))
        assert bin_idx_arr.max() < len(self.quantized_ts)
        return x_id_arr, pos_id_arr, neg_id_arr, bin_idx_arr, pos_ref_lls, neg_ref_lls, group_indices

    def _process_ref_lls(self, ref_lls_file: PathLike):
        ref_lls_df = pd.read_csv(ref_lls_file)
        triplet_ids = list(
            zip(
                ref_lls_df["x"].astype(str),
                ref_lls_df["y"].astype(str),
                ref_lls_df["b"].astype(int),
            )
        )
        ref_lls_dict = dict(zip(triplet_ids, ref_lls_df["ll"].values.astype(np.float32)))
        return ref_lls_dict

    def __getitem__(self, index):
        anchor_x = self.x_id_arr[index]
        bin_idx = self.bin_idx_arr[index]
        y1, y2 = self.pos_id_arr[index], self.neg_id_arr[index]
        item_id = f"{anchor_x};{y1};{y2};{bin_idx}"
        return (
            self.id_to_seq[anchor_x],                   # anchor sequence (x)
            self.id_to_seq[y1],                         # positive sequence (y1)
            self.id_to_seq[y2],                         # negative sequence (y2)
            self.quantized_ts[bin_idx],                 # quantized time (tau)
            self.pos_ref_lls[index],                    # reference model log p(y1 | x, tau)
            self.neg_ref_lls[index],                    # reference model log p(y2 | x, tau)
            item_id,                                    # unique identifier for the triplet item
        )

    def __len__(self):
        return len(self.x_id_arr)


class RankedTripletsDataset(RankedTripletItemDataset):
    """Same as RankedTripletItemDataset but returns blocks of triplets from the same anchor and time bin for 
    block based training where each batch contains multiple triplets for the same (x, b) group.
    """

    def __init__(self, anchor_bin_file: PathLike, block_size: int = 128, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.block_size = block_size
        self.anchor_bin_ids = self._process_anchor_bin_csv(anchor_bin_file)

    def _process_anchor_bin_csv(self, anchor_bin_file: PathLike):
        anchor_bin_df = pd.read_csv(anchor_bin_file)
        anchor_bin_df = anchor_bin_df[anchor_bin_df["num_triplets"] > 0].reset_index(drop=True)
        anchor_bin_tuple_list = list(
            zip(anchor_bin_df["anchor_x"].astype(str), anchor_bin_df["bin_b"].astype(int))
        )
        unique_ids = set(anchor_bin_df["anchor_x"].astype(str))
        max_bin = anchor_bin_df["bin_b"].max()
        assert unique_ids.issubset(set(self.id_to_seq.keys()))
        assert max_bin < len(self.quantized_ts)
        return anchor_bin_tuple_list

    def __getitem__(self, index):
        group_key = self.anchor_bin_ids[index]
        group_indices = self.group_indices[group_key]
        effective_block_size = min(self.block_size, len(group_indices))
        sampled_indices = np.random.choice(group_indices, size=effective_block_size, replace=False)

        # get anchor sequence and quantized time for the group
        anchor_x, bin_idx = group_key
        anchor_seq = self.id_to_seq[anchor_x]
        quantized_time = self.quantized_ts[bin_idx]

        # get positive and negative sequences for the sampled triplets
        positive_seqs = [self.id_to_seq[pid] for pid in self.pos_id_arr[sampled_indices]]
        negative_seqs = [self.id_to_seq[nid] for nid in self.neg_id_arr[sampled_indices]]
        positive_ref_lls = self.pos_ref_lls[sampled_indices]
        negative_ref_lls = self.neg_ref_lls[sampled_indices]

        return (
            anchor_seq,
            positive_seqs,
            negative_seqs,
            quantized_time,
            positive_ref_lls,
            negative_ref_lls,
        )

    def __len__(self):
        return len(self.anchor_bin_ids)


class FastaDataset(SizedDataset):
    """
    For loading protein sequence datasets in the common FASTA data format

    Modified from github.com/pytorch/fairseq.
    """

    def __init__(self, data_file: PathLike, cache_indices: bool = False):
        self.data_file = Path(data_file)
        if not self.data_file.exists():
            raise FileNotFoundError(
                f"{self.data_file}\n"
                "If using hydra, make sure you are using abolute instead of relative paths."
            )
        self.file = ThreadsafeFile(data_file, open)
        self.cache = Path(f"{data_file}.idx.npy")
        if cache_indices:
            if self.cache.exists():
                self.offsets, sizes = np.load(self.cache)
            else:
                self.offsets, sizes = self._build_index()
                np.save(self.cache, np.stack([self.offsets, sizes]))
        else:
            self.offsets, sizes = self._build_index()

        super().__init__(sizes)

    def __getitem__(self, idx):
        return self.get(idx)

    def get(self, idx: int):
        self.file.seek(self.offsets[idx])
        if idx == len(self) - 1:
            data = self.file.read()
        else:
            data = self.file.read(self.offsets[idx + 1] - self.offsets[idx])
        desc, *seq = data.split("\n")
        return desc[1:], "".join(seq)

    def __len__(self):
        return self.offsets.size

    def _build_index(self):
        # Use grep and awk to get 100M/s on local SSD.
        # Should process your enormous 100G fasta in ~10 min single core...
        bytes_offsets = subprocess.check_output(
            f"cat {self.data_file} | tqdm --bytes --total $(wc -c < {self.data_file})"
            "| grep --byte-offset '^>' -o | cut -d: -f1",
            shell=True,
        )
        fasta_lengths = subprocess.check_output(
            f"cat {self.data_file} | tqdm --bytes --total $(wc -c < {self.data_file})"
            '| awk \'/^>/ {print "";next;} { printf("%s",$0);}\' | tail -n+2 | awk '
            "'{print length($1)}'",
            shell=True,
        )
        bytes_np = np.fromstring(bytes_offsets, dtype=np.int64, sep=" ")
        sizes_np = np.fromstring(fasta_lengths, dtype=np.int64, sep=" ")
        return bytes_np, sizes_np


class EncodedFastaDataset(CollatableVocabDataset, FastaDataset):
    def __init__(self, data_file: PathLike, vocab: Vocab, cache_indices: bool = False):
        super().__init__(data_file=data_file, vocab=vocab, cache_indices=cache_indices)
        self._sizes += int(self.vocab.prepend_bos) + int(self.vocab.append_eos)

    def __getitem__(self, index: int) -> torch.Tensor:
        desc, seq = super().__getitem__(index)
        return torch.from_numpy(self.vocab.encode_single_sequence(seq))


class EncodedIndexedMSADataset(CollatableVocabDataset):
    def __init__(self, ffindex_path: PathLike, vocab: Vocab):
        super().__init__(vocab)

        ffindex_path = Path(ffindex_path)
        index_file = ffindex_path.with_suffix(".ffindex")
        data_file = ffindex_path.with_suffix(".ffdata")
        self.ffindex = MSAFFindex(index_file, data_file)

    def __len__(self):
        return len(self.ffindex)

    def __getitem__(self, idx):
        msa = self.ffindex[idx]
        return torch.from_numpy(self.vocab.encode(msa))

    def collater(self, batch: List[Any]) -> Any:
        return collate_tensors(batch)


class TorchDataset(CollatableVocabDataset):
    def __init__(self, data_file: PathLike, vocab: Vocab):
        data_file = Path(data_file)
        self.data_file = data_file
        self.data = torch.load(data_file)
        self.offsets, self.sizes = np.load(data_file.with_suffix(".fasta.idx.npy"))

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, idx):
        item = self.data[self.offsets[idx] : self.offsets[idx] + self.sizes[idx]]
        return item

    def collater(self, batch):
        return collate_tensors(batch, constant_value=self.vocab.pad_idx)


class MaxTokenBatch(object):
    def __init__(self, max_tokens: int, pad_idx: int):
        self.max_tokens = max_tokens
        self.pad_idx = pad_idx
        self.items: List[torch.Tensor] = []
        self.sizes = None

    def can_add_item(self, item: torch.Tensor) -> bool:
        sizes = np.asarray(item.size())
        if self.sizes is not None:
            sizes = np.max([self.sizes, sizes], 0)
        total_tokens = (len(self.items) + 1) * sizes.prod()
        return total_tokens <= self.max_tokens

    def add_item(self, item: torch.Tensor):
        self.items.append(item)
        sizes = np.asarray(item.size())
        if self.sizes is None:
            self.sizes = sizes
        else:
            self.sizes = np.max([self.sizes, sizes], 0)
        if self.num_tokens > self.max_tokens:
            raise RuntimeError("Too many sequences in batch!")

    def finalize(self) -> torch.Tensor:
        return collate_tensors(self.items, constant_value=self.pad_idx)

    @property
    def num_tokens(self) -> int:
        if self.sizes is None:
            return 0
        else:
            return len(self.items) * self.sizes.prod()


BatchOrSequence = TypeVar("BatchOrSequence", MaxTokenBatch, Sequence[MaxTokenBatch])


class AutoBatchingDataset(torch.utils.data.IterableDataset):
    def __init__(self, dataset: CollatableVocabDataset, max_tokens: int, shuffle: bool = False):
        super().__init__()
        self.dataset = dataset
        self.vocab = dataset.vocab
        self.max_tokens = max_tokens
        self.shuffle = shuffle

    def maybe_make_and_add_batch(
        self,
        batch: Optional[BatchOrSequence],
        item: Union[torch.Tensor, Sequence[torch.Tensor]],
    ) -> Tuple[BatchOrSequence, bool]:
        if batch is None:
            if isinstance(item, torch.Tensor):
                batch = MaxTokenBatch(self.max_tokens, self.vocab.pad_idx)  # type: ignore
            else:
                batch = [  # type: ignore
                    MaxTokenBatch(self.max_tokens, self.vocab.pad_idx) for _ in item
                ]

        if isinstance(batch, MaxTokenBatch):
            can_add = batch.can_add_item(item)  # type: ignore
            if can_add:
                batch.add_item(item)  # type: ignore
        else:
            can_add = batch[0].can_add_item(item[0])  # type: ignore
            if can_add:
                for b, i in zip(batch, item):  # type: ignore
                    b.add_item(i)
        return batch, can_add  # type: ignore

    def __iter__(self):
        indices = np.arange(len(self.dataset))

        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            worker_rank = dist.get_rank()
        else:
            world_size = 1
            worker_rank = 0

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            world_size *= worker_info.num_workers
            worker_rank = worker_rank * worker_rank.num_workers + worker_info.id

        chunk_size = math.ceil(len(indices) / world_size)
        indices = indices[chunk_size * worker_rank : chunk_size * (worker_rank + 1)]

        if self.shuffle:
            indices = np.random.permutation(indices)

        batch = None
        for idx in indices:
            items = self.dataset[idx]
            batch, added = self.maybe_make_and_add_batch(batch, items)
            if not added:
                if isinstance(batch, MaxTokenBatch):
                    yield batch.finalize()
                else:
                    yield type(items)(b.finalize() for b in batch)
                batch, added = self.maybe_make_and_add_batch(None, items)
                assert added, "Item size too large to include!"
        if batch:
            if isinstance(batch, MaxTokenBatch):
                yield batch.finalize()
            else:
                yield type(items)(b.finalize() for b in batch)


@numba.njit
def batch_by_size(indices: np.ndarray, sizes: np.ndarray, max_tokens: int) -> List[List[int]]:
    batches: List[List[int]] = []
    batch: List[int] = [0][:0]
    batch_size = 0
    for i in range(len(indices)):
        idx = indices[i]
        size = sizes[i]
        if size > max_tokens:
            raise RuntimeError("An item was too large to batch.")
        if size + batch_size > max_tokens:
            batches.append(batch)
            batch = [0][:0]
            batch_size = 0
        batch.append(idx)
        batch_size += size
    batches.append(batch)
    return batches


class BatchBySequenceLength(torch.utils.data.Sampler):
    def __init__(
        self,
        dataset: SizedDataset,
        max_tokens: int,
        shuffle=True,
        seed=0,
    ):
        super().__init__(dataset)
        indices = np.argsort(dataset.sizes)
        sizes = dataset.sizes[indices]
        batches = batch_by_size(indices, sizes, max_tokens)

        self.dataset = dataset
        self.max_tokens = max_tokens
        self.shuffle = shuffle
        self.batches = batches
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.batches), generator=g).tolist()
        else:
            indices = list(range(len(self.batches)))

        # add extra samples to make it evenly divisible
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == len(self)
        yield from (self.batches[idx] for idx in indices)

    def __len__(self):
        return math.ceil(len(self.batches) / self.num_replicas)

    @property
    def num_replicas(self) -> int:
        if dist.is_available() and dist.is_initialized():
            return dist.get_world_size()
        else:
            return 1

    @property
    def total_size(self) -> int:
        return len(self) * self.num_replicas

    @property
    def rank(self) -> int:
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
        else:
            return 0

    def set_epoch(self, epoch):
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all
        replicas use a different random ordering for each epoch. Otherwise, the next
        iteration of this sampler will yield the same ordering.

        Arguments:
            epoch (int): Epoch number.
        """
        self.epoch = epoch


class RandomIdentitySampler(torch.utils.data.Sampler):
    """Randomly samples N identities each with K instances.

    Args:
        pids (List[int]): List of group IDs for each instance in the dataset.
        batch_size (int): batch size.
        max_num_instances (int): Maximum number of instances per identity in a batch.
    """

    def __init__(self, pids: List[int], batch_size: int, max_num_instances: int):
        if batch_size < max_num_instances:
            print(
                "Warning: batch_size is smaller than max_num_instances, so max_num_instances will be set to batch_size."
            )
            max_num_instances = batch_size

        self.batch_size = batch_size
        self.max_num_instances = max_num_instances
        self.index_dic = defaultdict(list)

        for index, pid in enumerate(pids):
            self.index_dic[pid].append(index)

        self.pids = list(self.index_dic.keys())
        self.length = len(self.pids)

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            random.shuffle(idxs)

            # iterate over idxs in chunks of size max_num_instances
            for i in range(0, len(idxs), self.max_num_instances):
                batch_idxs_dict[pid].append(idxs[i : i + self.max_num_instances])

        avai_pids = copy.deepcopy(self.pids)
        random.shuffle(avai_pids)
        final_idxs = []

        while len(avai_pids) > 0:
            pid = avai_pids[0]
            batch_idxs = batch_idxs_dict[pid].pop(0)
            final_idxs.extend(batch_idxs)
            if len(batch_idxs_dict[pid]) == 0:
                avai_pids.remove(pid)

        return iter(final_idxs)

    def __len__(self):
        return self.length


class RandomCropDataset(BaseWrapperDataset):
    def __init__(self, dataset: CollatableVocabDataset, max_seqlen: int):
        super().__init__(dataset)
        self.max_seqlen = max_seqlen
        self.num_special = int(self.vocab.prepend_bos) + int(self.vocab.append_eos)
        self.max_seqlen_no_special = self.max_seqlen - self.num_special
        if isinstance(self.dataset, SizedDataset):
            self.sizes = np.minimum(self.sizes, max_seqlen)  # type: ignore

    def __getitem__(self, idx):
        item = self.dataset[idx]
        seqlen = item.size(-1)
        if seqlen > self.max_seqlen:
            low_idx = int(self.vocab.prepend_bos)
            high_idx = seqlen - int(self.vocab.append_eos)
            start_idx = np.random.randint(low_idx, high_idx)
            end_idx = start_idx + self.max_seqlen_no_special
            item = torch.cat(
                [
                    item[..., :low_idx],
                    item[..., start_idx:end_idx],
                    item[..., high_idx:],
                ],
                -1,
            )
        return item


class EncodedSubsampleMSADataset(TorchWrapperDataset):
    def __init__(
        self, dataset: torch.utils.data.Dataset, vocab: Vocab, max_seqs: int, *args, **kwargs
    ):
        super().__init__(dataset=dataset, vocab=vocab, *args, **kwargs)
        self.max_seqs = max_seqs

    def __getitem__(self, idx):
        msa = super().__getitem__(idx)
        num_alignments = msa.depth
        if self.max_seqs < num_alignments:
            indices = np.random.randint(0, num_alignments, size=self.max_seqs)
            msa = msa[indices]
        return torch.from_numpy(self.vocab.encode(msa))


class SubsampleMSADataset(BaseWrapperDataset):
    def __init__(
        self,
        dataset: CollatableVocabDataset,
        max_tokens: int,
        max_seqs: Optional[int] = None,
    ):
        super().__init__(dataset)
        self.max_tokens = max_tokens
        self.max_seqs = max_seqs if max_seqs is not None else float("inf")

    def __getitem__(self, idx):
        msa = self.dataset[idx]
        num_alignments, seqlen = msa.size()
        max_alignments = self.max_tokens // seqlen
        max_alignments = min(self.max_seqs, max_alignments)
        if max_alignments < num_alignments:
            indices = np.random.randint(1, num_alignments, size=max_alignments - 1)
            indices = np.append(0, indices)
            msa = msa[indices]
        return msa


class MaskedTokenWrapperDataset(BaseWrapperDataset):
    def __init__(
        self,
        dataset: CollatableVocabDataset,
        mask_prob: float = 0.15,
        random_token_prob: float = 0.1,
        leave_unmasked_prob: float = 0.1,
    ):
        # TODO - add column masking?
        # TODO - add collater
        super().__init__(dataset)
        assert 0 <= mask_prob <= 1
        assert 0 <= random_token_prob <= 1
        assert 0 <= leave_unmasked_prob <= 1

        self._mask_prob = mask_prob
        self._random_token_prob = random_token_prob
        self._leave_unmasked_prob = leave_unmasked_prob

    def __getitem__(self, idx):
        item = self.dataset[idx]
        random_probs = torch.rand_like(item, dtype=torch.float)
        random_probs[(item == self.vocab.bos_idx) | (item == self.vocab.eos_idx)] = 1
        do_mask = random_probs < self.mask_prob

        tgt = item.masked_fill(~do_mask, self.vocab.pad_idx)
        mask_with_token = random_probs < (self.mask_prob * (1 - self.leave_unmasked_prob))
        src = item.masked_fill(mask_with_token, self.vocab.mask_idx)
        mask_with_random = random_probs < (self.mask_prob * self.random_token_prob)
        # TODO - maybe prevent special tokens?
        rand_tokens = torch.randint_like(src, len(self.vocab))
        src[mask_with_random] = rand_tokens[mask_with_random]
        return src, tgt

    @property
    def mask_prob(self) -> float:
        return self._mask_prob

    @property
    def random_token_prob(self) -> float:
        return self._random_token_prob

    @property
    def leave_unmasked_prob(self) -> float:
        return self._leave_unmasked_prob

    def collater(self, batch: List[Any]) -> Any:
        src = collate_tensors(
            [el[0] for el in batch],
            constant_value=self.vocab.pad_idx,
        )
        tgt = collate_tensors(
            [el[1] for el in batch],
            constant_value=self.vocab.pad_idx,
        )
        return src, tgt
