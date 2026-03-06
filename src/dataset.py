import json
import jsonlines

import os
import typing
from tqdm.auto import tqdm
from collections import defaultdict

import torch
import numpy as np
import sentencepiece as spm

def load_multiplicity_codes(path: str) -> typing.DefaultDict[str, str]:
    with open(path, "r") as f:
        multiplicity_codes = json.load(f)
    multiplicity_codes = defaultdict(lambda: "m", multiplicity_codes)
    return multiplicity_codes


def process_data(x: dict, multiplicity_codes: typing.Mapping[str, str]) -> dict:
    import copy

    x_copy = copy.deepcopy(x)
    x_copy["spectrum"] = sorted(x_copy["spectrum"], key=lambda s: s["signal"])
    for spectrum in x_copy["spectrum"]:
        spectrum["multiplicity"] = multiplicity_codes[spectrum["multiplicity"]]
        if x_copy["spectrum_type"] == "C_NMR":
            if spectrum["multiplicity"] in {"dt", "td", "ddd"}:
                spectrum["multiplicity"] = "m"
    return x_copy

def load_split(
    path: str,
    multiplicity_codes : typing.Mapping[str, str],
    solvent : str
) -> typing.Dict[str, typing.List[dict]]:
    
    data: typing.DefaultDict[str, typing.List[dict]] = defaultdict(list)
    
    with jsonlines.open(path, "r") as f:
        for obj in tqdm(f, desc=f"Loading {os.path.basename(path)}"):
            data[obj["smiles"]].append(obj)
            
    processed = {
        k: [process_data(v_i, multiplicity_codes) for v_i in v]
        for k, v in tqdm(data.items(), desc="Processing spectra")
    }
    
    for value in processed.values():
        for cur in value:
            cur["solvent"] = solvent
            
    return processed


class NMRDataset(torch.utils.data.Dataset):
    MULTIPLICITY2IDX = {
        "<SYS>": 0,
        "s": 1,
        "d": 2,
        "t": 3,
        "q": 4,
        "dd": 5,
        "dt": 6,
        "td": 7,
        "ddd": 8,
        "m": 9,
    }

    def __init__(
        self,
        data: typing.Dict[str, typing.List[typing.Dict[str, typing.Any]]],
        solvent: str,
        tokenizer: spm.SentencePieceProcessor,
        smiles_bos_id: int,
        smiles_eos_id: int,
    ) -> None:
        self.smiles2idx = {smiles: idx for idx, smiles in enumerate(data.keys())}
        self.idx2smiles = {v: k for k, v in self.smiles2idx.items()}

        self.carbon_spectra = {
            smiles: [
                {
                    "spectrum": torch.tensor(
                        [cur["signal"] for cur in spectrum["spectrum"]],
                        dtype=torch.float32,
                    ),
                    "intensity": torch.tensor(
                        [cur["intensity"] for cur in spectrum["spectrum"]],
                        dtype=torch.float32,
                    ),
                    "multiplicity": torch.tensor(
                        [
                            self.MULTIPLICITY2IDX[cur["multiplicity"]]
                            for cur in spectrum["spectrum"]
                        ],
                        dtype=torch.int64,
                    ),
                }
                for spectrum in spectra
                if spectrum["spectrum_type"] == "C_NMR"
                and spectrum["solvent"] == solvent
            ]
            for smiles, spectra in data.items()
        }

        self.hydrogen_spectra = {
            smiles: [
                {
                    "spectrum": torch.tensor(
                        [cur["signal"] for cur in spectrum["spectrum"]],
                        dtype=torch.float32,
                    ),
                    "intensity": torch.tensor(
                        [cur["intensity"] for cur in spectrum["spectrum"]],
                        dtype=torch.float32,
                    ),
                    "multiplicity": torch.tensor(
                        [
                            self.MULTIPLICITY2IDX[cur["multiplicity"]]
                            for cur in spectrum["spectrum"]
                        ],
                        dtype=torch.int64,
                    ),
                }
                for spectrum in spectra
                if spectrum["spectrum_type"] == "H_NMR"
                and spectrum["solvent"] == solvent
            ]
            for smiles, spectra in data.items()
        }

        self.encoded_smiles = {
            smiles: torch.tensor(
                [smiles_bos_id] + tokenizer.encode(smiles) + [smiles_eos_id],
                dtype=torch.int64,
            )
            for smiles in data.keys()
        }

    def __len__(self) -> int:
        return len(self.smiles2idx)

    @staticmethod
    def _select_spectrum(
        values: typing.List[typing.Dict[str, torch.Tensor]]
    ) -> typing.Optional[typing.Dict[str, torch.Tensor]]:
        size = len(values)
        if size == 0:
            return None
        return values[np.random.randint(low=0, high=size)]

    def __getitem__(self, idx: int) -> typing.Dict[str, typing.Any]:
        smiles = self.idx2smiles[idx]

        carbon_spectrum = self._select_spectrum(self.carbon_spectra[smiles])
        hydrogen_spectrum = self._select_spectrum(self.hydrogen_spectra[smiles])

        if carbon_spectrum is not None and hydrogen_spectrum is not None:
            if np.random.rand() > 0.8:
                if np.random.rand() > 0.5:
                    carbon_spectrum = None
                else:
                    hydrogen_spectrum = None

        return {
            "smiles": self.encoded_smiles[smiles],
            "carbon_spectrum": carbon_spectrum,
            "hydrogen_spectrum": hydrogen_spectrum,
        }


def collate_fn(
    batch: typing.List[typing.Dict[str, typing.Any]],
    smiles_pad_id: int,
    spectrum_pad_token: float,
    max_smiles_len: int,
) -> typing.Dict[str, typing.Any]:
    smiles = []
    carbon_spectrum = []
    carbon_intensity = []
    carbon_multiplicity = []
    hydrogen_spectrum = []
    hydrogen_intensity = []
    hydrogen_multiplicity = []

    for cur in batch:
        smiles.append(cur["smiles"][:max_smiles_len])
        if cur["carbon_spectrum"] is not None:
            carbon_spectrum.append(cur["carbon_spectrum"]["spectrum"])
            carbon_intensity.append(cur["carbon_spectrum"]["intensity"])
            carbon_multiplicity.append(cur["carbon_spectrum"]["multiplicity"])
        else:
            carbon_spectrum.append(
                torch.tensor([spectrum_pad_token], dtype=torch.float32)
            )
            carbon_intensity.append(torch.tensor([0.0], dtype=torch.float32))
            carbon_multiplicity.append(
                torch.tensor(
                    [NMRDataset.MULTIPLICITY2IDX["<SYS>"]], dtype=torch.int64
                )
            )

        if cur["hydrogen_spectrum"] is not None:
            hydrogen_spectrum.append(cur["hydrogen_spectrum"]["spectrum"])
            hydrogen_intensity.append(cur["hydrogen_spectrum"]["intensity"])
            hydrogen_multiplicity.append(cur["hydrogen_spectrum"]["multiplicity"])
        else:
            hydrogen_spectrum.append(
                torch.tensor([spectrum_pad_token], dtype=torch.float32)
            )
            hydrogen_intensity.append(torch.tensor([0.0], dtype=torch.float32))
            hydrogen_multiplicity.append(
                torch.tensor(
                    [NMRDataset.MULTIPLICITY2IDX["<SYS>"]], dtype=torch.int64
                )
            )

    return {
        "smiles": torch.nn.utils.rnn.pad_sequence(
            sequences=smiles,
            batch_first=True,
            padding_value=smiles_pad_id,
        ),
        "C_NMR": {
            "spectrum": torch.nn.utils.rnn.pad_sequence(
                sequences=carbon_spectrum,
                batch_first=True,
                padding_value=spectrum_pad_token,
            ),
            "intensity": torch.nn.utils.rnn.pad_sequence(
                sequences=carbon_intensity,
                batch_first=True,
                padding_value=0.0,
            ),
            "multiplicity": torch.nn.utils.rnn.pad_sequence(
                sequences=carbon_multiplicity,
                batch_first=True,
                padding_value=NMRDataset.MULTIPLICITY2IDX["<SYS>"],
            ),
        },
        "H_NMR": {
            "spectrum": torch.nn.utils.rnn.pad_sequence(
                sequences=hydrogen_spectrum,
                batch_first=True,
                padding_value=spectrum_pad_token,
            ),
            "intensity": torch.nn.utils.rnn.pad_sequence(
                sequences=hydrogen_intensity,
                batch_first=True,
                padding_value=0.0,
            ),
            "multiplicity": torch.nn.utils.rnn.pad_sequence(
                sequences=hydrogen_multiplicity,
                batch_first=True,
                padding_value=NMRDataset.MULTIPLICITY2IDX["<SYS>"],
            ),
        },
    }
