"""Quantum circuit description (QASM) Shannon entropy analysis.

Generates ~1000 quantum circuits of varying types (random, QFT, VQE ansatz,
Grover-style) and varying depth/width, serialises each to QASM, then treats
the entire corpus as a single character stream for NBS entropy analysis.

Two sub-corpora are analysed separately:
  - random_circuits:   Qiskit random_circuit(), width 2-10, depth 2-20
  - structured:        QFT, EfficientSU2 (VQE), Grover oracle, BV, QAOA-style

Output: 1-research/nbs-survey/results/quantum_entropy.json
Data:   1-research/nbs-survey/data/quantum/  (QASM files saved per circuit)
"""

import json
import math
import sys
import random
import warnings
from pathlib import Path

import numpy as np

# Suppress Qiskit deprecation warnings during generation
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ── Qiskit imports ────────────────────────────────────────────────────────────
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.random import random_circuit
from qiskit.circuit.library import (
    efficient_su2,
    real_amplitudes,
    QFTGate,
)
from qiskit.qasm2 import dumps as qasm2_dumps

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent  # 1-research/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from entropy import (
    compute_ngram_counts,
    compute_entropy_miller_madow,
    compute_conditional_entropy,
    compute_structure_score,
    compute_sequential_score,
    shuffle_control,
    compute_mutual_information,
)

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent / "data" / "quantum"
RESULTS_FILE = Path(__file__).parent / "results" / "quantum_entropy.json"
MI_LAGS = [1, 2, 5, 10, 50, 100, 500]
RNG_SEED = 42

DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)

random.seed(RNG_SEED)
np.random.seed(RNG_SEED)


# ── Circuit generators ────────────────────────────────────────────────────────

def gen_random_circuits(n: int = 600) -> list[tuple[str, str]]:
    """Generate n random circuits, varying width (2-10) and depth (2-20).

    Returns list of (label, qasm_string).
    """
    results = []
    widths = list(range(2, 11))   # 2–10 qubits
    depths = list(range(2, 21))   # depth 2–20
    seed = 0
    for i in range(n):
        w = widths[i % len(widths)]
        d = depths[i % len(depths)]
        try:
            qc = random_circuit(w, d, measure=True, seed=seed)
            qasm = qasm2_dumps(qc)
            results.append((f"random_w{w}_d{d}_{i}", qasm))
        except Exception as e:
            pass  # some random circuits may not be QASM2-serialisable
        seed += 1
    print(f"Generated {len(results):,} random circuits")
    return results


def gen_qft_circuits() -> list[tuple[str, str]]:
    """Generate QFT circuits for widths 2–12."""
    results = []
    for n in range(2, 13):
        try:
            qc = QuantumCircuit(n, n)
            qc.append(QFTGate(n), range(n))
            qc = qc.decompose()
            qc.measure(range(n), range(n))
            qasm = qasm2_dumps(qc)
            results.append((f"qft_n{n}", qasm))
        except Exception:
            # fallback: manual H+CP construction
            try:
                qc = QuantumCircuit(n, n)
                for i in range(n):
                    qc.h(i)
                    for j in range(i + 1, n):
                        qc.cp(math.pi / (2 ** (j - i)), i, j)
                qc.measure(range(n), range(n))
                qasm = qasm2_dumps(qc)
                results.append((f"qft_manual_n{n}", qasm))
            except Exception:
                pass
    print(f"Generated {len(results):,} QFT circuits")
    return results


def gen_vqe_ansatz_circuits(n_per_width: int = 30) -> list[tuple[str, str]]:
    """Generate EfficientSU2 (VQE) ansatz circuits with random parameter binding.

    Widths 2–8, reps 1–3.
    """
    results = []
    for n in range(2, 9):
        for reps in range(1, 4):
            for trial in range(n_per_width):
                try:
                    qc = efficient_su2(n, reps=reps)
                    params = np.random.uniform(0, 2 * math.pi, qc.num_parameters)
                    bound = qc.assign_parameters(params)
                    bound.measure_all()
                    qasm = qasm2_dumps(bound)
                    results.append((f"vqe_n{n}_reps{reps}_{trial}", qasm))
                except Exception:
                    pass
    print(f"Generated {len(results):,} VQE ansatz circuits")
    return results


def gen_grover_oracle_circuits() -> list[tuple[str, str]]:
    """Generate simple Grover-style oracle circuits (diffuser + oracle pattern).

    Uses multi-qubit controlled-Z gates as oracle for various target states.
    Widths 2–8.
    """
    results = []
    for n in range(2, 9):
        # Target states: all-zeros, all-ones, random balanced states
        for target_bits in [0, (1 << n) - 1, random.randint(1, (1 << n) - 2)]:
            try:
                qc = QuantumCircuit(n, n)
                # Initial superposition
                qc.h(range(n))
                # Oracle: flip phase of target state
                target_str = format(target_bits, f"0{n}b")
                for i, bit in enumerate(reversed(target_str)):
                    if bit == "0":
                        qc.x(i)
                if n == 2:
                    qc.cz(0, 1)
                elif n == 3:
                    qc.ccz(0, 1, 2)
                else:
                    qc.h(n - 1)
                    qc.mcx(list(range(n - 1)), n - 1)
                    qc.h(n - 1)
                for i, bit in enumerate(reversed(target_str)):
                    if bit == "0":
                        qc.x(i)
                # Diffuser
                qc.h(range(n))
                qc.x(range(n))
                if n == 2:
                    qc.cz(0, 1)
                elif n == 3:
                    qc.ccz(0, 1, 2)
                else:
                    qc.h(n - 1)
                    qc.mcx(list(range(n - 1)), n - 1)
                    qc.h(n - 1)
                qc.x(range(n))
                qc.h(range(n))
                qc.measure(range(n), range(n))
                qasm = qasm2_dumps(qc)
                results.append((f"grover_n{n}_target{target_bits}", qasm))
            except Exception:
                pass
    print(f"Generated {len(results):,} Grover-style circuits")
    return results


def gen_bv_circuits() -> list[tuple[str, str]]:
    """Bernstein-Vazirani circuits for hidden strings of length 2–10."""
    results = []
    for n in range(2, 11):
        for trial in range(5):
            hidden = random.randint(0, (1 << n) - 1)
            try:
                qc = QuantumCircuit(n + 1, n)
                # Ancilla |->
                qc.x(n)
                qc.h(n)
                # Superposition on query register
                qc.h(range(n))
                # Oracle: CNOT for each set bit in hidden
                for i in range(n):
                    if hidden & (1 << i):
                        qc.cx(i, n)
                # Inverse Hadamard
                qc.h(range(n))
                qc.measure(range(n), range(n))
                qasm = qasm2_dumps(qc)
                results.append((f"bv_n{n}_hidden{hidden}_{trial}", qasm))
            except Exception:
                pass
    print(f"Generated {len(results):,} Bernstein-Vazirani circuits")
    return results


def gen_qaoa_circuits() -> list[tuple[str, str]]:
    """QAOA-style circuits (parameterized ZZ + RX mixer) with random angles.

    Widths 2–8, p-layers 1–4.
    """
    results = []
    for n in range(2, 9):
        for p in range(1, 5):
            for trial in range(10):
                try:
                    qc = QuantumCircuit(n, n)
                    qc.h(range(n))
                    # p QAOA layers
                    for layer in range(p):
                        gamma = np.random.uniform(0, 2 * math.pi)
                        beta = np.random.uniform(0, 2 * math.pi)
                        # Cost layer: ZZ couplings on edges of a ring
                        for i in range(n):
                            j = (i + 1) % n
                            qc.cx(i, j)
                            qc.rz(2 * gamma, j)
                            qc.cx(i, j)
                        # Mixer layer: RX rotations
                        for i in range(n):
                            qc.rx(2 * beta, i)
                    qc.measure(range(n), range(n))
                    qasm = qasm2_dumps(qc)
                    results.append((f"qaoa_n{n}_p{p}_{trial}", qasm))
                except Exception:
                    pass
    print(f"Generated {len(results):,} QAOA circuits")
    return results


# ── QASM corpus utilities ─────────────────────────────────────────────────────

def build_corpus(circuits: list[tuple[str, str]], save_dir: Path | None = None) -> str:
    """Concatenate QASM strings into one stream, optionally saving each to disk."""
    parts = []
    for label, qasm in circuits:
        parts.append(qasm)
        if save_dir is not None:
            (save_dir / f"{label}.qasm").write_text(qasm, encoding="utf-8")
    return "\n".join(parts)


def report_alphabet(sequence: str) -> dict:
    counts: dict[str, int] = {}
    for ch in sequence:
        counts[ch] = counts.get(ch, 0) + 1
    sorted_counts = sorted(counts.items(), key=lambda x: -x[1])
    print("\nTop 25 characters by frequency:")
    for ch, n in sorted_counts[:25]:
        display = repr(ch)
        print(f"  {display:12s}: {n:9,}  ({100 * n / len(sequence):5.2f}%)")
    return counts


def run_entropy_analysis(label: str, seq: list) -> dict:
    """Run the full NBS entropy pipeline on a character list."""
    print(f"\n{'-'*50}")
    print(f"  Sub-corpus: {label}  (n={len(seq):,} chars)")
    print("-" * 50)

    # H0
    unigram_counts = compute_ngram_counts(seq, 1)
    h0 = compute_entropy_miller_madow(unigram_counts)
    alphabet_size = len(unigram_counts)
    max_h0 = math.log2(alphabet_size) if alphabet_size > 1 else 0.0
    print(f"H0 = {h0:.4f} bits  (max {max_h0:.4f} for {alphabet_size} symbols)")

    # H2, H3
    h2 = compute_conditional_entropy(seq, order=2)
    h3 = compute_conditional_entropy(seq, order=3)
    print(f"H2 = {h2:.4f} bits,  H3 = {h3:.4f} bits")

    # Structure + sequential scores
    structure_score = compute_structure_score(seq)
    sequential_score = compute_sequential_score(seq)
    print(f"structure_score = {structure_score:.4f},  sequential_score = {sequential_score:.4f}")

    # Shuffled control
    shuffled = shuffle_control(seq, seed=RNG_SEED)
    h0_sh = compute_entropy_miller_madow(compute_ngram_counts(shuffled, 1))
    h3_sh = compute_conditional_entropy(shuffled, order=3)
    ss_sh = max(0.0, 1 - h3_sh / h0_sh) if h0_sh > 0 else 0.0
    print(f"Shuffled: H0={h0_sh:.4f}, H3={h3_sh:.4f}, structure_score={ss_sh:.6f}")

    # MI profile
    mi_profile: dict[int, float] = {}
    print("MI decay:")
    for lag in MI_LAGS:
        mi = compute_mutual_information(seq, lag)
        mi_profile[lag] = mi
        print(f"  MI(lag={lag:4d}) = {mi:.4f} bits")

    return {
        "n_chars": len(seq),
        "alphabet_size": alphabet_size,
        "h0": round(h0, 6),
        "h2": round(h2, 6),
        "h3": round(h3, 6),
        "structure_score": round(structure_score, 6),
        "sequential_score": round(sequential_score, 6),
        "shuffled_control": {
            "h0": round(h0_sh, 6),
            "h3": round(h3_sh, 6),
            "structure_score": round(ss_sh, 6),
        },
        "mi_profile": {str(k): round(v, 6) for k, v in mi_profile.items()},
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Quantum Circuit (QASM) Entropy Analysis")
    print("=" * 60)

    # ── Generate circuits ─────────────────────────────────────────────────────
    print("\n--- Generating circuits ---")
    random_circs = gen_random_circuits(600)
    qft_circs = gen_qft_circuits()
    vqe_circs = gen_vqe_ansatz_circuits(30)
    grover_circs = gen_grover_oracle_circuits()
    bv_circs = gen_bv_circuits()
    qaoa_circs = gen_qaoa_circuits()

    structured_circs = qft_circs + vqe_circs + grover_circs + bv_circs + qaoa_circs
    all_circs = random_circs + structured_circs

    print(f"\nTotal circuits: {len(all_circs):,}  "
          f"(random={len(random_circs)}, structured={len(structured_circs)})")

    # ── Build corpora ─────────────────────────────────────────────────────────
    print("\n--- Building QASM text corpora ---")
    random_corpus = build_corpus(random_circs, save_dir=DATA_DIR)
    structured_corpus = build_corpus(structured_circs)
    pooled_corpus = build_corpus(all_circs)

    print(f"Random corpus:     {len(random_corpus):>12,} chars")
    print(f"Structured corpus: {len(structured_corpus):>12,} chars")
    print(f"Pooled corpus:     {len(pooled_corpus):>12,} chars")

    # ── Alphabet report (pooled) ──────────────────────────────────────────────
    print("\n--- Alphabet (pooled corpus) ---")
    alpha_counts = report_alphabet(pooled_corpus)

    # ── Entropy analysis ──────────────────────────────────────────────────────
    print("\n=== Entropy analysis: POOLED (all circuits) ===")
    pooled_results = run_entropy_analysis("pooled", list(pooled_corpus))

    print("\n=== Entropy analysis: RANDOM circuits only ===")
    random_results = run_entropy_analysis("random_circuits", list(random_corpus))

    print("\n=== Entropy analysis: STRUCTURED circuits only ===")
    structured_results = run_entropy_analysis("structured_circuits", list(structured_corpus))

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, res in [("Pooled", pooled_results),
                      ("Random circuits", random_results),
                      ("Structured circuits", structured_results)]:
        print(f"  {name:25s}:  structure_score={res['structure_score']:.3f}  "
              f"H0={res['h0']:.2f}  H3={res['h3']:.2f}  "
              f"alphabet={res['alphabet_size']}  n={res['n_chars']:,}")

    # ── Assemble results JSON ─────────────────────────────────────────────────
    results = {
        "domain": "Quantum circuit descriptions (OpenQASM 2.0)",
        "data_source": (
            "Generated with Qiskit 2.x circuit library. "
            f"Total {len(all_circs):,} circuits: "
            f"{len(random_circs)} random (width 2-10, depth 2-20), "
            f"{len(qft_circs)} QFT (n=2-12), "
            f"{len(vqe_circs)} VQE/EfficientSU2 (n=2-8, reps=1-3, random params), "
            f"{len(grover_circs)} Grover-style (n=2-8), "
            f"{len(bv_circs)} Bernstein-Vazirani (n=2-10), "
            f"{len(qaoa_circs)} QAOA-style (n=2-8, p=1-4, random angles). "
            "All serialised to OpenQASM 2.0, concatenated into character stream."
        ),
        "encoding_notes": (
            "QASM is text-based: gate names (h, cx, rz, ccx, ...), qubit indices "
            "q[0], parameter expressions (pi/2, numeric floats), register declarations, "
            "semicolons, whitespace. Treated as raw character stream — character-level "
            "entropy measures the notation's sequential constraint. "
            "Pooled counting across all circuits. "
            "Two sub-corpora separated: random circuits vs algorithmically structured circuits."
        ),
        "n_circuits_total": len(all_circs),
        "n_circuits_random": len(random_circs),
        "n_circuits_structured": len(structured_circs),
        "circuit_type_breakdown": {
            "random": len(random_circs),
            "qft": len(qft_circs),
            "vqe_efficient_su2": len(vqe_circs),
            "grover_style": len(grover_circs),
            "bernstein_vazirani": len(bv_circs),
            "qaoa_style": len(qaoa_circs),
        },
        "top_chars_by_freq_pooled": {
            k: int(v)
            for k, v in sorted(alpha_counts.items(), key=lambda x: -x[1])[:30]
        },
        "pooled": pooled_results,
        "random_circuits": random_results,
        "structured_circuits": structured_results,
        "primary_structure_score": pooled_results["structure_score"],
        "primary_sequential_score": pooled_results["sequential_score"],
    }

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {RESULTS_FILE}")
    print(f"QASM files saved to {DATA_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
