"""The evidence core must stay free of any model.

The decision this system makes is a threshold crossing on a ledger of reports, and its
value to an operator is that the same ledger yields the same verdict on every run. A
model anywhere on that path ends the guarantee quietly, because nothing about the output
announces that it has become unreproducible.

So the property is asserted against the package instead of being remembered. The list is
deliberately wider than any one vendor: what matters is that the decision path stays
arithmetic, not which library would have broken it.
"""
import inspect

from cfas import assimilate, generalise, hazard, neutral, waggle

DECISION_PATH = (waggle, neutral, hazard, assimilate, generalise)

MODEL_LIBRARIES = ("anthropic", "openai", "cohere", "mistralai", "ollama",
                   "google.generativeai", "llama_cpp", "torch", "tensorflow")


def test_the_evidence_core_has_no_model_in_its_import_graph():
    for mod in DECISION_PATH:
        src = inspect.getsource(mod)
        for lib in MODEL_LIBRARIES:
            assert lib not in src, f"{mod.__name__} reached for {lib}"
