import base64
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fl_utils"))

from s1_compat import S1Tokenizer, configure_s1_tokenizer


class S1TokenizerTests(unittest.TestCase):
    def make_checkpoint(self, directory):
        path = Path(directory)
        ranks = "\n".join(
            f"{base64.b64encode(bytes([value])).decode()} {value}"
            for value in range(256)
        )
        (path / "tokenizer.tiktoken").write_text(ranks, encoding="utf-8")
        special_tokens = {
            "<|im_start|>": 256,
            "<|im_end|>": 257,
            "<|text|>": 258,
            "<|voice|>": 259,
            "<|interleave|>": 260,
            **{f"<|semantic:{index}|>": 261 + index for index in range(4)},
        }
        (path / "special_tokens.json").write_text(json.dumps(special_tokens), encoding="utf-8")
        return path

    def test_loads_tiktoken_checkpoint_and_semantic_range(self):
        with tempfile.TemporaryDirectory() as directory:
            tokenizer = S1Tokenizer.from_pretrained(self.make_checkpoint(directory))

        self.assertEqual(tokenizer.semantic_begin_id, 261)
        self.assertEqual(tokenizer.semantic_end_id, 264)
        self.assertEqual(tokenizer.get_token_id("<|im_end|>"), 257)
        self.assertEqual(tokenizer.decode(tokenizer.encode("test")), "test")

    def test_configures_model_with_checkpoint_tokenizer(self):
        model = SimpleNamespace(
            config=SimpleNamespace(codebook_size=4, semantic_begin_id=0, semantic_end_id=0),
            tokenizer=None,
        )
        with tempfile.TemporaryDirectory() as directory:
            tokenizer = configure_s1_tokenizer(model, self.make_checkpoint(directory))

        self.assertIs(model.tokenizer, tokenizer)
        self.assertEqual(model.config.semantic_begin_id, 261)
        self.assertEqual(model.config.semantic_end_id, 264)


if __name__ == "__main__":
    unittest.main()
