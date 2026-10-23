"""Public behavior of the small model-independent utility nodes."""

import unittest

import torch

from _sampling_support import load_implementation


class CheckerboardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.node = load_implementation(
            "gen2/misc_nodes/checkerboard.py", "_gen2_test_checkerboard",
        ).Gen2_Checkerboard()

    def test_partial_blocks_keep_image_dimensions_and_rgb_contract(self):
        image, = self.node.generate(width=5, height=3, block_size=2)
        expected = torch.tensor([[0, 0, 1, 1, 0], [0, 0, 1, 1, 0], [1, 1, 0, 0, 1]], dtype=torch.float32)
        self.assertEqual(image.shape, (1, 3, 5, 3))
        self.assertEqual(image.dtype, torch.float32)
        self.assertEqual(image.device.type, "cpu")
        for channel in range(3):
            torch.testing.assert_close(image[0, :, :, channel], expected)

    def test_one_pixel_blocks_alternate_in_both_directions(self):
        image, = self.node.generate(width=3, height=2, block_size=1)
        torch.testing.assert_close(image[0, :, :, 0], torch.tensor([[0., 1., 0.], [1., 0., 1.]]))

    def test_block_larger_than_image_is_solid_black(self):
        image, = self.node.generate(width=1, height=2, block_size=64)
        self.assertEqual(torch.count_nonzero(image).item(), 0)


class StringReplaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.node = load_implementation(
            "gen2/misc_nodes/string_replace.py", "_gen2_test_string_replace",
        ).Gen2_StringReplace()

    def test_replaces_all_exact_literals_without_regex_interpretation(self):
        self.assertEqual(self.node.replace_string("[a].* [a].* [A].*", "[a].*", "cat"), ("cat cat [A].*",))

    def test_empty_search_is_a_noop(self):
        self.assertEqual(self.node.replace_string("cat", "", "x"), ("cat",))

    def test_unicode_multiline_and_empty_replacement(self):
        self.assertEqual(self.node.replace_string("猫\n猫 dog", "猫", ""), ("\n dog",))

    def test_unmatched_text_and_empty_input(self):
        self.assertEqual(self.node.replace_string("cat", "dog", "bird"), ("cat",))
        self.assertEqual(self.node.replace_string("", "cat", "dog"), ("",))


if __name__ == "__main__":
    unittest.main()
