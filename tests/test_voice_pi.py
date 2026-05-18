from __future__ import annotations

import importlib
import io
import os
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stderr, contextmanager
from unittest.mock import patch


def load_voice_pi(cuda_devices: int = 0):
    for name in ("voice_pi", "ctranslate2", "faster_whisper", "numpy",
                 "sounddevice", "pynput", "pynput.keyboard"):
        sys.modules.pop(name, None)

    ctranslate2 = types.ModuleType("ctranslate2")
    ctranslate2.get_cuda_device_count = lambda: cuda_devices
    sys.modules["ctranslate2"] = ctranslate2

    faster_whisper = types.ModuleType("faster_whisper")
    faster_whisper.WhisperModel = object
    sys.modules["faster_whisper"] = faster_whisper

    sys.modules["numpy"] = types.ModuleType("numpy")
    sys.modules["sounddevice"] = types.ModuleType("sounddevice")

    pynput = types.ModuleType("pynput")
    keyboard = types.ModuleType("keyboard")
    keyboard.Controller = object
    keyboard.Key = types.SimpleNamespace(
        ctrl_l=object(), ctrl_r=object(),
        shift_l=object(), shift_r=object(),
        alt_l=object(), alt_r=object(),
        esc=object(),
    )
    keyboard.Listener = object
    pynput.keyboard = keyboard
    sys.modules["pynput"] = pynput
    sys.modules["pynput.keyboard"] = keyboard

    return importlib.import_module("voice_pi")


class DeviceResolutionTests(unittest.TestCase):
    def test_auto_uses_cuda_when_available(self):
        voice_pi = load_voice_pi(cuda_devices=1)

        self.assertEqual(
            voice_pi._resolve_device("auto"),
            ("cuda", "int8_float16"),
        )

    def test_auto_falls_back_to_cpu_without_cuda(self):
        voice_pi = load_voice_pi(cuda_devices=0)

        self.assertEqual(voice_pi._resolve_device("auto"), ("cpu", "int8"))

    def test_explicit_cpu_and_cuda(self):
        voice_pi = load_voice_pi()

        self.assertEqual(voice_pi._resolve_device("cpu"), ("cpu", "int8"))
        self.assertEqual(
            voice_pi._resolve_device("cuda"),
            ("cuda", "int8_float16"),
        )

    def test_invalid_device_is_rejected(self):
        voice_pi = load_voice_pi()

        with self.assertRaises(ValueError):
            voice_pi._resolve_device("cdua")


class ArgumentParserTests(unittest.TestCase):
    def test_parser_rejects_invalid_device(self):
        voice_pi = load_voice_pi()
        parser = voice_pi.build_arg_parser()

        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parser.parse_args(["--device", "cdua"])

    def test_parser_accepts_supported_devices(self):
        voice_pi = load_voice_pi()
        parser = voice_pi.build_arg_parser()

        for device in voice_pi.VALID_DEVICES:
            with self.subTest(device=device):
                self.assertEqual(
                    parser.parse_args(["--device", device]).device,
                    device,
                )


@contextmanager
def _env(**kwargs):
    old = {k: os.environ.get(k) for k in kwargs}
    os.environ.update(kwargs)
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class BuildYdotoolOpsTests(unittest.TestCase):
    """_build_ydotool_ops: tekst → liste af (subkommando, *args) tupler."""

    def setUp(self):
        self.vp = load_voice_pi()
        self.dk = self.vp._LAYOUT_KEYCODES['dk']

    def test_ascii_only_is_single_type_op(self):
        ops = self.vp._build_ydotool_ops("hello", {})
        self.assertEqual(ops, [('type', '--', 'hello')])

    def test_empty_string_gives_no_ops(self):
        ops = self.vp._build_ydotool_ops("", {})
        self.assertEqual(ops, [])

    def test_oe_splits_into_key_op(self):
        ops = self.vp._build_ydotool_ops("ø", self.dk)
        self.assertEqual(ops, [('key', '40:1', '40:0')])

    def test_mixed_flushes_ascii_buffer_before_special(self):
        # "høre" → type "h", key ø, type "re"
        ops = self.vp._build_ydotool_ops("høre", self.dk)
        self.assertEqual(ops, [
            ('type', '--', 'h'),
            ('key', '40:1', '40:0'),
            ('type', '--', 're'),
        ])

    def test_question_mark_uses_nordic_keycode(self):
        # '?' er shift+KEY_MINUS i nordiske layouts, ikke shift+KEY_SLASH
        ops = self.vp._build_ydotool_ops("hvad?", self.dk)
        self.assertEqual(ops, [
            ('type', '--', 'hvad'),
            ('key', '42:1', '12:1', '12:0', '42:0'),
        ])

    def test_consecutive_special_chars_each_get_key_op(self):
        ops = self.vp._build_ydotool_ops("æøå", self.dk)
        self.assertEqual(ops, [
            ('key', '39:1', '39:0'),  # æ
            ('key', '40:1', '40:0'),  # ø
            ('key', '26:1', '26:0'),  # å
        ])

    def test_uppercase_special_char(self):
        ops = self.vp._build_ydotool_ops("Ø", self.dk)
        self.assertEqual(ops, [('key', '42:1', '40:1', '40:0', '42:0')])

    def test_ascii_after_special_is_flushed(self):
        ops = self.vp._build_ydotool_ops("åben", self.dk)
        self.assertEqual(ops, [
            ('key', '26:1', '26:0'),  # å
            ('type', '--', 'ben'),
        ])

    def test_no_map_passthrough(self):
        # Uden keycode_map (f.eks. us-layout) → alt sendes som type
        ops = self.vp._build_ydotool_ops("høre", {})
        self.assertEqual(ops, [('type', '--', 'høre')])


class LayoutKeycodeMapTests(unittest.TestCase):
    """Mapningens indhold: hvert layout har de forventede specialtegn."""

    def setUp(self):
        self.vp = load_voice_pi()

    def _assert_has_chars(self, layout: str, chars: str):
        m = self.vp._LAYOUT_KEYCODES[layout]
        for ch in chars:
            with self.subTest(layout=layout, char=ch):
                self.assertIn(ch, m)

    def test_dk_has_ae_oe_aa(self):
        self._assert_has_chars('dk', 'æøåÆØÅ')

    def test_no_aliases_dk(self):
        self.assertIs(
            self.vp._LAYOUT_KEYCODES['no'],
            self.vp._LAYOUT_KEYCODES['dk'],
        )

    def test_se_has_ae_oe_aa(self):
        self._assert_has_chars('se', 'äöåÄÖÅ')

    def test_de_has_umlauts(self):
        self._assert_has_chars('de', 'äöüÄÖÜ')

    def test_fi_has_ae_oe(self):
        self._assert_has_chars('fi', 'äöÄÖ')

    def test_all_layouts_have_nordic_punct(self):
        punct = '?-_:;/"'
        for layout in ('dk', 'no', 'se', 'de', 'fi'):
            self._assert_has_chars(layout, punct)

    def test_es_has_n_tilde_and_accented_vowels(self):
        self._assert_has_chars('es', 'ñÑáéíóúÁÉÍÓÚüÜ')

    def test_pt_has_cedilla_and_accented_vowels(self):
        self._assert_has_chars('pt', 'çÇáéíóúÁÉÍÓÚàÀãõÃÕâêôÂÊÔ')

    def test_br_has_cedilla_tilde_circumflex(self):
        self._assert_has_chars('br', 'çÇãõÃÕâêôÂÊÔáéíóúÁÉÍÓÚ')

    def test_pl_has_polish_chars(self):
        self._assert_has_chars('pl', 'ąęóśźżćńłĄĘÓŚŹŻĆŃŁ')

    def test_ua_has_full_cyrillic_alphabet(self):
        self._assert_has_chars('ua', 'йцукенгшщзхїфівапролджєґячсмитьбюЙЦУКЕНГШЩЗХЇФІВАПРОЛДЖЄҐЯЧСМИТЬБЮ')

    def test_ru_not_in_lang_to_xkb(self):
        self.assertNotIn('ru', self.vp._LANG_TO_XKB)

    def test_uk_maps_to_ua(self):
        self.assertEqual(self.vp._LANG_TO_XKB.get('uk'), 'ua')

    def test_keycodes_are_balanced_per_key(self):
        # Stærkere end "lige antal codes": hvert keycode skal have lige
        # mange press (N:1) og release (N:0) i en sekvens — ellers hænger
        # fx Shift(42)/AltGr(100) og korrumperer efterfølgende input.
        import collections
        for layout, m in self.vp._LAYOUT_KEYCODES.items():
            for ch, codes in m.items():
                with self.subTest(layout=layout, char=ch):
                    bal: "collections.Counter[str]" = collections.Counter()
                    for tok in codes:
                        key, sep, state = tok.partition(":")
                        self.assertTrue(sep and state in ("0", "1"),
                                        f"Ugyldig token {tok!r} for '{ch}'")
                        bal[key] += 1 if state == "1" else -1
                    for key, net in bal.items():
                        self.assertEqual(
                            net, 0,
                            f"Keycode {key} ubalanceret for '{ch}' i "
                            f"layout '{layout}' (net={net} press-release)")


class DetectXkbLayoutTests(unittest.TestCase):
    """_detect_xkb_layout: prioritetsrækkefølge og fallback."""

    def setUp(self):
        self.vp = load_voice_pi()
        # Ryd env-variabler der ellers forstyrrer
        self._patches = [
            patch.dict(os.environ, {}, clear=False),
        ]
        for p in self._patches:
            p.start()
        os.environ.pop('VOICEPI_XKB_LAYOUT', None)
        os.environ.pop('XKB_DEFAULT_LAYOUT', None)

    def tearDown(self):
        for p in self._patches:
            p.stop()

    def test_voicepi_env_var_takes_priority(self):
        with _env(VOICEPI_XKB_LAYOUT='se', XKB_DEFAULT_LAYOUT='de'):
            result = self.vp._detect_xkb_layout('da')
        self.assertEqual(result, 'se')

    def test_xkb_default_layout_beats_keyboard_file(self):
        with _env(XKB_DEFAULT_LAYOUT='de'):
            with patch('builtins.open', side_effect=FileNotFoundError):
                result = self.vp._detect_xkb_layout('da')
        self.assertEqual(result, 'de')

    def test_keyboard_file_parsed_correctly(self):
        content = 'XKBLAYOUT="dk"\nXKBVARIANT=""\n'
        with patch('builtins.open',
                   unittest.mock.mock_open(read_data=content)):
            result = self.vp._detect_xkb_layout(None)
        self.assertEqual(result, 'dk')

    def test_us_layout_in_keyboard_file_is_ignored(self):
        content = 'XKBLAYOUT="us"\n'
        with patch('builtins.open',
                   unittest.mock.mock_open(read_data=content)):
            result = self.vp._detect_xkb_layout('da')
        # Falder igennem til lang-hint: da → dk
        self.assertEqual(result, 'dk')

    def test_lang_hint_da_gives_dk(self):
        with patch('builtins.open', side_effect=FileNotFoundError):
            result = self.vp._detect_xkb_layout('da')
        self.assertEqual(result, 'dk')

    def test_lang_hint_nb_gives_no(self):
        with patch('builtins.open', side_effect=FileNotFoundError):
            result = self.vp._detect_xkb_layout('nb')
        self.assertEqual(result, 'no')

    def test_no_hints_returns_none(self):
        with patch('builtins.open', side_effect=FileNotFoundError):
            result = self.vp._detect_xkb_layout(None)
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
