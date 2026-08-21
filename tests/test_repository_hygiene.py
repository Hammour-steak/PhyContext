from __future__ import annotations

import re
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {".json", ".md", ".py", ".sh", ".txt", ".toml", ".yaml", ".yml"}
FORBIDDEN = (
    re.compile("/home/" + "yueconghan"),
    re.compile("/mnt/data/" + "yueconghan"),
    re.compile(r"C:\\Users\\" + "11659", re.IGNORECASE),
    re.compile(r"hf_[A-Za-z0-9]{20,}"),
    re.compile(r"-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY-----"),
)


class RepositoryHygieneTest(unittest.TestCase):
    @staticmethod
    def public_files() -> list[Path]:
        output = subprocess.check_output(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
            cwd=ROOT,
        )
        return [ROOT / value.decode() for value in output.split(b"\0") if value]

    def test_public_text_has_no_machine_paths_or_secrets(self) -> None:
        findings = []
        for path in self.public_files():
            if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
                continue
            text = path.read_text(encoding="utf-8")
            for pattern in FORBIDDEN:
                if pattern.search(text):
                    findings.append(str(path.relative_to(ROOT)))
        self.assertEqual(findings, [])

    def test_public_payload_has_no_github_oversized_file(self) -> None:
        oversized = []
        for path in self.public_files():
            if path.stat().st_size >= 95 * 1024 * 1024:
                oversized.append(str(path.relative_to(ROOT)))
        self.assertEqual(oversized, [])

    def test_public_shell_scripts_use_lf_line_endings(self) -> None:
        invalid = [
            str(path.relative_to(ROOT))
            for path in self.public_files()
            if path.is_file() and path.suffix == ".sh" and b"\r\n" in path.read_bytes()
        ]
        self.assertEqual(invalid, [])

    def test_public_training_config_is_not_ignored(self) -> None:
        output = subprocess.run(
            ["git", "check-ignore", "configs/training/one_object.json"],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        self.assertEqual(output.returncode, 1)


if __name__ == "__main__":
    unittest.main()
