import unittest
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = REPOSITORY_ROOT / "scripts" / "windows" / "start-worker.ps1"


class WindowsWorkerLauncherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.script = LAUNCHER.read_text(encoding="ascii")

    def test_launcher_is_ascii_and_locates_repository_from_script(self):
        self.assertTrue(LAUNCHER.is_file())
        self.assertIn('$PSScriptRoot "..\\.."', self.script)
        self.assertIn('"project\\.env"', self.script)

    def test_launcher_does_not_require_conda_activation(self):
        self.assertNotIn("conda activate", self.script.lower())
        self.assertIn("run --no-capture-output -n $EnvironmentName", self.script)
        self.assertIn("rag-qa --workspace $repositoryRoot worker serve", self.script)

    def test_launcher_discovers_tailscale_and_checks_port(self):
        self.assertIn("tailscale.exe", self.script)
        self.assertIn("Resolve-TailscaleAddress", self.script)
        self.assertIn("Test-TcpPortInUse", self.script)

    def test_launcher_keeps_token_private_and_uses_dotenv_as_source(self):
        self.assertIn("Assert-WorkerTokenConfigured", self.script)
        self.assertIn('Remove-Item "Env:$_"', self.script)
        self.assertNotIn("Write-Host $tokenValue", self.script)
        self.assertIn("Worker token: configured in project/.env", self.script)


if __name__ == "__main__":
    unittest.main()
