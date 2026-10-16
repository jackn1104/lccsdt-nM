"""Public-interface contract tests; no molecular calculation is launched."""
import importlib.machinery
import importlib.util
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
loader = importlib.machinery.SourceFileLoader("lccsdt_cli", str(ROOT / "lccsdt"))
spec = importlib.util.spec_from_loader(loader.name, loader)
cli = importlib.util.module_from_spec(spec)
loader.exec_module(cli)


class LauncherContract(unittest.TestCase):
    def config(self):
        return dict(storage="memory", symmetry="spin", waters=1, launcher="serial",
                    nodes=1, ranks_per_node=1, cpus_per_task=2, threads=2,
                    stage="full", memory_mib=2048, scratch_mib=4096, identity="fixture",
                    engine="/example/build/lccsdt_engine", basis="avdz")

    def test_geometry(self):
        self.assertEqual(cli.check_geometry(ROOT / "examples/water.xyz"), 1)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.xyz"
            for content in ("2\nbad\nO 0 0 0\nH 0 0 1\n",
                            "3\nbad\nO nan 0 0\nH 0 0 1\nH 0 1 0\n",
                            "3\nbad\nH 0 0 0\nO 0 0 1\nH 0 1 0\n"):
                path.write_text(content)
                with self.assertRaises(ValueError):
                    cli.check_geometry(path)

    def test_conflicting_symmetry(self):
        cfg = self.config()
        cfg.update(symmetry="point-group", storage="disk", waters=4)
        with self.assertRaises(ValueError):
            cli.check_options(cfg)

    def test_full_model_and_restart(self):
        cfg = self.config()
        env = cli.environment(cfg, Path("/run"), True)
        self.assertEqual(env["CCSD_SELECTED_MAXIMUM_MONOMERS"], "full")
        self.assertEqual(env["CCSD_REQUIRE_CONVERGED_RESTART"], "1")
        self.assertEqual(env["CCSD_REFINE_SAVED_CCSD"], "0")
        self.assertEqual(env["CCSD_TRIPLES_TWO_SPIN_SECTORS"], "1")
        self.assertEqual(env["CCSD_CONVERGENCE_THRESHOLD"], "1e-8")
        self.assertEqual(env["CCSD_LAPLACE_TRIPLES_ERROR_TOLERANCE"], "1e-7")
        self.assertIn("required", cli.command(cfg, Path("/run"), True))

    def test_slurm_shape(self):
        cfg = self.config()
        cfg.update(launcher="slurm", nodes=2, ranks_per_node=4, threads=32, cpus_per_task=64)
        cmd = cli.command(cfg, Path("/run"), False)
        self.assertIn("--ntasks=8", cmd)
        self.assertIn("--cpus-per-task=64", cmd)
        self.assertIn("--nodes=2", cmd)

    def test_sector_checkpoint_platform(self):
        cfg = self.config()
        with patch.object(cli.sys, "platform", "darwin"):
            env = cli.environment(cfg, Path("/run"), False)
            self.assertNotIn("CCSD_TRIPLES_SECTOR_CHECKPOINT", env)
            self.assertEqual(env["CCSD_TRIPLES_TWO_SPIN_SECTORS"], "1")
        with patch.object(cli.sys, "platform", "linux"):
            env = cli.environment(cfg, Path("/run"), False)
            self.assertEqual(env["CCSD_TRIPLES_SECTOR_CHECKPOINT"], "/run/triples-sectors.chk")

    def test_ccsd_stage_mapping(self):
        cfg = self.config()
        cfg["stage"] = "ccsd"
        self.assertEqual(cli.environment(cfg, Path("/run"), False)["CCSD_EXECUTION_STAGE"], "ccsd-only")
        self.assertEqual(cli.environment(cfg, Path("/run"), True)["CCSD_EXECUTION_STAGE"], "full")

    def test_output_contract(self):
        text = ("RHF -1.0 Eh\nlocalized CCSD corr -0.1 Eh\nCCSD M1 -0.1 Eh\n"
                "localized (T) -0.01 Eh\n(T) M1 -0.01 Eh\nlocalized CCSD(T) -1.11 Eh\n")
        cli.check_results(text, 1, "full")
        with self.assertRaises(RuntimeError):
            cli.check_results(text.replace("(T) M1 -0.01", "(T) M1 -0.02"), 1, "full")
        with self.assertRaises(RuntimeError):
            cli.check_results(text.replace("localized (T) -0.01", "localized (T) nan"), 1, "full")


if __name__ == "__main__":
    unittest.main()
