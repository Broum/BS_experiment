"""Qubit selection from the stored calibration files."""

import unittest

import numpy as np

from utils import qubit_select as qs
from utils import vlq
from utils.bs_circuits import make_bs


class QubitSelectTest(unittest.TestCase):
    def test_errors_all_files(self):
        for folder, n in (("VLQ", 24), ("SIRIUS", 16)):
            for path in (qs.CALIBRATION_DIR / folder).glob("*.json"):
                obs, cal_id, _ = qs.load_calibration(path)
                errors = qs.qubit_errors(obs)
                self.assertEqual(len(errors), n, path)
                self.assertTrue(cal_id)
                for e in errors.values():  # every metric present and plausible
                    self.assertTrue(all(0 < e[k] < 0.2 for k in ("prx", "cz", "move", "readout")), path)

    def test_best_layout_beats_default(self):
        backend = vlq.fake_vlq()
        from utils.fake_vlq_backend import CALIBRATION_FILE

        errors = qs.qubit_errors(qs.load_calibration(qs.CALIBRATION_DIR / "VLQ" / CALIBRATION_FILE)[0])
        qc = make_bs(np.pi / 8, "simplified")
        (best, layout), *_ = qs.best_layouts(qc, backend, errors, n_candidates=4, seed=1)
        tr, default = vlq.transpile([qc], backend, seed=1)
        self.assertEqual(len(set(layout)), 4)
        self.assertGreater(best, qs.esp(tr[0], backend, errors))


if __name__ == "__main__":
    unittest.main()
