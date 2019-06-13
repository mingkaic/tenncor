# non-hermatic
def genyacc(name, out_hdr, out_src, src):
    cmd = "yacc --defines=$(location %s) -o $(location %s) $(location %s)" %\
        (out_hdr, out_src, src)
    native.genrule(
        name = name,
        outs = [out_hdr, out_src],
        srcs = [src],
        cmd = cmd,
    )

def genlex(name, out, src):
    cmd = "flex -o $(location %s) $(location %s)" % (out, src)
    native.genrule(
        name = name,
        outs = [out],
        srcs = [src],
        cmd = cmd,
    )
