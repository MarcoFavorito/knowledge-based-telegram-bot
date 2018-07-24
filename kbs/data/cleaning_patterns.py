import os
import re
import config as c
from constants import old2new

def main():
    s = c.PATTERNS_FOLDER_PATH
    os.listdir(s)
    patterns_files = os.listdir(s)
    res = []

    for pf in patterns_files:
        print("reading ",pf)
        cur_pf = s + pf
        with open(cur_pf) as f:
            for l in f.readlines():
                res.append(l.strip())

    print("Applying some regex for cleaning the patterns...")
    no_dupl = list(set(res))
    print(len(no_dupl))
    if "" in no_dupl: del no_dupl[no_dupl.index("")]


    positive_regex = [
        "^.*(X|Y)+.*(X|Y|)+\t.*$",
        "(\w+)\?\t(\w+)",
    ]

    negative_regex = [
        "\[.*\]",
        # "#",
        # "y/n",
        # '^".+"$',
        # ".*\|.*",
        # "(\Wyes\W|\Wno\W)"
    ]


    replace_regex = [
        (" *\t *", "\t"),
        ("#", ""),
        ("\?\W*(yes|no)$", "\?"),
        ("^([^XY]+)\t(.*)$", "$2\t$1"),
        ("\tcolor$","\tcolorPattern"),
        ("\tisa$", "\tgeneralization"),
        ("\tpart_of$", "\tpart"),
        ("\tsimilar_to$", "\tsimilarity"),
        ("\tsimilar to$", "\tsimilarity"),
        ("\tsepcialization$", "\tspecialization"),
        ("\tcomposition$", "\tpart"),
    ]

    for old, new in old2new.items():
        replace_regex.append(("\t"+old, "\t"+new))
        replace_regex.append(("\t" + old[0].upper() + old[1:], "\t" + new))


    no_dupl_filtered = no_dupl

    # for pr in positive_regex:
    #     no_dupl_filtered = list(filter(lambda x: re.search(pr, x), no_dupl_filtered))
    #     print("after " + pr)
    #     print(len(no_dupl_filtered))

    for before, after in replace_regex:
        no_dupl_filtered = list(map(lambda x: re.sub(before, after, x), no_dupl_filtered))
        print("after " + before + " " + after)
        print(len(no_dupl_filtered))

    for nr in negative_regex:
        no_dupl_filtered = list(filter(lambda x: not re.search(nr, x), no_dupl_filtered))
        print("after " + nr)
        print(len(no_dupl_filtered))

    print(len(no_dupl_filtered))


    qpattern = "^.*(X|Y).*(X|Y|).*\?\t(\w+)$"
    no_dupl_filtered = list(filter(lambda x: re.search(qpattern, x), no_dupl_filtered))
    print("after " + qpattern)
    print(len(no_dupl_filtered))



    with open("kbs/data/cleaned_patterns.tsv", "w") as f:
        f.write("\n".join(no_dupl_filtered))


if __name__ == '__main__':
    main()