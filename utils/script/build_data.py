import config
import kbs.data.cleaning_patterns as cleaning_patterns
import constants as c



def build_relations_file():
    with open(config.DOMAINS2RELATIONS_OLD_PATH, "r") as fr:
        relations = []
        for l in fr.readlines():
            rels = l.strip().split("\t")
            relations += rels[1:]
        relations = set(relations)

    try:
        relations.remove("")
    except:
        pass

    relations = map(lambda x: c.old2new[x[0].lower() + x[1:]], relations)

    with open(config.RELATIONS_PATH, "w") as fw:
        fw.write("\n".join(relations))


def build_domains2relations_file():
    with open(config.DOMAINS2RELATIONS_OLD_PATH, "r") as fr:
        domains2relations = {}
        for l in fr.readlines():
            l = l.strip()
            spl_l = l.split("\t")
            domain_simple_name = (spl_l[0]).strip()
            relation_simple_names = spl_l[1:]
            try:
                relation_simple_names.remove("")
            except:
                pass
            domains2relations[domain_simple_name]=list(map(lambda x: c.old2new[x],relation_simple_names))
        domains2relations["Geography and places"].append("PLACE")
    with open(config.DOMAINS2RELATIONS_NEW_PATH, "w") as fw:
        fw.write(
            "\n".join(
                    [d+"\t"+"\t".join(rels)
                        for d, rels in domains2relations.items()]
                      )
        )


def main():
    cleaning_patterns.main()
    build_relations_file()
    build_domains2relations_file()

if __name__ == '__main__':
    main()


