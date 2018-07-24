from kbs.DBManager import DBManager
import config as c
from kbs.models import *
from sqlalchemy.orm import load_only

from utils.script import retrieve_concepts, build_data


def main():
    db = DBManager()
    if input("Are you sure you want to DROP ALL the tables? [yes/NO]:")=="yes":
        print("ok, I'm dropping and recreating tables... At your own risk.")
        db.drop_all()
        db.create_all()

    print("Prepare patterns, relation and domain2relations files...")
    build_data.main()
    print("Unpack of the data: done!")

    print("Populating the main relations of the db...")
    db.sync_from_file(
        items_json_file=c.KBS_DUMP_PATH,
        relations_file=c.RELATIONS_PATH,
        patterns_file=c.PATTERNS2RELATIONS_PATH,
        domains_file=c.DOMAINS2RELATIONS_NEW_PATH,
    )

    print("Paginate...")
    new_items = db.sync()
    print("Retrieved ", len(new_items), " items.")

    print("Populating concepts...")
    retrieve_concepts.main()

    print("Done!")




if __name__ == '__main__':
    main()
