from sqlalchemy import create_engine, text
from pandas import pandas

class DBQuery(object):
    def __init__(self, hostname, port, database, schema, db_user, db_password):
        connectionstring = f"postgresql+psycopg2://{db_user}:{db_password}@{hostname}:{port}/{database}"
        self.engine = create_engine(connectionstring, connect_args={"options": f"-c search_path={schema}"})


    def query(self, sql):
        with self.engine.connect() as con:
            return con.execute(text(sql))

    def get_dataframe(self, sql):
        with self.engine.connect() as con:
            return pandas.read_sql(text(sql), con=con)


if __name__ == '__main__':
    import os
    import argparse
    parser = argparse.ArgumentParser(description="Test module")
    parser.add_argument("-H", "--server", action = "store",
                    help="database server name/ip address", default = os.getenv('DB_HOST', 'localhost'))
    parser.add_argument("-P", "--port", action = "store",
                     help = "database server port", default = os.getenv('DB_PORT', 5432))
    parser.add_argument("-D", "--database", action = "store",
                     help = "database name", default = os.getenv('DB', 'sewage'))
    parser.add_argument("-s", "--schema", action = "store",
                     help = "schema name", default = os.getenv('DB_SCHEMA', 'chat'))
    parser.add_argument("-u", "--user", action = "store",
                     help = "database user", default = os.getenv('SECRET_USERNAME'))
    parser.add_argument("-p", "--password", action = "store",
                     help = "database password", default = os.getenv('SECRET_PASSWORD'))

    subparsers = parser.add_subparsers(dest="command", required=True, help="Available commands")
    # query command
    query_parser = subparsers.add_parser("query", help="Run query")
    query_parser.add_argument("-S", "--sql", action = "store",
                     help = "run sql statement", required=True)
    args = parser.parse_args()


    db = DBQuery(hostname=args.server, port=args.port, database=args.database, schema=args.schema, db_user=args.user, db_password=args.password)

    if args.command == "query":
        r= db.query(sql=args.sql)
        print( r.all() )
