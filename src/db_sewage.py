from sqlalchemy import create_engine, text
from pandas import pandas

class DBQuery(object):
    def __init__(self, hostname, port, database, schema, db_user, db_password):
        connectionstring = f"postgresql+psycopg2://{db_user}:{db_password}@{hostname}:{port}/{database}"
        self.engine = create_engine(connectionstring, connect_args={"options": f"-c search_path={schema}"})
        self.schema=schema

    def query(self, sql, subst={}):
        with self.engine.connect() as con:
            return con.execute(text(sql), subst)

    def query_to_df(self, sql, subst={}):
        with self.engine.connect() as con:
            return pandas.DataFrame(con.execute(text(sql), subst))
        
    def get_dialect(self):
        return self.engine.dialect.name

    def describe_tables(self):
        q="""
SELECT c.relname AS table_name, obj_description(c.oid) AS table_comment
FROM pg_class c
JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE c.relkind in ('r', 'v') AND n.nspname = :schema
ORDER BY c.relname;
        """
        r=self.query(q, {'schema': self.schema})
        return r.fetchall()

    def describe_columns(self):
        q="""
SELECT c.relname AS table_name, a.attname AS column_name, col_description(a.attrelid, a.attnum) AS description, pg_catalog.format_type(a.atttypid, a.atttypmod) AS datatype
FROM pg_class c
JOIN pg_namespace n ON n.oid = c.relnamespace
JOIN pg_attribute a ON a.attrelid = c.oid
WHERE c.relkind in ('r', 'v') AND n.nspname = :schema AND a.attnum > 0 AND NOT a.attisdropped
ORDER BY c.relname, a.attnum;
        """
        r=self.query(q, {'schema': self.schema})
        return r.fetchall()

    @staticmethod
    def as_str(lst):
        return "\n".join(map(lambda x: '\n'.join(x), lst))

    def DDL_tables(self):
# newer postgres may support
#SELECT '--- TABLE: ' || c.relname AS ddl_header,
#       pg_get_tabledef(c.oid)
#FROM pg_class c
#JOIN pg_namespace n ON n.oid = c.relnamespace
#WHERE n.nspname = 'your_schema' AND c.relkind = 'r';
        q="""
SELECT
    '--- TABLE: ' || c.relname AS ddl_header,
    'CREATE TABLE ' || n.nspname || '.' || c.relname || E' (\n' ||
    string_agg(
        '    ' || a.attname || ' ' || pg_catalog.format_type(a.atttypid, a.atttypmod),
        E',\n'
        ORDER BY a.attnum
    ) || E'\n);' AS ddl
FROM pg_class c
JOIN pg_namespace n ON n.oid = c.relnamespace
JOIN pg_attribute a ON a.attrelid = c.oid
WHERE n.nspname = :schema
  AND c.relkind = 'r'
  AND a.attnum > 0 AND NOT a.attisdropped
GROUP BY c.oid, c.relname, n.nspname
ORDER BY c.relname;
        """
        r=self.query(q, {'schema': self.schema})
        return r.fetchall()

    def DDL_views(self):
        q="""
SELECT '--- VIEW: ' || c.relname AS ddl_header,
       'CREATE VIEW ' || n.nspname || '.' || c.relname || ' AS ' || pg_get_viewdef(c.oid, true) || ';'
FROM pg_class c
JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE n.nspname = :schema AND c.relkind = 'v';
        """
        r=self.query(q, {'schema': self.schema})
        return r.fetchall()

    def DDL_types(self):
        q="""
SELECT '--- TYPE: ' || t.typname AS ddl_header,
       'CREATE TYPE ' || n.nspname || '.' || t.typname || ' AS (' ||
       string_agg(a.attname || ' ' || pg_catalog.format_type(a.atttypid, a.atttypmod), ', ') || ');'
FROM pg_type t
JOIN pg_namespace n ON n.oid = t.typnamespace
JOIN pg_class c ON c.oid = t.typrelid
JOIN pg_attribute a ON a.attrelid = c.oid
WHERE n.nspname = :schema
  AND t.typtype = 'c'
  AND a.attnum > 0 AND NOT a.attisdropped
GROUP BY t.typname, n.nspname
UNION
SELECT '--- ENUM TYPE: ' || t.typname AS ddl_header,
       'CREATE TYPE ' || n.nspname || '.' || t.typname || ' AS ENUM (' ||
       string_agg(quote_literal(e.enumlabel), ', ') || ');'
FROM pg_type t
JOIN pg_enum e ON e.enumtypid = t.oid
JOIN pg_namespace n ON n.oid = t.typnamespace
WHERE n.nspname = :schema
GROUP BY t.typname, n.nspname;
        """
        r=self.query(q, {'schema': self.schema})
        return r.fetchall()

    def DDL_functions(self):
        q="""
SELECT
    '--- FUNCTION: ' || p.proname || '(' || pg_get_function_arguments(p.oid) || ')' AS ddl_header,
    pg_get_functiondef(p.oid) AS ddl
FROM pg_proc p
JOIN pg_namespace n ON n.oid = p.pronamespace
WHERE n.nspname = :schema
  AND p.prokind = 'f'  -- 'f' = function, 'a' = aggregate, 'p' = procedure, 'w' = window
UNION
SELECT
    '--- AGGREGATE: ' || p.proname || '(' || pg_get_function_arguments(p.oid) || ')' AS ddl_header,
    'CREATE AGGREGATE ' || n.nspname || '.' || p.proname || '(' ||
        pg_get_function_arguments(p.oid) || ') ' ||
        '(SFUNC = ' || sfunc.proname || ', STYPE = ' || format_type(aggtranstype, NULL) || ')' AS ddl
FROM pg_proc p
JOIN pg_namespace n ON n.oid = p.pronamespace
JOIN pg_aggregate a ON a.aggfnoid = p.oid
JOIN pg_proc sfunc ON sfunc.oid = a.aggtransfn
WHERE n.nspname = :schema;
        """
        r=self.query(q, {'schema': self.schema})
        return r.fetchall()



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
    query_parser.add_argument("-p", "--pandas", action = "store_true",
                     help = "return pandas", default=False)
    subparsers.add_parser("tables", help="get tables and descriptions")
    subparsers.add_parser("columns", help="get columns and descriptions")
    ddl=subparsers.add_parser("ddl", help="generate ddl code")
    ddl.add_argument("-o", "--object", required=True, help="What database object to generate DDL",
                     choices=['table', 'view', 'type', 'function'])
    args = parser.parse_args()


    db = DBQuery(hostname=args.server, port=args.port, database=args.database, schema=args.schema, db_user=args.user, db_password=args.password)

    if args.command == "query":
        r= db.query_to_df(sql=args.sql) if args.pandas else db.query(sql=args.sql).all()
        print(r)
    elif args.command == "tables":
        print(db.describe_tables())
    elif args.command == "columns":
        print(db.describe_columns())
    elif args.command == "ddl":
        if args.object=="table":
            print(db.as_str(db.DDL_tables()))
        elif args.object=="view":
            print(db.as_str(db.DDL_views()))
        elif args.object=="type":
            print(db.as_str(db.DDL_types()))
        elif args.object=="function":
            print(db.as_str(db.DDL_functions()))
