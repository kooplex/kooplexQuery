from sqlalchemy import create_engine, text

class DBChat(object):
    @staticmethod
    def create_database_if_missing(hostname, port, database):
        from dotenv import load_dotenv
        import os

        load_dotenv("config.env")

        schema_manager = os.getenv('CHAT_SCHEMA_MANAGER', 'schema_manager')
        schema_manager_password = os.getenv('CHAT_SCHEMA_MANAGER_PASSWORD', 'schema_manager_password')

        if (
            not database
            or not database.replace("_", "").isalnum()
            or not database[0].isalpha()
        ):
            raise ValueError(f"Invalid chat database name: {database}")

        connectionstring = (
            f"postgresql+psycopg2://{schema_manager}:{schema_manager_password}"
            f"@{hostname}:{port}/postgres"
        )
        engine = create_engine(connectionstring, isolation_level="AUTOCOMMIT")

        try:
            with engine.connect() as con:
                exists = con.execute(
                    text("SELECT 1 FROM pg_database WHERE datname = :dbname"),
                    {'dbname': database},
                ).scalar() is not None

                if exists:
                    return False

                con.execute(text(f"CREATE DATABASE {database}"))
                return True
        finally:
            engine.dispose()

    def __init__(self, hostname, port, database, schema, user, password, generated_callback=lambda c: None):
        connectionstring = f"postgresql+psycopg2://{user}:{password}@{hostname}:{port}/{database}"
        
        self.cb_generated = generated_callback
        self.schema = schema
        self.port = port
        self.hostname = hostname
        self.database = database
        self.user = user
        self.password = password

        if not self._check_schema():
            print(f"Schema {schema} does not exist. Creating schema and tables...")
            self.create_schema()
            self.engine = create_engine(connectionstring, connect_args={"options": f"-c search_path={schema}"})
        else:
            print(f"Schema {schema} exists. Ready to use.")

        self.engine = create_engine(connectionstring, connect_args={"options": f"-c search_path={schema}"})
        

    def _check_schema(self):
        import os
        _ge = lambda x,d: os.getenv(x, d)
        schema_manager = _ge('CHAT_SCHEMA_MANAGER', 'schema_manager')
        schema_manager_password = _ge('CHAT_SCHEMA_MANAGER_PASSWORD', 'schema_manager_password')
        
        connectionstring = f"postgresql+psycopg2://{schema_manager}:{schema_manager_password}@{self.hostname}:{self.port}/{self.database}"
        engine = create_engine(connectionstring)
        q = text("""
SELECT schema_name
FROM information_schema.schemata
WHERE schema_name = :schema;
        """)
        try:
            with engine.connect() as con:
                r = con.execute(q, {'schema': self.schema}).scalar()
                return r is not None
        except Exception as e:
            print(f"Error checking schema: {e}\n If chat database is alredy initialized, and read-only, than ignore this!")
            return False

    def _get_userid(self, username, email):
        with self.engine.connect() as con:
            rec = {"username": username, "email": email}
            # try lookup userid
            q = text("""
SELECT id 
FROM "user" 
WHERE username=:username AND email=:email;
            """)
            r = con.execute(q, rec).scalar()
            if r is None:
                # store new user
                qc = text("""
INSERT INTO "user" (username, email)
VALUES (:username, :email)
RETURNING id;
                """)
                r = con.execute(qc, rec).scalar()
            con.commit()
            return r

    def _get_metaid(self, meta):
        with self.engine.connect() as con:
            rec = {"content": meta}
            # try lookup metaid
            q = text("""
SELECT id 
FROM meta
WHERE content=:content;
            """)
            r = con.execute(q, rec).scalar()
            if r is None:
                # store new meta
                qc = text("""
INSERT INTO meta (content)
VALUES (:content)
RETURNING id;
                """)
                r = con.execute(qc, rec).scalar()
            con.commit()
            return r

    def new_session(self, username, email, label, meta, referenced_session=None):
        with self.engine.connect() as con:
            rec = {
                "label": label, 
                "meta_id": self._get_metaid(meta), 
                "user_id": self._get_userid(username, email),
                "ref": referenced_session
            }
            # store new session
            qc = text("""
INSERT INTO "session" (label, meta_id, user_id, referenced_session_id)
VALUES (:label, :meta_id, :user_id, :ref)
RETURNING id;
            """)
            r = con.execute(qc, rec).scalar()
            con.commit()
            return r
            
    def load_knowledge(self, reference):
        q = text("""
SELECT content
FROM knowledge
WHERE reference=:reference
        """)
        with self.engine.connect() as con:
            r=con.execute(q, {'reference': reference})
            return r.scalar()

    def fetch_all_knowledge(self):
        q = text("""
SELECT id, reference, content
FROM knowledge
ORDER BY reference, id
        """)
        with self.engine.connect() as con:
            result = con.execute(q)
            keys = result.keys()
            data = result.fetchall()
            return keys, data

    # FIXME update instead of insert with conflict handling
    def save_knowledge(self, reference, content):
#         q = text("""
# UPDATE knowledge
# SET content=:content
# WHERE reference=:reference
#         """)
        q = text("""
INSERT INTO knowledge (reference, content)
VALUES (:reference, :content)
ON CONFLICT (reference) 
DO UPDATE SET 
    content = EXCLUDED.content
        """)
        #ON CONFLICT (reference) DO UPDATE SET content = EXCLUDED.content
        with self.engine.connect() as con:
            r = con.execute(q, {'reference': reference, 'content': content})
            con.commit()
            return r  # returns True if an existing record was updated, False if no record with the reference exists


    def save_chat_item(self, session_id, user_prompt, agent_response, model_name): #TODO: save model_name in DB
        rec = {
            'session_id': session_id,
            'content_user': user_prompt,
            'content_agent': agent_response,
        }
        qc = text("""
INSERT INTO chathistory (session_id, role, content)
VALUES (:session_id, 'user', :content_user);
INSERT INTO chathistory (session_id, role, content)
VALUES (:session_id, 'agent', :content_agent);
        """)
        with self.engine.connect() as con:
            con.execute(qc, rec)
            con.commit()


    def save_query(self, session_id, question_content, sql, question_type = 'user', public = True):
        assert question_type in ['user', 'train', 'followup'], "Wrong question type. Choose from 'user', 'followup' or 'train'"
        with self.engine.connect() as con:
            rec = {
                "session_id": session_id,
                "type": question_type,
                "content": question_content, 
                "generated": self.cb_generated(question_content),
                "public": public,
            }
            # store new session
            qc = text("""
INSERT INTO question (type, content, generated, public, session_id)
VALUES (:type, :content, :generated, :public, :session_id)
RETURNING id;
            """)
            question_id = con.execute(qc, rec).scalar()
            qc2 = text("""
INSERT INTO query (sql, question_id)
VALUES (:sql, :qid)
            """)
            con.execute(qc2, {'sql': sql, 'qid': question_id})
            con.commit()

    def fetch_examples(self, limit=3):
        q = text("""
select q.content as "question", a.sql
from question q
join query a
on q.id=a.question_id
where q.type='train' and q.public
ORDER BY RANDOM()
LIMIT :limit
        """)
        with self.engine.connect() as con:
            r = con.execute(q, {'limit': limit}).fetchall()
            return r

    def fetch_all_examples(self):
        q = text("""
select q.id as "question_id", q.type, q.content as "question_content", q.generated, q.public, q.session_id, a.id as "query_id", a.sql, a.score
from question q
join query a
on q.id=a.question_id;
        """)
        with self.engine.connect() as con:
            result = con.execute(q)
            keys = result.keys()  # <-- This gives you the column names
            data = result.fetchall()
            return keys, data
    
    def delete_row(self, question_id):
        question_id = int(question_id)
        with self.engine.begin() as con:
            # Step 1: Delete from child table first (chat.query)
            ddl = f"""
DELETE FROM {self.schema}.query
WHERE question_id = :question_id
            """
            con.execute(text(ddl), {
                'question_id': question_id
            })
        
            # Step 2: Delete from parent table (chat.question)
            ddl = f"""
DELETE FROM {self.schema}.question
WHERE id = :question_id  """
            con.execute(text(ddl), {
                'question_id': question_id
            })
        return True

    def validate_question(self, question_id):
        """ 
        After an expert overviewed the question and the corresponding results
        it adds this row to the training set, makes it public and adds 1 to the overall score
        A next validation eill further increase it's score
        """
        question_id = int(question_id)
        with self.engine.begin() as con:
            ddl = f"""
UPDATE {self.schema}.query
SET 
    score = :score
WHERE question_id = :question_id
                """
            con.execute(text(ddl), {
                    'score': 1,
                    'question_id': question_id
                })
            ddl = f"""
UPDATE {self.schema}.question
SET type = :type,
    public = :public
WHERE id = :question_id
                """
            con.execute(text(ddl), {
                    'type': 'train',
                    'public': True,
                    'question_id': question_id
                })
        return True

    # Functions for creating and managing the database schema
    def create_schema(self):
        from dotenv import load_dotenv
        load_dotenv("config.env")
        # Connect with schema manager role to create the schema
        import os
        _ge = lambda x,d: os.getenv(x, d)
        schema_manager = _ge('CHAT_SCHEMA_MANAGER', 'schema_manager')
        schema_manager_password = _ge('CHAT_SCHEMA_MANAGER_PASSWORD', 'schema_manager_password')
        
        connectionstring = f"postgresql+psycopg2://{schema_manager}:{schema_manager_password}@{self.hostname}:{self.port}/{self.database}"
        engine = create_engine(connectionstring)


        ddl_reader = f"""
CREATE ROLE {self.user} WITH LOGIN PASSWORD :password;
"""
        ddl_schema = f"""
CREATE SCHEMA IF NOT EXISTS {self.schema} AUTHORIZATION {schema_manager};
GRANT ALL PRIVILEGES ON SCHEMA {self.schema} TO {self.user};
GRANT USAGE ON ALL SEQUENCES IN SCHEMA {self.schema} TO {self.user};
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA {self.schema} TO {self.user};
ALTER DEFAULT PRIVILEGES IN SCHEMA {self.schema}
  GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO {self.user};
ALTER DEFAULT PRIVILEGES IN SCHEMA {self.schema}
  GRANT USAGE ON SEQUENCES TO {self.user};
"""

        ddl_tables = f"""
-- create tables and sequences in the specified schema
SET search_path = {self.schema}, public;
CREATE TYPE {self.schema}.type_role AS ENUM ('agent','user');
CREATE TYPE {self.schema}.type_question AS ENUM ('train','user','followup');

CREATE SEQUENCE IF NOT EXISTS user_id_seq;
CREATE TABLE {self.schema}.user (
    id integer DEFAULT nextval('{self.schema}.user_id_seq'::regclass) NOT NULL,
    username character varying,
    email character varying,
    PRIMARY KEY (id)
);

CREATE SEQUENCE IF NOT EXISTS knowledge_id_seq;
CREATE TABLE {self.schema}.knowledge (
    id integer DEFAULT nextval('{self.schema}.knowledge_id_seq'::regclass) NOT NULL,
    reference character varying(64) UNIQUE,
    content text,
    PRIMARY KEY (id)
);

CREATE SEQUENCE IF NOT EXISTS meta_id_seq;
CREATE TABLE {self.schema}.meta (
    id integer DEFAULT nextval('{self.schema}.meta_id_seq'::regclass) NOT NULL,
    content text,
    PRIMARY KEY (id)
);

CREATE SEQUENCE IF NOT EXISTS session_id_seq;
CREATE TABLE {self.schema}.session (
    id integer DEFAULT nextval('{self.schema}.session_id_seq'::regclass) NOT NULL,
    timestamp timestamp without time zone DEFAULT now(),
    label character varying NOT NULL,
    meta_id integer NOT NULL,
    user_id integer NOT NULL,
    referenced_session_id integer,
    PRIMARY KEY (id),
    FOREIGN KEY (meta_id) REFERENCES {self.schema}.meta(id) ON UPDATE NO ACTION ON DELETE NO ACTION,
    FOREIGN KEY (referenced_session_id) REFERENCES {self.schema}.session(id) ON UPDATE NO ACTION ON DELETE NO ACTION,
    FOREIGN KEY (user_id) REFERENCES {self.schema}.user(id) ON UPDATE NO ACTION ON DELETE NO ACTION
);


CREATE SEQUENCE IF NOT EXISTS chathistory_id_seq;
CREATE SEQUENCE IF NOT EXISTS chathistory_sequence_seq;
CREATE TABLE {self.schema}.chathistory (
    id integer DEFAULT nextval('{self.schema}.chathistory_id_seq'::regclass) NOT NULL,
    session_id integer NOT NULL,
    sequence integer NOT NULL DEFAULT nextval('{self.schema}.chathistory_sequence_seq'::regclass),
    role {self.schema}.type_role NOT NULL,
    timestamp timestamp without time zone DEFAULT now(),
    content text NOT NULL,
    PRIMARY KEY (id),
    FOREIGN KEY (session_id) REFERENCES {self.schema}.session(id) ON UPDATE NO ACTION ON DELETE NO ACTION
);

CREATE SEQUENCE IF NOT EXISTS question_id_seq;
CREATE TABLE {self.schema}.question (
    id integer DEFAULT nextval('{self.schema}.question_id_seq'::regclass) NOT NULL,
    type {self.schema}.type_question DEFAULT 'user'::{self.schema}.type_question,
    content text NOT NULL,
    generated text,
    public boolean,
    session_id integer,
    PRIMARY KEY (id),
    FOREIGN KEY (session_id) REFERENCES {self.schema}.session(id) ON UPDATE NO ACTION ON DELETE NO ACTION
);

CREATE SEQUENCE IF NOT EXISTS equivalence_id_seq;
CREATE TABLE {self.schema}.equivalence (
    id integer DEFAULT nextval('{self.schema}.equivalence_id_seq'::regclass) NOT NULL,
    question1_id integer NOT NULL,
    question2_id integer NOT NULL,
    count_acceptance integer DEFAULT 0,
    count_rejection integer DEFAULT 0,
    PRIMARY KEY (id),
    FOREIGN KEY (question1_id) REFERENCES {self.schema}.question(id) ON UPDATE NO ACTION ON DELETE NO ACTION,
    FOREIGN KEY (question2_id) REFERENCES {self.schema}.question(id) ON UPDATE NO ACTION ON DELETE NO ACTION
);

CREATE SEQUENCE IF NOT EXISTS query_id_seq;
CREATE TABLE {self.schema}.query (
    id integer DEFAULT nextval('{self.schema}.query_id_seq'::regclass) NOT NULL,
    sql text NOT NULL,
    question_id integer NOT NULL,
    score integer,
    PRIMARY KEY (id),
    FOREIGN KEY (question_id) REFERENCES {self.schema}.question(id) ON UPDATE NO ACTION ON DELETE NO ACTION
);
"""
        
        try: 
            with engine.connect() as con:
                try:
                    con.execute(text(ddl_reader), {'password': self.password})
                except Exception as e:
                    print(f"Error creating reader role (might already exist): {e}")
                con.commit()
                con.execute(text(ddl_schema))
                con.commit()
                con.execute(text(ddl_tables))
                con.commit()
        except:
            print("Error connecting to DB chat as schema manager!\n If chat database is alredy initialized, and read-only, than ignore this!")


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
    # session command
    session_parser = subparsers.add_parser("session", help="Start a new session")
    session_parser.add_argument("-U", "--chatuser", action = "store",
                     help = "mock username", required=True)
    session_parser.add_argument("-E", "--chatemail", action = "store",
                     help = "mock user's email", required=True)
    session_parser.add_argument("-L", "--label", action = "store",
                     help = "session label", required=True)
    session_parser.add_argument("-M", "--meta", action = "store",
                     help = "session meta info repr", required=True)
    # conversation command
    conversation_parser = subparsers.add_parser("conversation", help="Add a message to a session")
    conversation_parser.add_argument("-I", "--session_id", action = "store",
                     help = "session_id", required=True)
    conversation_parser.add_argument("-Q", "--user_content", action = "store",
                     help = "user's prompt", required=True)
    conversation_parser.add_argument("-R", "--agent_content", action = "store",
                     help = "agent's response", required=True)
    # finalize command
    finalize_parser = subparsers.add_parser("finalize", help="Finalize a session")
    finalize_parser.add_argument("-I", "--session_id", action = "store",
                     help = "session_id", required=True)
    finalize_parser.add_argument("-F", "--question", action = "store",
                     help = "the final question", required=True)
    finalize_parser.add_argument("-S", "--sql", action = "store",
                     help = "the sql equivalent", required=True)
    # example command
    example_parser = subparsers.add_parser("example", help="Fetch example queries")
    example_parser.add_argument("-n", "--limit", action = "store",
                    help = "number of records", default=3)
    # knowledge retrieval command
    knowledge_parser = subparsers.add_parser("knowledge", help="Fetch extra knowledge by reference")
    knowledge_parser.add_argument("-r", "--reference", action = "store",
                    help = "reference", required=True)
    args = parser.parse_args()


    chatter = DBChat(hostname=args.server, port=args.port, database=args.database, schema=args.schema, user=args.user, password=args.password)

    if args.command == "session":
        print(f"Starting session for user {args.chatuser} with label {args.label}")
        session_id = chatter.new_session(username=args.chatuser, email=args.chatemail, label=args.label, meta=args.meta)
        print(session_id)
    elif args.command == "conversation":
        print(f"(in session {args.session_id}) [user says: {args.user_content} and agent replies {args.agent_content}")
        chatter.save_chat_item(session_id=args.session_id, user_prompt=args.user_content, agent_response=args.agent_content, model_name="cli")
    elif args.command == "finalize":
        print(f"Finalizing session {args.session_id} {args.question} |-> {args.sql}")
        chatter.save_query(session_id=args.session_id, question_content=args.question, sql=args.sql)
    elif args.command == "example":
        print(chatter.fetch_examples(args.limit))
    elif args.command == "knowledge":
        print(chatter.load_knowledge(args.reference))
