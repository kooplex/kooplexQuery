from sqlalchemy import create_engine, text

class DBChat(object):
    def __init__(self, hostname, port, database, schema, db_user, db_password, generated_callback=lambda c: None):
        connectionstring = f"postgresql+psycopg2://{db_user}:{db_password}@{hostname}:{port}/{database}"
        self.engine = create_engine(connectionstring, connect_args={"options": f"-c search_path={schema}"})
        self.cb_generated = generated_callback


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


    chatter = DBChat(hostname=args.server, port=args.port, database=args.database, schema=args.schema, db_user=args.user, db_password=args.password)

    if args.command == "session":
        print(f"Starting session for user {args.chatuser} with label {args.label}")
        session_id = chatter.new_session(username=args.chatuser, email=args.chatemail, label=args.label, meta=args.meta)
        print(session_id)
    elif args.command == "conversation":
        print(f"(in session {args.session_id}) [user says: {args.user_content} and agent replies {args.agent_content}")
        chatter.save_chat_item(session_id=args.session_id, user_prompt=args.user_content, agent_response=args.agent_content)
    elif args.command == "finalize":
        print(f"Finalizing session {args.session_id} {args.question} |-> {args.sql}")
        chatter.save_query(session_id=args.session_id, question_content=args.question, sql=args.sql)
    elif args.command == "example":
        print(chatter.fetch_examples(args.limit))
    elif args.command == "knowledge":
        print(chatter.load_knowledge(args.reference))
