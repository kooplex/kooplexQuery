from sqlalchemy import create_engine, text
import os
import pandas as pd
from dotenv import load_dotenv
load_dotenv("config.env")
_ge = lambda x,d: os.getenv(x, d)
hostname=_ge('DB_HOST', 'localhost')
port=int(_ge('DB_PORT', 5432))
database=_ge('DB_DATABASE', 'sewage')
schema=_ge('DB_SCHEMA', 'chat')
db_user=_ge('DB_USER', 'chat_agent')
db_password=_ge('DB_PASSWORD', '')
                                        
# For postgres, the connection string would look like this:
connectionstring = f"postgresql+psycopg2://{db_user}:{db_password}@{hostname}:{port}/{database}"
# For mssql, the connection string would look like this:
connectionstring = f"mssql+pymssql://{db_user}:{db_password}@{hostname}:{port}/{database}"
                                        
engine = create_engine(connectionstring)    