New part

## Setup database connections


# kooplexQuery
Text2SQL solutions for any SQL database, built with Streamlit and Langchain.

It consists of three main components:
* The chat app built with Streamlit, which allows users to interact with the system and ask questions in natural language.
* The Metadata database manager (manage_db.py), which handles the creation and management of the metadata database that stores information about the tables and columns in the SQL database.
* The sqlValidator, which helps to validate the generated SQL queries against the metadata database to ensure they are correct and can be executed on the SQL database. Only validated queries will be used as additional context for the LLM to generate the final SQL query.

## Suppoerted SQL databases:
* PostgreSQL
* Microsoft SQL Server

## Getting Started
To get started with kooplexQuery, follow these steps:
1. Clone the repository and navigate to the project directory.
2. Install the required dependencies using pip:
3. Run the three applications in the following order:
   * First, run the Metadata database manager to create and populate the metadata database:
     ```
     python kooplexQuery_utils/manage_db.py
     ```
   * Next, run the sqlValidator to start the validation service:
     ```
     python kooplexQuery_utils/sql_validator.py
     ```
   * Finally, run the Streamlit chat app to start interacting with the system:
     ```
     streamlit run kooplexQuery/prompt.py
     ```
4. Upload all necessary metadata with the manage_db, which will be used as context for the LLM to generate SQL queries. This includes:
   1. Database schema
   2. Prompt instructions for the LLM
   3. Data descripto(s)
   4. (optionally) Table, column descriptions
   5. (Optionally) Furthe documents
5. Start asking questions in natural language through the Streamlit chat app, and the system will generate and execute SQL queries to retrieve the relevant data from the SQL database.
  --> save successfull queries to the metadata database, which will be used as additional context for the LLM to generate more accurate SQL queries in the future.
7. Validate the generated SQL queries using the sqlValidator to ensure they are correct and can be executed on the SQL database.   

## Requirements
* Python 3.12 or higher
* PostgreSQL database to store the metadata

## Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| DB_TITLE | Title of the main database | "Main Database" |
| DB_HOST | Hostname or IP address of the main database server | "sqlserver.example.com" |
| DB_PORT | Port number for the main database connection | 1433 |
| DB_USER | Username for the main database connection | "db_user" |
| DB_PASSWORD | Password for the main database connection | "password" |
| DB_DATABASE | Name of the main database | "my_database" |
| DB_TYPE | Type of the main database (postgresql or mssql) | "mssql" |
| CHAT_HOST | Hostname or IP address of the chat metadata database server | "metadata.example.com" |
| CHAT_PORT | Port number for the chat metadata database connection | 5432 |
| CHAT_USER | Username for the chat metadata database connection | "chat_reader" |
| CHAT_PASSWORD | Password for the chat metadata database connection | "chat_reader" |
| CHAT_DATABASE | Name of the chat metadata database | "chat" |
| CHAT_SCHEMA | Schema name in the chat metadata database | "chat" |
| VECSTORE_PATH | Path to the vector store database file | "vecstore.db" |
| OLLAMA_HOST | Hostname for the Ollama service (leave empty if not used) | "" |
| OPENAI_API_KEY | API key for OpenAI services (leave empty if not used) | "" |
| OPENAI_ORG | Organization ID for OpenAI services (leave empty if not used) | "" |
| CHAT_SCHEMA_MANAGER | Username for schema management in the chat database | "schema_manager" |
| CHAT_SCHEMA_MANAGER_PASSWORD | Password for schema management in the chat database | "schema_manager_password" |

