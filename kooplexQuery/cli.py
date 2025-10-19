import logging
import time
import asyncio

logging.basicConfig(level=logging.INFO)
#logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def init_motor():
    from kooplexQuery.motor import Motor
    logger.info(f"Initialize motor")
    return Motor(table_name_filter='viralprimer_%')

async def ask_llm(motor, model, question, parse, run_sql):
    resp=""
    kw = {'model_name': model} if model else {}
    logger.info(f"Request stream")
    async for c in motor.chat(question, **kw):
        print(c, end="", flush=True)
        resp += c
    print(flush=True)
    logger.info(f"End of stream")
    if parse or run_sql:
        try:
            resp = resp.split("```sql")[-1].split('```')[0]
            logger.info("Successfully parsed")
        except:
            logger.error('Could not find sql markdown')
    with open(args.output, 'w') as f:
        f.write(resp)
        logger.info(f"Wrote file {args.output}")
    if run_sql:
        result=motor.db_source.query(resp).all()
        print(result)

if __name__ == '__main__':
    import os
    import argparse
    parser = argparse.ArgumentParser(description="Command line helper")
    parser.add_argument("-E", "--loadenv", action = "store",
                    help="configuration details", default = 'config.env')
    parser.add_argument("-u", "--username", action = "store",
                    help="username", default = 'fakeuser')
    parser.add_argument("-c", "--email", action = "store",
                    help="email address", default = 'fake@em.ail')
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available commands")
    p_list = subparsers.add_parser("list", help="List models")
    p_list.add_argument("-w", "--what", action = "store",
                     choices=['models', 'example'], help="list what", required=True)
    p_query = subparsers.add_parser("query", help="Ask oneshot question")
    p_query.add_argument("-q", "--question", action = "store",
                     help="ask it from LLM", required=True)
    p_query.add_argument("-m", "--model", action = "store",
                     help="which model to use", default=None)
    p_query.add_argument("-p", "--parse", action = "store_true",
                     help="look only for sql part")
    p_query.add_argument("-r", "--run_sql", action = "store_true",
                     help="run sql after response from llm")
    p_query.add_argument("-o", "--output", action = "store",
                     help="fale to save the response", default="/dev/null")
    args = parser.parse_args()

    if args.command=="list":
        if args.what=="models":
            from motor import supported_models
            for m in supported_models:
                print(m.name)
        elif args.what=="example":
            motor = init_motor()
            _q, _sql = motor.fetch_examples(1)[0]
            print (_q, '\n---->\n', _sql)
    elif args.command=="query":
        motor = init_motor()
        motor.new_session(username=args.username, email=args.email)
        asyncio.run(ask_llm(motor, args.model, args.question, args.parse, args.run_sql))
