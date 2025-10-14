import streamlit as st
from streamlit_extras.chart_container import *
import streamlit.components.v1 as components
from streamlit_elements import html
import logging
import time
from motor import Motor, supported_models
import asyncio
import uuid

logger = logging.getLogger(__name__)

# Set some session state defaults
st.session_state.setdefault('sql_submit', None)
st.session_state.setdefault('sql_fix', None)
st.session_state.setdefault('sql_saved', None)
st.session_state.setdefault('plot_instruction', None)
st.session_state.setdefault('save_req', False)
st.session_state.setdefault('rerun', False)

st.session_state.setdefault('df', None)


###
username="fakeuser"
email="fake@em.ail"
#####
# Inject CSS globally
st.html("""
<style>
div[data-testid="stDialog"] div[role="dialog"]:has(.big-dialog) {
    width: 80vw;
    height: None;
}
</style>
""")

# Main containers
example_container = st.empty()
histo_container = st.empty()
control_container = st.empty()

# Pop up dialogs
@st.dialog("DB browser")
def show_schema_browser():
    import pandas
    st.html("<span class='big-dialog'></span>")
    table_description=dict(st.session_state.motor.describe_tables())
    column_description=pandas.DataFrame(st.session_state.motor.describe_columns(), columns=['table', 'column', 'description', 'type'])
    selected_tables = st.multiselect("Select Tables", table_description.keys())
    if selected_tables:
        for t in selected_tables:
            with st.expander(f"Table {t}"):
                st.write(table_description[t])
            with st.expander(f"Column Descriptions for {t}"):
                st.write(column_description[column_description['table']==t].drop(columns=['table']))
            with st.expander(f"Example records from {t}"):
                sql=f"SELECT * FROM {t} ORDER BY RANDOM() LIMIT 2"
                st.write(st.session_state.motor.db_source.query_to_df(sql))


# Form where user may modify the SQL query
def form_sql(sql, fix_error=None, last_record=False):
    uid=uuid.uuid4()
    with st.form(f'f-{uid}'):
        st.text_area(
            "You may rewrite the query proposal or run it directly:", 
            value=sql.strip(),
            key=f"ns-{uid}",
            help="If you are familiar with SQL syntax you may modify and fine tune the query and rerun")
        def _t():
            st.session_state.sql_submit=getattr(st.session_state, f'ns-{uid}')
        st.form_submit_button("🛢️ Run the query", on_click=_t)
        if fix_error:
            if last_record and st.session_state.autocorrect and st.session_state.sql_fix is None:
                st.session_state.sql_fix=(sql, fix_error)
                st.rerun()
            def _t():
                st.session_state.sql_fix=(sql, fix_error)
            st.form_submit_button("🔨 Let LLM fix", on_click=_t)


# Form where user may modify plot instructions
def form_plot(plot_instructions="Plot data"):
    uid=uuid.uuid4()
    with st.form(f'f-{uid}'):
        st.text_area(
            "Enter your instructions for plotting the data",
            value=plot_instructions.strip(),
            key=f"pi-{uid}",
            help="Provide instructions on how you want the data to be plotted. For example, 'Plot the histogram of temperature and pH'."
        )
        def _t():
            st.session_state.plot_instruction=getattr(st.session_state, f'pi-{uid}')
        st.form_submit_button("📊 Generate Plot", on_click=_t)


# Parse and write LLM response
def render_sql_parsed(chunks):
    for chunk in chunks:
        if chunk.type=='txt':
            st.write(chunk.content)
        elif chunk.type=='sql':
            sql=chunk.content
            tab1, tab2 = st.tabs(["🛢️ Sql", "🦖 Rewrite code"])
            with tab1:
                with st.form(str(uuid.uuid4())):
                    def _q(q):
                        st.session_state.sql_submit=q
                    st.code(sql, language="sql", wrap_lines=True)
                    st.form_submit_button(on_click=_q, args=(sql,))
            with tab2:
                form_sql(sql)
        else:
            st.error("Unhandled chunk")
            st.write(chunk)


# Print dataset
def render_df(sql, df):
    tab1, tab2, tab3 = st.tabs(["📋 Dataset", "🚀 Plot", "🦖 Rewrite SQL"])
    with tab1:
        st.dataframe(df)
    with tab2:
        form_plot()
    with tab3:
        form_sql(sql)

# Show plot
def render_plot(instructions, answer, pyplot):
    fig=answer['content']
    tab1, tab2, tab3 = st.tabs(["📈 Plot", "🧠 Code", "🚀 Replot"])
    with tab1:
        if pyplot:
            st.pyplot(fig)
        else:
            st.plotly_chart(fig, key=str(uuid.uuid4()))
    with tab2:
        st.code(answer["code"])
    with tab3:
        form_plot(instructions)

# Present error message
def render_error(answer, last_record):
    _tabs=["💥 Error"]
    error=answer["content"]
    if "code" in answer:
        _tabs.append("🧠 Code")
    if "query" in answer:
        _tabs.append("🛠️ Fix SQL code")
    tabs=st.tabs(_tabs)
    with tabs[0]:
        e=getattr(error, 'orig', error)
        st.error(e)
    if "code" in answer:
        with tabs[1]:
            st.code(answer["code"])
    if "query" in answer:
        with tabs[-1]:
            form_sql(answer['query'], fix_error=error, last_record=last_record)

# Display chat history
def history():
    with histo_container.container():
        for r in st.session_state.motor.chat_history:
            prompt=r['question']
            response=r['answer']
            response_meta=r['answer_meta']
            response_type=response_meta.get('type')
            last_pair=r['is_last']
            label = f"{prompt[:80]}{'...' if len(prompt) > 80 else ''}"
            with st.expander(label=label, expanded=last_pair):
                with st.chat_message("user"):
                    if r['question_meta'].get('type')=='submit_sql':
                        st.code(prompt, language="sql", wrap_lines=True)
                    else:
                        st.write(prompt)
                with st.chat_message("assistant"):
                    if response_type=="dataframe":
                        render_df(prompt, response_meta['dataframe'])
                    elif response_type in ["pyplot", "plotly_chart"]:
                        render_plot(prompt, response_meta, response_type=="pyplot")
                    elif response_type=="error":
                        render_error(response_meta, last_pair)
                    else:
                        render_sql_parsed(response_meta.get('parsed'))
                if dt:=response_meta.get('duration'):
                    st.info(f"duration: {dt} s")
                if last_pair:
                    if st.button('Delete'):
                        st.session_state.motor.pop()
                        st.rerun()
#        if len(chat_history)%1 == 1:
#            user_msg=chat_history[-1]
#            st.chat_message("user").write(user_msg["content"])

    

# Initialize the backend motor
if st.session_state.get('motor') is None:
    logger.info(f"Initialize motor")
    try:
        st.session_state.motor = Motor(table_name_filter='viralprimer_%')
    except Exception as e:
        st.error(e)
        st.stop()


# Convenient function to clear parts of the session state
def clear_session_keys(*keys):
    for key in keys:
        if key in st.session_state:
            st.session_state[key] = None

# A helper to setup a new session
def _newsession():
    st.session_state['session']=st.session_state.motor.new_session(username=username, email=email) #TODO: label, referenced_session
    clear_session_keys('sql_submit', 'save_success') #FIXME
    st.rerun()

# Setup a new session on the first run
if 'session' not in st.session_state:
    #TODO ask for a session label
    #TODO elaborate meta and be it json dump
    logger.info(f"Creating new session")
    _newsession()

# Handle model selection change
def on_model_selection_change():
    st.session_state.current_model=st.session_state.selected_model.name


# Page render logic
with st.sidebar:
    st.title('Viralprimer GPT')
    if st.button("Schema browser", use_container_width=True):
        show_schema_browser()
    if st.button("New Session", use_container_width=True, disabled=st.session_state.motor.chat_history.is_empty):
        _newsession()
    st.toggle("Autocorrect", value=False, key="autocorrect", on_change=None, args=None, kwargs=None)
    selected_model=st.sidebar.selectbox(
        "Select model for Text2SQL",
        supported_models,
        key="selected_model",
        format_func=lambda x:f"{x.name} - {x.type}",
        on_change=on_model_selection_change,
    )
    st.session_state.current_model=selected_model.name
    if st.button("Save Accurate Query", use_container_width=True, disabled=not st.session_state.motor.can_prepare_save):
        st.session_state.save_req=True
        st.rerun()

# The page body
def show_examples(n_examples=3):
    if st.session_state.motor.chat_history.is_empty:
        with example_container.container():
            with st.form("example_form"):
                st.markdown("##### _Example questions_")
                cols = st.columns(n_examples)
                def _s(prompt, sql):
                    st.session_state.motor.select_example(prompt, sql)
                    st.session_state.rerun=True
        
                for i, (_prompt, _sql) in enumerate(st.session_state.motor.fetch_examples(n_examples)):
                    cols[i].form_submit_button(_prompt, type="secondary", on_click=_s, args=[_prompt, _sql])
    else:
        example_container.empty()

show_examples()
if st.session_state.rerun:
    st.session_state.rerun=False
    st.rerun()
history()

# Save requested
if st.session_state.save_req:
    st.session_state.save_req=False
    with control_container.container():
        st.markdown("💡 Note: _An expert will validate this relation later._")
        with st.spinner("Digesting conversation..."):
            st.write_stream(st.session_state.motor.prepare_save())
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("💾 Save", disabled=not st.session_state.motor.can_save):
                    st.session_state.sql_saved=st.session_state.motor.sql
                    st.session_state.motor.save_query()
                    st.rerun()
            with col2:
                if st.button("❌ Cancel"):
                    st.rerun()

# Show success
if st.session_state.sql_saved:
     st.session_state.sql_saved=None
     st.success("Saved query")


# Submit SQL
if st.session_state.sql_submit:
    with st.spinner("✅ Data retrieval..."):
        st.session_state.motor.execute(st.session_state.sql_submit)
        st.session_state.sql_submit=None
        st.rerun()

# Fix SQL
if st.session_state.sql_fix:
    with control_container.container():
        with st.spinner("⏳ Thinking...."):
            with st.chat_message("assistant"):
                st.write_stream(st.session_state.motor.correct_error(st.session_state.sql_fix[1]))
        st.session_state.sql_fix=None
        st.rerun()

# Generate plot
if st.session_state.plot_instruction:
    with st.spinner("✅ Generating plot"):
        with st.chat_message("assistant"):
            st.write_stream(st.session_state.motor.plot(st.session_state.plot_instruction, st.session_state.selected_model.name))
            st.session_state.plot_instruction=None
            st.rerun()


# Handle prompt input
if "awaiting_response" not in st.session_state:
    st.session_state.awaiting_response = False
if "latest_user_input" not in st.session_state:
    st.session_state.latest_user_input = None

# Show input only if not currently processing
if not st.session_state.awaiting_response:
    user_input = st.chat_input("Enter a prompt", key="user_input")
    if user_input:
        st.session_state.latest_user_input = user_input
        st.session_state.awaiting_response = True
        st.rerun()

# Show response if awaiting
if st.session_state.awaiting_response and st.session_state.latest_user_input:
    with st.chat_message("user"):
        st.write(st.session_state.latest_user_input)

    with st.spinner("⏳ Thinking..."):
        with st.chat_message("assistant"):
            st.write_stream(st.session_state.motor.chat(st.session_state.latest_user_input, st.session_state.selected_model.name))

    # Reset state and allow new input
    st.session_state.awaiting_response = False
    st.session_state.latest_user_input = None
    st.rerun()



# Uncomment to debug session_state
#st.write(st.session_state)

