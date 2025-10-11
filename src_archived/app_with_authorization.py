#From https://raw.githubusercontent.com/marshmellow77/streamlit-chatgpt-ui/

import openai
from openai import OpenAI


try: 
    with open("./openai-api.token") as f:
        token = f.read().strip()
except:
    with open("/v/wfct0p/API-tokens/openai-api.token") as f:
        token = f.read().strip()

client = OpenAI(api_key=token, organization="org-zYlK9QX4WxhHbWaBm36KHpHR")

import streamlit as st
from streamlit_chat import message
from streamlit_oauth import OAuth2Component
import os

# Set environment variables
# AUTHORIZE_URL = os.environ.get('AUTHORIZE_URL')
# TOKEN_URL = os.environ.get('TOKEN_URL')
# REFRESH_TOKEN_URL = os.environ.get('REFRESH_TOKEN_URL')
# REVOKE_TOKEN_URL = os.environ.get('REVOKE_TOKEN_URL')
# CLIENT_ID = os.environ.get('CLIENT_ID')
# CLIENT_SECRET = os.environ.get('CLIENT_SECRET')
# REDIRECT_URI = os.environ.get('REDIRECT_URI')
# SCOPE = os.environ.get('SCOPE')
AUTHORIZE_URL = "https://kooplex-auth.elte.hu/oauth/o/authorize/"
TOKEN_URL = "https://kooplex-auth.elte.hu/oauth/o/token/"
REFRESH_TOKEN_URL = ""
REVOKE_TOKEN_URL = "" 
CLIENT_ID = "cR93T0FI05vMNIaxSr2Eh4jY6FapTV0iDMeRjPjA"
CLIENT_SECRET = "uSGxqCzGsXypTNowknoArv6S5kKJIXOrMsLZTj3vO8mZaZxgCLkHDrZ35YlTQhKaZ6UTmAmqdWBzsBjn2C3Ltr9Q15f2nhuq1blTZHZWja7xakSpWjCqwhhczWY3SNI0"
#REDIRECT_URI = "https://k8plex-veo.vo.elte.hu/notebook/report/wfct0p-streamlitchatgpt/"
REDIRECT_URI = "https://k8plex-veo.vo.elte.hu/notebook/report/wfct0p-dd/"
SCOPE = "email openid"

st.set_page_config(page_title="Csetrobot", page_icon=":robot_face:")

# Create OAuth2Component instance
oauth2 = OAuth2Component(CLIENT_ID, CLIENT_SECRET, AUTHORIZE_URL, TOKEN_URL, REFRESH_TOKEN_URL, REVOKE_TOKEN_URL)

# Check if token exists in session state
if 'token' not in st.session_state:
    # If not, show authorize button
    result = oauth2.authorize_button("Authorize", REDIRECT_URI, SCOPE)
    if result and 'token' in result:
        # If authorization successful, save token in session state
        st.session_state.token = result.get('token')
        st.experimental_rerun()
else:
    # If token exists in session state, show the token
    token = st.session_state['token']
    st.json(token)
    if st.button("Refresh Token"):
        # If refresh token button is clicked, refresh the token
        token = oauth2.refresh_token(token)
        st.session_state.token = token
        st.experimental_rerun()

# Setting page title and header

st.markdown("<h1 style='text-align: center;'>Hello, mire vagy kíváncsi? 😬</h1>", unsafe_allow_html=True)

# Set org ID and API key
# TODO: The 'openai.organization' option isn't read in the client API. You will need to pass it when you instantiate the client, e.g. 'OpenAI(organization="org-zYlK9QX4WxhHbWaBm36KHpHR")'
# openai.organization = "org-zYlK9QX4WxhHbWaBm36KHpHR"

# Initialise session state variables
if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []
if 'messages' not in st.session_state:
    st.session_state['messages'] = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]
if 'model_name' not in st.session_state:
    st.session_state['model_name'] = []
if 'cost' not in st.session_state:
    st.session_state['cost'] = []
if 'total_tokens' not in st.session_state:
    st.session_state['total_tokens'] = []
if 'total_cost' not in st.session_state:
    st.session_state['total_cost'] = 0.0

# Sidebar - let user choose model, show total cost of current conversation, and let user clear the current conversation
st.sidebar.title("Sidebar")
model_name = st.sidebar.radio("Choose a model:", ("GPT-3.5", "GPT-4"))
counter_placeholder = st.sidebar.empty()
counter_placeholder.write(f"Total cost of this conversation: ${st.session_state['total_cost']:.5f}")
clear_button = st.sidebar.button("Clear Conversation", key="clear")

# Map model names to OpenAI model IDs
if model_name == "GPT-3.5":
    model = "gpt-3.5-turbo"
else:
    model = "gpt-4"

# reset everything
if clear_button:
    st.session_state['generated'] = []
    st.session_state['past'] = []
    st.session_state['messages'] = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]
    st.session_state['number_tokens'] = []
    st.session_state['model_name'] = []
    st.session_state['cost'] = []
    st.session_state['total_cost'] = 0.0
    st.session_state['total_tokens'] = []
    counter_placeholder.write(f"Total cost of this conversation: ${st.session_state['total_cost']:.5f}")


# generate a response
def generate_response(prompt):
    st.session_state['messages'].append({"role": "user", "content": prompt})

    completion = client.chat.completions.create(model=model,
    messages=st.session_state['messages'])
    response = completion.choices[0].message.content
    st.session_state['messages'].append({"role": "assistant", "content": response})

    # print(st.session_state['messages'])
    total_tokens = completion.usage.total_tokens
    prompt_tokens = completion.usage.prompt_tokens
    completion_tokens = completion.usage.completion_tokens
    return response, total_tokens, prompt_tokens, completion_tokens


# container for chat history
response_container = st.container()
# container for text box
container = st.container()

with container:
    with st.form(key='my_form', clear_on_submit=True):
        user_input = st.text_area("You:", key='input', height=100)
        submit_button = st.form_submit_button(label='Send')

    if submit_button and user_input:
        output, total_tokens, prompt_tokens, completion_tokens = generate_response(user_input)
        st.session_state['past'].append(user_input)
        st.session_state['generated'].append(output)
        st.session_state['model_name'].append(model_name)
        st.session_state['total_tokens'].append(total_tokens)

        # from https://openai.com/pricing#language-models
        if model_name == "GPT-3.5":
            cost = total_tokens * 0.002 / 1000
        else:
            cost = (prompt_tokens * 0.03 + completion_tokens * 0.06) / 1000

        st.session_state['cost'].append(cost)
        st.session_state['total_cost'] += cost

if st.session_state['generated']:
    with response_container:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state["past"][i], is_user=True, key=str(i) + '_user')
            message(st.session_state["generated"][i], key=str(i))
            st.write(
                f"Model used: {st.session_state['model_name'][i]}; Number of tokens: {st.session_state['total_tokens'][i]}; Cost: ${st.session_state['cost'][i]:.5f}")
            counter_placeholder.write(f"Total cost of this conversation: ${st.session_state['total_cost']:.5f}")

