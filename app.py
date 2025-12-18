import validators, streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader

## Streamlit APP
st.set_page_config(
    page_title="LangChain: Summarize Text From YT or Website",
    page_icon="🦜"
)
st.title("🦜 LangChain: Summarize Text From YT or Website")
st.subheader("Summarize URL")

## Sidebar
with st.sidebar:
    groq_api_key = st.text_input("Groq API Key", value="", type="password")

generic_url = st.text_input("URL", label_visibility="collapsed")

## LLM
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    groq_api_key=groq_api_key
)

## Prompt
prompt_template = """
Provide a summary of the following content in about 300 words:

{context}
"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context"]
)

# Runnable = Stuff Chain
chain = prompt | llm

if st.button("Summarize the Content from YT or Website"):

    if not groq_api_key.strip() or not generic_url.strip():
        st.error("Please provide the information to get started")

    elif not validators.url(generic_url):
        st.error("Please enter a valid URL (YouTube or Website)")

    else:
        try:
            with st.spinner("Loading and summarizing..."):

                # Load documents
                if "youtube.com" in generic_url:
                    loader = YoutubeLoader.from_youtube_url(
                        generic_url,
                        add_video_info=True
                    )
                else:
                    loader = UnstructuredURLLoader(
                        urls=[generic_url],
                        ssl_verify=False,
                        headers={
                            "User-Agent": (
                                "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) "
                                "AppleWebKit/537.36 (KHTML, like Gecko) "
                                "Chrome/116.0.0.0 Safari/537.36"
                            )
                        }
                    )

                docs = loader.load()

                # Convert documents → single text (stuffing)
                combined_text = "\n\n".join(d.page_content for d in docs)

                # Invoke LLM
                result = chain.invoke({"context": combined_text})

                st.success(result.content)

        except Exception as e:
            st.exception(e)
