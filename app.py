import streamlit as st
from streamlit_option_menu import option_menu
import json
import os
from src.pipeline.prediction_pipeline import PredictPipeline

st.markdown('''
    <style>
    .stApp {
    background-image: url("https://assets-global.website-files.com/640ac6692bb2f64f1e589f16/645bb3607b4a0b25157b61c6_nlp-text-summarization.jpg");
    background-size: cover;
    }
    [data-testid=stSidebar] {
        background-color: #c2f2f1;
    }
    div[data-baseweb="select"] > div {
    background-color: #c2f2f1 ;
    }
    </style>
    ''',
    unsafe_allow_html=True
)

with st.sidebar:
            selected = option_menu(
            menu_title=None,  # required
            options=["Home", "Summarizer", "About"],  # required
            icons=["house", "body-text", "info-circle"],  # optional
            menu_icon="cast",  # optional
            default_index=0,  # optional
            orientation="horizontal",
            styles={
                "container": {"padding": "0!important", "background-color": "#d7f7c8"},
                "icon": {"color": "orange", "font-size": "25px"},
                "nav-link": {
                    "font-size": "25px",
                    "text-align": "left",
                    "margin": "0px",
                    "--hover-color": "#eee",
                },
                "nav-link-selected": {"background-color": "green"},
            },
        )
            
if selected == "Home":
    mark='<p style="font-family:Marker Felt; color:#a86a58; font-size: 42px;">WelCome To Multilingual News Summarizer</p>'
    st.markdown(mark, unsafe_allow_html=True)

if selected == "Summarizer":
    st.markdown("""
    <style>
    .stTextArea [data-baseweb=base-input] {
        background-image: linear-gradient(140deg, rgb(54, 36, 31) 0%, rgb(121, 56, 100) 50%, rgb(106, 117, 25) 75%);
        -webkit-text-fill-color: white;
    }

    .stTextArea [data-baseweb=base-input] [disabled=""]{
        background-image: linear-gradient(45deg, red, purple, red);
        -webkit-text-fill-color: gray;
    }

    </style>
    """,unsafe_allow_html=True)
    mark='<p style="font-family:cursive; color:#decf4e; font-size: 22px;">Please Chooose the language and enter the news you want to summarize</p>'
    st.markdown(mark, unsafe_allow_html=True)
    lang_dict_path=os.path.join("artifacts","lang_dict.json")
    with open(lang_dict_path,'r') as file_obj:
            lang_dict=json.load(file_obj)
    file_obj.close()

    language = st.selectbox(":red[Choose Language]", lang_dict.keys())
    raw_language=lang_dict[language]
    summary="Summarized news will be Showing Here."
    c1,c2= st.columns(2)
    with c1:
        news=st.text_area(
                        label=":books:",
                        value="Your Text Here",
                        height= 200,
                        )
        if st.button(label="Summarize"):
                progress=st.progress(0)
                placeholder = st.empty()
                progress.progress(10)
                placeholder.text("Loading Models............")
                text_translation_model,news_summary_model,eng_tokenizer=PredictPipeline.load_models()
                if language == "English":
                        progress.progress(50)
                        placeholder.text("Summarizing News...........")
                        summary=PredictPipeline.summary_genaretor(news,news_summary_model)
                        progress.progress(100)
                        placeholder.text("Task Completed")
                else:
                    placeholder.text(f"Translating {language} to Model language.....")
                    translate=PredictPipeline.other_to_eng(news,raw_language,text_translation_model)
                    progress.progress(60)
                    placeholder.text("Summarizing News.........")
                    raw_summary=PredictPipeline.summary_genaretor(translation=translate[0],news_summary_model=news_summary_model)
                    progress.progress(80)
                    placeholder.text(f"Translating Back to {language} language....")
                    summary = PredictPipeline.eng_to_other(raw_summary,raw_language,eng_tokenizer,text_translation_model)
                    progress.progress(100)
                    placeholder.text("Task Completed")
    with c2:
        text=st.text_area(
                            label=":orange_book:",
                                    value=summary,
                                    height= 200,
                                )



if selected == "About":
    with open("about.txt","r") as file_obj:
          data=file_obj.read()
    mark=f'''<p style="font-family:cursive; color:#decf4e; font-size: 22px;">{data}</p>'''
    st.markdown(mark, unsafe_allow_html=True)
