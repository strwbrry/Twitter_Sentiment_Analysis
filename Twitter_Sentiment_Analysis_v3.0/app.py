import streamlit as st
import pandas as pd
import helper_functions as hf

st.set_page_config(
    page_title="Twitter Sentiment/Emotional Distress Analyzer", page_icon="📊", layout="wide" 
)

adjust_top_pad = """
    <style>
        div.block-container {padding-top:1rem;}
    </style>
    """
st.markdown(adjust_top_pad, unsafe_allow_html=True)

def search_callback():
    try:
        st.session_state.df = hf.get_latest_tweet_df(
            st.session_state.username, st.session_state.number_of_tweets
        )      
        st.session_state.df = hf.predict_sentiment(st.session_state.df)
    except:
        st.toast('Warning!', icon="⚠️")
    # print(hf.get_latest_tweet_df(st.session_state.username, st.session_state.number_of_tweets))  


with st.sidebar:
    st.title("Twitter Sentiment Analyzer")

    st.markdown(
        """
        <div style="text-align: justify;">
            This app performs sentiment analysis on the latest tweets based on 
            the user. The app can only predict positive or 
            negative sentiment.
            Only English and Filipino tweets are supported.
        </div>
        """,
        unsafe_allow_html=True
    )

    with st.form(key="search_form"):
        st.subheader("Search Parameters")

        st.text_input("Search User", key="username")

        st.slider("Number of tweets", min_value=5, max_value=500, key="number_of_tweets")

        st.form_submit_button(label="Fetch Tweets", on_click=search_callback)

        st.markdown("Note: it may take a while to load the results, especially with large number of tweets")


try:
    if "df" in st.session_state:            
        # function to make the dashboard
        def make_dashboard(tweet_df):
            # show the dataframe containing the tweets and their sentiment  
            table_plot = hf.plot_table(tweet_df)
            table_plot.update_layout(width=2000, height=500)
            st.plotly_chart(table_plot, theme=None, use_container_width=True)

            #Second Row
            col1, col2 = st.columns([30, 70])
            with col1:
                # plot the sentiment distribution
                sentiment_plot = hf.plot_sentiment(tweet_df)
                sentiment_plot.update_layout(height=500, title_x=0.5)
                st.plotly_chart(sentiment_plot, theme=None, use_container_width=True)

            with col2:
                probability_plot = hf.plot_probability(tweet_df)
                probability_plot.update_layout(height=500, title_x=0.5)
                st.plotly_chart(probability_plot, theme=None, use_container_width=True)

        # increase the font size of text inside the tab
        adjust_tab_font = """
        <style>
        button[data-baseweb="tab"] > div[data-testid="stMarkdownContainer"] > p {
            font-size: 20px;
        }
        </style>
        """
        st.write(adjust_tab_font, unsafe_allow_html=True)

        # create 3 tabs for all, positive, and negative tweets
        tab1, tab2, tab3 = st.tabs(["All", "Suicide", "Non-Suicide"])
        with tab1:
            # make dashboard for all tweets
            tweet_df = st.session_state.df
            make_dashboard(tweet_df)

        with tab2:
            # make dashboard for tweets with suicide sentiment
            tweet_df = st.session_state.df.query("Sentiment == 'Suicide'")
            make_dashboard(tweet_df)

        with tab3:
            # make dashboard for tweets with non-suicide sentiment
            tweet_df = st.session_state.df.query("Sentiment == 'Non-suicide'")
            make_dashboard(tweet_df)
            
except Exception as ex:
    st.error('Process Encountered an error. One of these might be the reason:', icon="🚨")
    st.info('Twitter user does not exist.', icon="ℹ️")
    st.info('Twitter profile is in private', icon="ℹ️")
    st.info(ex, icon="ℹ️")