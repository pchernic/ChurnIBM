import pickle
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Churn Prediction", page_icon="img/dnc.webp")
st.title("Telco Churn prediction")

st.image('img/customer_churn.jpeg',use_column_width=True)

st.markdown("""
<style>
    .justified-text {
        text-align: justify;
        max-width: 800px;  /* Adjust the max-width as needed */
        margin: auto;  /* Center the content */
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="justified-text">
    Churn prediction stands as a pivotal element for any business committed to customer retention. Within the framework of a machine learning prediction application, churn prediction encompasses the process of discerning customers with a high likelihood of discontinuing the use of a company's products or services. Leveraging historical data and advanced machine learning algorithms, the application scrutinizes patterns and behaviors exhibited by previous customers who have undergone churn. Subsequently, this accumulated knowledge is applied to pinpoint customers who are most inclined to discontinue their engagement in the future. Precise anticipation of customer churn empowers a company to proactively implement strategies, retaining valuable customers and mitigating the adverse impact of churn on its financial performance.
    The utilization of this machine learning application for churn prediction not only enables companies to save time and resources but also empowers them to make well-informed and data-driven business decisions.
</div>
""", unsafe_allow_html=True)

# -- Model -- #
with open('models/model.pkl', 'rb') as file:
    model = pickle.load(file)

data = st.file_uploader('Upload your file')
if data:
    df_input = pd.read_csv(data)
    df_output = df_input.assign(
        churn=model.predict(df_input),
        churn_probability=model.predict_proba(df_input)[:,1]
        )

    st.markdown('Churn prediction:')
    st.write(df_output)
    st.download_button(
        label='Download CSV', data=df_output.to_csv(index=False).encode('utf-8'),
        mime='text/csv', file_name='churn_prediction.csv'
        )
