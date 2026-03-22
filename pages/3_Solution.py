import streamlit as st

st.set_page_config(page_title = 'Solution', page_icon=':iphone:', layout='wide')
st.title('Solution')
tab1, tab2 = st.tabs(["English", "Bahasa Indonesia"])

with tab1:
        st.write("""
                Based on the results of the telco customer data analysis, there are 2 solutions to retain telco customers:
                1. We can provide special treatment (such as offering special packages) to customers who are predicted to churn by the machine learning model that has been developed.
                2. Improve the quality of service provided to customers.
        """)

with tab1:
        st.write("""
                Berdasarkan hasil analisis data pelanggan telco terdapat 2 solusi untuk mempertahankan pelanggan telco, yaitu:
                1. kita bisa memberikan perlakuan khusus (seperti menawarkan paket khusus) kepada pelanggan yang diprediksi akan menjadi churn oleh machine learning yang telah dibangun.
                2. Meningkatkan kualitas pelayanan yang diberikan kepada customer.
        """)
