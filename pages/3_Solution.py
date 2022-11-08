import streamlit as st

st.set_page_config(page_title = 'Solution', page_icon=':iphone:', layout='wide')
st.title('Solution')
st.write("""
        Berdasarkan hasil analisis data pelanggan telco terdapat 2 solusi untuk mempertahankan pelanggan telco, yaitu:
        1. kita bisa memberikan perlakuan khusus (seperti menawarkan paket khusus) kepada pelanggan yang diprediksi akan menjadi churn oleh machine learning yang telah dibangun.
        2. Meeningkatkan kualitas pelayanan yang diberikan kepada customer.
""")
