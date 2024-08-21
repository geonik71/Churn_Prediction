import streamlit as st
import pandas as pd
import numpy as np
import joblib



# Load pre-trained models
model_names = ["Logistic Regression", "KNeighborsClassifier", "Decision Tree","Support Vector Machine(SVM)" ,"Random Forest Classifier", "XGBoost Classifier"]
models = {
    "Logistic Regression": joblib.load('LogisticRegression_best_model.joblib'),
    "KNeighborsClassifier": joblib.load('KNeighborsClassifier_best_model.joblib'),
    "Decision Tree": joblib.load('DecisionTreeClassifier_best_model.joblib'),
    "Support Vector Machine(SVM)": joblib.load('SVC_best_model.joblib'),
    "Random Forest Classifier": joblib.load('RandomForestClassifier_best_model.joblib'),
    "XGBoost Classifier": joblib.load('XGBClassifier_best_model.joblib')
}


# Streamlit app interface
# Introduction Text
st.title("Telco Customer Churn")
st.markdown('<hr class="title-line">', unsafe_allow_html=True)
st.image('https://miro.medium.com/v2/resize:fit:1400/0*8Iu_eymr6eR-YuQw')
st.header("Introduction to the Telco Customer Churn Dataset")

st.markdown("""
<div style="text-align: justify;">
The <li><strong>Telco Customer Churn dataset</strong> is a popular dataset used for predicting customer churn in the telecommunications industry. 
Churn refers to the phenomenon where customers stop using a service or product. Understanding and predicting customer churn 
is crucial for businesses as retaining existing customers is often more cost-effective than acquiring new ones.
</div>
""", unsafe_allow_html=True)

st.subheader("Dataset Overview")

st.markdown("""
<div style="text-align: justify;">
The dataset includes information on a diverse set of customers, capturing various attributes that can influence whether a 
customer is likely to churn. These attributes range from customer demographics to services they have subscribed to, as well 
as their billing and payment information. The dataset contains both categorical and numerical data, making it an ideal candidate 
for various machine learning models.
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Key Features")
    st.markdown("""
    <div style="text-align: justify;">
    <ul>
        <li><strong>Customer Demographics:</strong> Information such as gender, age, and more.</li>
        <li><strong>Services Subscribed:</strong> Details about the services customers have signed up for, including phone service, internet service, and more.</li>
        <li><strong>Account Information:</strong> Data on the length of the customer's relationship with the company, contract type, paperless billing, and payment methods.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.subheader("Purpose of the Analysis")
    st.markdown("""
    <div style="text-align: justify;">
    The primary objective of analyzing the Telco Customer Churn dataset is to build predictive models that can identify customers 
    who are at risk of leaving the service. By understanding the factors contributing to churn, telecom companies can take proactive 
    measures to improve customer retention and enhance customer satisfaction.
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.subheader("Why This Dataset?")
    st.markdown("""
    <div style="text-align: justify;">
    This dataset is widely used in machine learning and data science communities due to its rich feature set and practical business relevance. 
    It provides an excellent opportunity to apply various data analysis, visualization, and machine learning techniques to a real-world problem.
    </div>
    """, unsafe_allow_html=True)


st.header('Select Your Features')
# Features 
col4, col5, col6 = st.columns(3)


df = pd.read_csv('IT_customer_churn.csv')

PaymentMethod = list(df['PaymentMethod'].astype(str).unique())
PaymentMethod.sort()

InternetService = list(df['InternetService'].astype(str).unique())
InternetService.sort()

Contract = list(df['Contract'].astype(str).unique())
Contract.sort()


with col4:

    gender_dic = {0: "Male", 1: "Female"}
    Gender_select= st.radio('Gender', options=list(gender_dic.keys()), format_func=lambda x: gender_dic[x])

    dependance_dic = {0: "No", 1: "Yes"}
    Dependance_select= st.radio('Dependance', options=list(dependance_dic.keys()), format_func=lambda x: dependance_dic[x])

    online_backup_dic = {0: "No", 1: "Yes"}
    online_backup_select= st.radio('Online Backup', options=list(online_backup_dic.keys()), format_func=lambda x: online_backup_dic[x])

    streaming_tv_dic = {0: "No", 1: "Yes"}
    streaming_tv_select= st.radio('Streamin TV', options=list(streaming_tv_dic.keys()), format_func=lambda x: streaming_tv_dic[x])   

    paperless_billing_dic = {0: "No", 1: "Yes"}
    paperless_billing_select= st.radio('Paperless Billing TV', options=list(paperless_billing_dic.keys()), format_func=lambda x: paperless_billing_dic[x])  



with col5:

    senior_dic = {0: "No", 1: "Yes"}
    Senior_select= st.radio('Senior', options=list(senior_dic.keys()), format_func=lambda x: senior_dic[x])

    PhoneService_dic = {0: "No", 1: "Yes"}
    PhoneService_select= st.radio('Phone Service', options=list(PhoneService_dic.keys()), format_func=lambda x: PhoneService_dic[x])


    device_protection_dic = {0: "No", 1: "Yes"}
    device_protection_select= st.radio('Device Protection', options=list(device_protection_dic.keys()), format_func=lambda x: device_protection_dic[x])

    streaming_movies_dic = {0: "No", 1: "Yes"}
    streaming_movies_select= st.radio('Streaming Movies', options=list(streaming_movies_dic.keys()), format_func=lambda x: streaming_movies_dic[x]) 


with col6:

    partner_dic = {0: "No", 1: "Yes"}
    Partner_select= st.radio('Partner', options=list(partner_dic.keys()), format_func=lambda x: partner_dic[x])

    MultipleLines_dic = {0: "No", 1: "Yes"}
    MultipleLines_select= st.radio('Multiple Lines', options=list(MultipleLines_dic.keys()), format_func=lambda x: MultipleLines_dic[x])

    online_security_dic = {0: "No", 1: "Yes"}
    online_security_select= st.radio('Monline security', options=list(online_security_dic.keys()), format_func=lambda x: online_security_dic[x])

    tech_support_dic = {0: "No", 1: "Yes"}
    tech_support_select= st.radio('Tech Support', options=list(tech_support_dic.keys()), format_func=lambda x: tech_support_dic[x])



tenure_select = st.number_input("Tenure (in months)", min_value=0, max_value=100, value=12, step=1)
internet_service_select = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
contract_select = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
payment_method_select = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])
monthly_charges_select = st.number_input("Monthly Charges", min_value=0.0, max_value=200.0, value=50.0, step=0.1)
total_charges_select = st.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=500.0, step=1.0)

selected_model_name = st.selectbox("Choose Your Model According :", model_names)


row_list=[Gender_select, Senior_select, Partner_select, Dependance_select, tenure_select,
          PhoneService_select, MultipleLines_select, online_security_select, online_backup_select,
          device_protection_select, tech_support_select, streaming_tv_select, streaming_movies_select,
          paperless_billing_select, monthly_charges_select, total_charges_select,
          
 ]


PaymentMethod_vector=[1 if p == payment_method_select else 0 for p in PaymentMethod]
row_list.extend(PaymentMethod_vector)

InternetService_vector=[1 if i == internet_service_select else 0 for i in InternetService]
row_list.extend(InternetService_vector)

Contract_vector=[1 if c == contract_select else 0 for c in Contract]
row_list.extend(Contract_vector)


# Convert to numpy array and reshape
row_array = np.array(row_list).reshape(1, -1)

# Prediction
selected_model = models[selected_model_name]
prediction = selected_model.predict(row_array)

if st.button('Predict'):
    if prediction[0] == 1:
        st.error("❌❌❌❌Unfortunately the customer has decided to leave.❌❌❌❌")
    else:
        st.success('✅✅✅✅The customer has chosen to stay✅✅✅✅')

