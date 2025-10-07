#Imports
import pandas as pd
import streamlit as st
import joblib
import os
# App Heading
st.title("Employee-Retention-Prediction")
st.write("Fill your details")
# Form Inputs
with st.form("Employee-Retention_Form"):
    city = st.selectbox("city", ['city_103' 'city_40' 'city_21' 'city_115' 'city_162' 'city_176''city_160' 'city_46' 'city_61' 'city_114' 'city_13' 'city_159' 'city_102''city_67' 'city_100' 'city_16' 'city_71' 'city_104' 'city_64' 'city_101'
          'city_83' 'city_105' 'city_73' 'city_75' 'city_41' 'city_11' 'city_93' 'city_90' 'city_36' 'city_20' 'city_57' 'city_152' 'city_19' 'city_65''city_74' 'city_173' 'city_136' 'city_98' 'city_97' 'city_50' 'city_138''city_82' 'city_157' 'city_89' 'city_150' 'city_70' 'city_175' 'city_94'
           'city_28' 'city_59' 'city_165' 'city_145' 'city_142' 'city_26' 'city_12''city_37' 'city_43' 'city_116' 'city_23' 'city_99' 'city_149' 'city_10''city_45' 'city_80' 'city_128' 'city_158' 'city_123' 'city_7' 'city_72''city_106' 'city_143' 'city_78' 'city_109' 'city_24' 'city_134' 'city_48''city_144' 'city_91' 'city_146' 'city_133' 'city_126' 'city_118' 'city_9''city_167' 'city_27' 'city_84' 'city_54' 'city_39' 'city_79' 'city_76''city_77' 'city_81' 'city_131' 'city_44' 'city_117' 'city_155' 'city_33''city_141' 'city_127' 'city_62' 'city_53' 'city_25' 'city_2' 'city_69''city_120' 'city_111' 'city_30' 'city_1' 'city_140' 'city_179' 'city_55'
            'city_14' 'city_42' 'city_107' 'city_18' 'city_139' 'city_180' 'city_166''city_121' 'city_129' 'city_8' 'city_31' 'city_171'])
   
    city_development_index = st.number_input("city_development_index",min_value=0.0)
    gender = st.selectbox("gender", ["Male","Female","Other"])
    relevent_experience = st.selectbox("relevent_experience", ['Has relevent experience','No relevent experience'])
    enrolled_university = st.selectbox("enrolled_university", ['no_enrollment','Full time course','Part time course'])
    education_level = st.selectbox("education_level", ['Graduate','Masters','High School','Phd', 'Primary School'])
    major_discipline = st.selectbox("major_discipline", ['STEM','Business Degree','Arts','Humanities', 'No Major','Other'])
    experience = st.number_input("experience",min_value=0.0)
    company_size = st.number_input("company_size")
    company_type = st.selectbox("company_type", ['Pvt Ltd','Funded Startup', 'Early Stage Startup','Public Sector','NGO','Other'])
    last_new_job = st.selectbox("last_new_job", ['1','2','3','4','5','never'])
    training_hours = st.number_input("training_hours",min_value=0)
   
 
    Submitted = st.form_submit_button("Submit")
    # Create DataFrame
if Submitted:
    input_data = pd.DataFrame([{
        "city": city,
        "gender": gender,
        "relevent_experience": relevent_experience,
        "enrolled_university": enrolled_university ,
        "education_level": education_level,
        "major_discipline": major_discipline,
        "experience": experience,
        "company_size": company_size,
        "company_type": company_type,
        "last_new_job": last_new_job,
        "training_hours":training_hours
     }])
    
    prediction = pipeline.predict(input_data)[0]
    probability = pipeline.predict_proba(input_data)[0][1]
    st.success(f"Prediction: {'Yes' if prediction == 1 else 'No'}")
    st.write(f"Employee-Retention-Prediction Probability : {probability*100:.2f}%")

# Load pipeline
model_path = "lightgbm_model.pkl"
if os.path.exists(model_path):
    pipeline = joblib.load(model_path)
    st.success("MODEL LOADED SUCCESSFULLY")
else:
    st.warning(f"Model file {model_path} does not exist")