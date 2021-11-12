from contextlib import nullcontext
from pycaret.classification import load_model, predict_model
import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path
from PIL import Image
import os


class StreamlitApp:
    
    def __init__(self):
        pathing = Path(__file__).parents[1] / '/app/ml2tmdb_03/box_office'
        self.model = load_model(pathing) 
        self.save_fn = 'path.csv'
        

    def predict(self, input_data):
        return predict_model(self.model, data=input_data)

    def store_prediction(self, output_df):
        if os.path.exists(self.save_fn):
            safe_df = pd.read_csv(self.save_fn)
            safe_df = safe_df.append(output_df, ignore_index=True)
            safe_df.to_csv(self.save_fn, index=False)

        else:
            output_df.to_csv(self.save_fn, index=False)
    
    def run(self):
        st.title('BOX OFFICE REVENUE PREDICTION')

        if True:
            genre = st.selectbox('Select genres:', ['Action', 'Comedy', 'Romance', 'Sci-fi', 'Drama', 'Fantasy', 'Horror', 'Mystery', 'Thriller', 'Western', 'Adventure', 'Documentary'])
            budget = st.number_input('Budget : ', min_value=1, max_value=10000000, value=10000)
            runtime = st.number_input('Movie RunTime', min_value=1, max_value=1000, value=1)
            popularity = st.number_input('Popularity Rating (0-100))', min_value=0, max_value=100, value=0)
            original_language = st.selectbox('Language', ['en', 'hi','ko','sr','fr','it','nl','zh','es','cs','ta','cn','ru','tr','ja','fa','sv','de','te','pt','mr'])

            output =''
            input_dict = {'budget':budget, 'popularity':popularity, 'runtime':runtime, 'genres':genre,
                        'original_language':original_language, }
            input_df = pd.DataFrame(input_dict, index=[0])

            if st.button('Predict revenue'):
                output = self.predict(input_df)
                self.store_prediction(output)
                output = (output['Label'][0])/1000000  
                st.success('Predicted revenue: {revenue:,.2f}$ million'.format(revenue = output).replace(",", " "))


sa = StreamlitApp()
sa.run()